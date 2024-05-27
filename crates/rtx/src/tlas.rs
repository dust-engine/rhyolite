use std::{
    collections::BTreeMap,
    ffi::CString,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
};

use bevy::{
    app::{App, Plugin, PostUpdate},
    ecs::{
        component::Component,
        entity::Entity,
        query::{
            Added, ArchetypeFilter, Or, QueryFilter, QueryItem, ReadOnlyQueryData, With, Without,
        },
        removal_detection::RemovedComponents,
        schedule::{IntoSystemConfigs, SystemSet},
        system::{
            Commands, In, Query, Res, ResMut, Resource, StaticSystemParam, SystemParam,
            SystemParamItem,
        },
        world::FromWorld,
    },
    math::{Mat4, Quat, Vec3, Vec4},
    utils::tracing,
};
use rhyolite::{
    ash::{khr::acceleration_structure::Meta as AccelerationStructureExt, vk},
    commands::{
        BatchCopy, CommonCommands, ComputeCommands, ResourceTransitionCommands, TransferCommands,
    },
    cstr,
    debug::DebugObject,
    ecs::{Barriers, IntoRenderSystemConfigs, RenderCommands, RenderRes},
    staging::{StagingBelt, StagingBeltSuballocation},
    Access, Allocator, Buffer, BufferArray, BufferLike, Device, HasDevice, Instance,
};

use crate::{AccelStruct, SbtIndex};

pub struct TLASInstanceData<'a> {
    ty: vk::AccelerationStructureMotionInstanceTypeNV,
    dirty: bool,
    host_build: bool,
    data: &'a mut MaybeUninit<vk::AccelerationStructureInstanceKHR>,
    motion_data: Option<&'a mut MaybeUninit<vk::AccelerationStructureMotionInstanceNV>>,
}
#[derive(Clone)]
pub struct SRTTransform {
    pub scale: Vec3,
    pub shear: Vec3,
    pub pivot: Vec3,
    pub rotation: Quat,
    pub translation: Vec3,
}
impl From<SRTTransform> for Mat4 {
    fn from(_value: SRTTransform) -> Self {
        todo!()
    }
}
impl From<SRTTransform> for vk::SRTDataNV {
    fn from(transform: SRTTransform) -> Self {
        Self {
            sx: transform.scale.x,
            sy: transform.scale.y,
            sz: transform.scale.z,
            a: transform.shear.x,
            b: transform.shear.y,
            c: transform.shear.z,
            pvx: transform.pivot.x,
            pvy: transform.pivot.y,
            pvz: transform.pivot.z,
            qx: transform.rotation.x,
            qy: transform.rotation.y,
            qz: transform.rotation.z,
            qw: transform.rotation.w,
            tx: transform.translation.x,
            ty: transform.translation.y,
            tz: transform.translation.z,
        }
    }
}
impl TLASInstanceData<'_> {
    pub fn set_transform(&mut self, transform: Mat4) {
        if self.dirty {
            assert_eq!(
                self.ty,
                vk::AccelerationStructureMotionInstanceTypeNV::STATIC,
                "Instance type cannot be changed at this point"
            );
        }
        self.ty = vk::AccelerationStructureMotionInstanceTypeNV::STATIC;
        let slice = &transform.transpose().to_cols_array()[0..12];
        unsafe {
            self.data
                .assume_init_mut()
                .transform
                .matrix
                .clone_from_slice(slice);
        }
        if let Some(motion) = &mut self.motion_data {
            unsafe {
                motion.assume_init_mut().ty = vk::AccelerationStructureMotionInstanceTypeNV::STATIC;
                motion
                    .assume_init_mut()
                    .data
                    .static_instance
                    .transform
                    .matrix
                    .clone_from_slice(slice);
            }
        }
    }
    pub fn set_matrix_motion(&mut self, from: Mat4, to: Mat4) {
        let Some(motion_data) = &mut self.motion_data else {
            self.set_transform(to);
            return;
        };
        if self.dirty {
            assert_eq!(
                self.ty,
                vk::AccelerationStructureMotionInstanceTypeNV::MATRIX_MOTION,
                "Instance type cannot be changed at this point"
            );
        }
        self.ty = vk::AccelerationStructureMotionInstanceTypeNV::MATRIX_MOTION;
        let from = &from.transpose().to_cols_array()[0..12];
        let to = &to.transpose().to_cols_array()[0..12];
        unsafe {
            self.data
                .assume_init_mut()
                .transform
                .matrix
                .clone_from_slice(from);

            motion_data
                .assume_init_mut()
                .data
                .matrix_motion_instance
                .transform_t0
                .matrix
                .clone_from_slice(from);
            motion_data
                .assume_init_mut()
                .data
                .matrix_motion_instance
                .transform_t1
                .matrix
                .clone_from_slice(to);
        }
    }
    pub fn set_srt_motion(&mut self, from: SRTTransform, to: SRTTransform) {
        let mat: Mat4 = to.clone().into();
        let Some(motion_data) = &mut self.motion_data else {
            self.set_transform(mat);
            return;
        };
        if self.dirty {
            assert_eq!(
                self.ty,
                vk::AccelerationStructureMotionInstanceTypeNV::SRT_MOTION,
                "Instance type cannot be changed at this point"
            );
        }
        self.ty = vk::AccelerationStructureMotionInstanceTypeNV::SRT_MOTION;
        let slice = &mat.transpose().to_cols_array()[0..12];
        unsafe {
            self.data
                .assume_init_mut()
                .transform
                .matrix
                .clone_from_slice(slice);
            motion_data
                .assume_init_mut()
                .data
                .srt_motion_instance
                .transform_t0 = from.into();
            motion_data
                .assume_init_mut()
                .data
                .srt_motion_instance
                .transform_t1 = to.into();
        }
    }
    pub fn set_custom_index_and_mask(&mut self, custom_index: u32, mask: u8) {
        assert_eq!(
            custom_index & !0xffffff,
            0,
            "Custom index must be in the range 0 ..= 2^24"
        );
        if let Some(motion) = &mut self.motion_data {
            unsafe {
                let dst = match self.ty {
                    vk::AccelerationStructureMotionInstanceTypeNV::STATIC => {
                        &mut motion
                            .assume_init_mut()
                            .data
                            .static_instance
                            .instance_custom_index_and_mask
                    }
                    vk::AccelerationStructureMotionInstanceTypeNV::MATRIX_MOTION => {
                        &mut motion
                            .assume_init_mut()
                            .data
                            .matrix_motion_instance
                            .instance_custom_index_and_mask
                    }
                    vk::AccelerationStructureMotionInstanceTypeNV::SRT_MOTION => {
                        &mut motion
                            .assume_init_mut()
                            .data
                            .srt_motion_instance
                            .instance_custom_index_and_mask
                    }
                    _ => panic!(),
                };
                *dst = vk::Packed24_8::new(custom_index, mask);
            }
        }
        unsafe {
            self.data.assume_init_mut().instance_custom_index_and_mask =
                vk::Packed24_8::new(custom_index, mask);
        }
        self.dirty = true;
    }
    pub fn set_sbt_offset_and_flags(&mut self, offset: u32, flags: vk::GeometryInstanceFlagsKHR) {
        assert_eq!(
            offset & !0xffffff,
            0,
            "SBT offset must be in the range 0 ..= 2^24"
        );
        if let Some(motion) = &mut self.motion_data {
            let dst = unsafe {
                match self.ty {
                    vk::AccelerationStructureMotionInstanceTypeNV::STATIC => {
                        &mut motion
                            .assume_init_mut()
                            .data
                            .static_instance
                            .instance_shader_binding_table_record_offset_and_flags
                    }
                    vk::AccelerationStructureMotionInstanceTypeNV::MATRIX_MOTION => {
                        &mut motion
                            .assume_init_mut()
                            .data
                            .matrix_motion_instance
                            .instance_shader_binding_table_record_offset_and_flags
                    }
                    vk::AccelerationStructureMotionInstanceTypeNV::SRT_MOTION => {
                        &mut motion
                            .assume_init_mut()
                            .data
                            .srt_motion_instance
                            .instance_shader_binding_table_record_offset_and_flags
                    }
                    _ => panic!(),
                }
            };

            *dst = vk::Packed24_8::new(offset, flags.as_raw() as u8);
        }
        unsafe {
            self.data
                .assume_init()
                .instance_shader_binding_table_record_offset_and_flags =
                vk::Packed24_8::new(offset, flags.as_raw() as u8);
        }
        self.dirty = true;
    }
    /// Disable the current instance by setting its acceleration structure reference to 0.
    pub fn disable(&mut self) {
        unsafe {
            self.data.assume_init_mut().acceleration_structure_reference =
                vk::AccelerationStructureReferenceKHR { device_handle: 0 };
            if let Some(motion) = &mut self.motion_data {
                match self.ty {
                    vk::AccelerationStructureMotionInstanceTypeNV::STATIC => {
                        motion
                            .assume_init_mut()
                            .data
                            .static_instance
                            .acceleration_structure_reference
                            .device_handle = 0;
                    }
                    vk::AccelerationStructureMotionInstanceTypeNV::MATRIX_MOTION => {
                        motion
                            .assume_init_mut()
                            .data
                            .matrix_motion_instance
                            .acceleration_structure_reference
                            .device_handle = 0;
                    }
                    vk::AccelerationStructureMotionInstanceTypeNV::SRT_MOTION => {
                        motion
                            .assume_init_mut()
                            .data
                            .srt_motion_instance
                            .acceleration_structure_reference
                            .device_handle = 0;
                    }
                    _ => panic!(),
                }
            }
        }
    }
    pub fn set_blas(&mut self, accel_struct: &AccelStruct) {
        let reference = if self.host_build {
            vk::AccelerationStructureReferenceKHR {
                host_handle: accel_struct.raw,
            }
        } else {
            vk::AccelerationStructureReferenceKHR {
                device_handle: accel_struct.device_address,
            }
        };
        if let Some(motion) = &mut self.motion_data {
            unsafe {
                match self.ty {
                    vk::AccelerationStructureMotionInstanceTypeNV::STATIC => {
                        motion
                            .assume_init_mut()
                            .data
                            .static_instance
                            .acceleration_structure_reference = reference;
                    }
                    vk::AccelerationStructureMotionInstanceTypeNV::MATRIX_MOTION => {
                        motion
                            .assume_init_mut()
                            .data
                            .matrix_motion_instance
                            .acceleration_structure_reference = reference;
                    }
                    vk::AccelerationStructureMotionInstanceTypeNV::SRT_MOTION => {
                        motion
                            .assume_init_mut()
                            .data
                            .srt_motion_instance
                            .acceleration_structure_reference = reference;
                    }
                    _ => panic!(),
                }
            }
        }
        unsafe {
            self.data.assume_init_mut().acceleration_structure_reference = reference;
        }
    }
}

pub struct DefaultTLAS;

/// The index in the TLAS input buffer.
#[derive(Component)]
pub struct TLASIndex<T: Send + Sync + 'static> {
    pub index: u32,
    initialized: bool,
    _marker: std::marker::PhantomData<T>,
}

/// Trait for building top-level acceleration structures.
pub trait TLASBuilder: Send + Sync + 'static {
    type TLASType: Send + Sync + 'static = DefaultTLAS;

    /// Associated components to be passed.
    type QueryData: ReadOnlyQueryData;

    /// Additional filter for instances.
    type QueryFilter: ArchetypeFilter;

    /// Additional system entities to be passed.
    type Params: SystemParam;

    fn has_motion(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> bool {
        false
    }

    fn should_update(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> bool;
    fn instance(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
        dst: TLASInstanceData,
    );
}

#[derive(Resource)]
pub struct TLASDeviceBuildStore<T> {
    static_buffer: RenderRes<BufferArray<vk::AccelerationStructureInstanceKHR>>,

    /// We maintain both buffers at the same time because the scene may not be in motion at all times.
    /// When it's not in motion, we can use the static buffer to build the TLAS.
    motion_buffer: Option<RenderRes<BufferArray<vk::AccelerationStructureMotionInstanceNV>>>,
    has_motion: bool,
    free_entries: Vec<u32>,
    entity_map: BTreeMap<Entity, u32>,
    scratch_buffer: Option<RenderRes<Buffer>>,
    accel_struct: Option<RenderRes<AccelStruct>>,
    _marker: std::marker::PhantomData<T>,
}
impl<T> Deref for TLASDeviceBuildStore<T> {
    type Target = Option<RenderRes<AccelStruct>>;
    fn deref(&self) -> &Self::Target {
        &self.accel_struct
    }
}
impl<T> DerefMut for TLASDeviceBuildStore<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.accel_struct
    }
}
impl<T> FromWorld for TLASDeviceBuildStore<T> {
    fn from_world(world: &mut bevy::prelude::World) -> Self {
        let mut motion_buffer = None;
        let allocator = world.resource::<Allocator>().clone();
        if world
            .resource::<Device>()
            .feature::<vk::PhysicalDeviceRayTracingMotionBlurFeaturesNV>()
            .map(|f| f.ray_tracing_motion_blur == vk::TRUE)
            .unwrap_or(false)
        {
            motion_buffer = Some(BufferArray::new_resource(
                allocator.clone(),
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                16,
            ))
        }
        Self {
            static_buffer: RenderRes::new(BufferArray::new_resource(
                allocator,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                16,
            )),
            has_motion: false,
            motion_buffer: motion_buffer.map(RenderRes::new),
            entity_map: BTreeMap::new(),
            free_entries: Vec::new(),
            scratch_buffer: None,
            accel_struct: None,
            _marker: std::marker::PhantomData,
        }
    }
}

fn assign_index<B: TLASBuilder>(
    mut commands: Commands,
    new_instances: Query<(Entity, B::QueryData), (B::QueryFilter, Without<TLASIndex<B::TLASType>>)>,
    mut store: ResMut<TLASDeviceBuildStore<B::TLASType>>,
    mut removed: RemovedComponents<TLASIndex<B::TLASType>>,
) {
    for removed_entity in removed.read() {
        let index = store.entity_map.remove(&removed_entity).unwrap();
        if index != store.entity_map.len() as u32 {
            store.free_entries.push(index);
        }
    }
    for (entity, _data) in new_instances.iter() {
        assert!(!store.entity_map.contains_key(&entity));
        let index = store
            .free_entries
            .pop()
            .unwrap_or_else(|| store.entity_map.len() as u32);
        store.entity_map.insert(entity, index);
        commands.entity(entity).insert(TLASIndex::<B::TLASType> {
            index,
            initialized: false,
            _marker: Default::default(),
        });
    }
}

fn resize_buffer<B: Send + Sync + 'static>(
    mut commands: RenderCommands<'c'>,
    mut store: ResMut<TLASDeviceBuildStore<B>>,
) {
    if store.entity_map.is_empty() {
        return;
    }
    let new_capacity = store.entity_map.len() as u32;
    let old_buffer = store
        .static_buffer
        .try_swap(|buffer| buffer.realloc(new_capacity as usize).unwrap());
    if let Some(mut old_buffer) = old_buffer {
        commands
            .transition_resources()
            .transition(&mut old_buffer, Access::COPY_READ, true, ())
            .transition(&mut store.static_buffer, Access::COPY_WRITE, true, ());
        commands.copy_buffer(
            old_buffer.raw_buffer(),
            store.static_buffer.raw_buffer(),
            &[vk::BufferCopy {
                size: old_buffer.size(),
                ..Default::default()
            }],
        );
    }
    if let Some(motion) = store.motion_buffer.as_mut() {
        tracing::info!("Resizing TLAS motion input buffer to {}", new_capacity);
        let old_buffer = motion.try_swap(|buffer| buffer.realloc(new_capacity as usize).unwrap());
        if let Some(mut old_buffer) = old_buffer {
            commands
                .transition_resources()
                .transition(&mut old_buffer, Access::COPY_READ, true, ())
                .transition(motion, Access::COPY_WRITE, true, ());
            commands.copy_buffer(
                old_buffer.raw_buffer(),
                motion.raw_buffer(),
                &[vk::BufferCopy {
                    size: old_buffer.size(),
                    ..Default::default()
                }],
            );
        }
    }
}

fn extract_input_barrier<B: TLASBuilder>(
    In(mut barriers): In<Barriers>,
    mut store: ResMut<TLASDeviceBuildStore<B::TLASType>>,
) {
    // Inputs from extract may overlap regions from resize copy, so we need a barrier here
    barriers.transition(&mut store.static_buffer, Access::COPY_WRITE, true, ());
    if let Some(motion) = store.motion_buffer.as_mut() {
        barriers.transition(motion, Access::COPY_WRITE, true, ());
    }
}

fn extract_input<B: TLASBuilder>(
    mut commands: RenderCommands<'c'>,
    mut updated_instances: Query<(Entity, B::QueryData, &mut TLASIndex<B::TLASType>), B::QueryFilter>,
    mut staging_belt: ResMut<StagingBelt>,
    mut store: ResMut<TLASDeviceBuildStore<B::TLASType>>,
    mut params: StaticSystemParam<B::Params>,
) {
    let mut staging: rhyolite::staging::StagingBeltBatchJob<'_> = staging_belt.start(&mut commands);
    let mut job = BatchCopy::new(&mut commands);
    let mut has_motion = false;

    for (entity, data, mut index) in updated_instances.iter_mut() {
        if index.initialized && !B::should_update(&mut params, &data) {
            continue;
        }
        index.initialized = true;
        has_motion |= B::has_motion(&mut params, &data);

        let mut allocation = staging.allocate_item::<vk::AccelerationStructureInstanceKHR>();
        let mut motion_allocation = if has_motion {
            Some(staging.allocate_item::<vk::AccelerationStructureMotionInstanceNV>())
        } else {
            None
        }; // FIXME: Not optimal. will increase number of entries in regions.
        B::instance(
            &mut params,
            &data,
            TLASInstanceData {
                ty: vk::AccelerationStructureMotionInstanceTypeNV::STATIC,
                dirty: false,
                host_build: false,
                data: allocation.uninit_mut(),
                motion_data: motion_allocation
                    .as_mut()
                    .map(StagingBeltSuballocation::uninit_mut),
            },
        );
        job.copy_buffer(
            allocation.raw_buffer(),
            store.static_buffer.raw_buffer(),
            &[vk::BufferCopy {
                src_offset: allocation.offset(),
                dst_offset: index.index as u64
                    * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64,
                size: std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64,
            }],
        );
        if let Some(motion_allocation) = motion_allocation {
            job.copy_buffer(
                motion_allocation.raw_buffer(),
                store.motion_buffer.as_ref().unwrap().raw_buffer(),
                &[vk::BufferCopy {
                    src_offset: motion_allocation.offset(),
                    dst_offset: index.index as u64
                        * std::mem::size_of::<vk::AccelerationStructureMotionInstanceNV>() as u64,
                    size: std::mem::size_of::<vk::AccelerationStructureMotionInstanceNV>() as u64,
                }],
            );
        }
    }
    drop(job);
    store.has_motion |= has_motion;
    // TODO: if no motion for any instance, we build the TLAS without the motion flag.
}

/// Calculate build sizes and create the TLAS & scratch buffer.
fn prepare_tlas<B: Send + Sync + 'static>(
    allocator: Res<Allocator>,
    mut store: ResMut<TLASDeviceBuildStore<B>>,
) {
    if store.entity_map.is_empty() {
        return;
    }
    let geometry = vk::AccelerationStructureGeometryKHR {
        geometry_type: vk::GeometryTypeKHR::INSTANCES,
        geometry: vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR {
                data: vk::DeviceOrHostAddressConstKHR {
                    device_address: store.static_buffer.device_address(),
                },
                ..Default::default()
            },
        },
        ..Default::default()
    };
    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
        ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        mode: vk::BuildAccelerationStructureModeKHR::BUILD,
        geometry_count: 1,
        p_geometries: &geometry,
        ..Default::default()
    };
    let build_size = unsafe {
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        allocator
            .device()
            .extension::<AccelerationStructureExt>()
            .get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[store.entity_map.len() as u32],
                &mut sizes,
            );
        sizes
    };
    if let Some(accel_struct) = store.accel_struct.as_mut() {
        if accel_struct.size() < build_size.acceleration_structure_size {
            *accel_struct = RenderRes::new(
                AccelStruct::new_tlas(allocator.clone(), build_size.acceleration_structure_size)
                    .unwrap()
                    .with_name(CString::new(std::any::type_name::<B>()).unwrap().as_c_str()),
            );
        }
    } else {
        store.accel_struct = Some(RenderRes::new(
            AccelStruct::new_tlas(allocator.clone(), build_size.acceleration_structure_size)
                .unwrap()
                .with_name(CString::new(std::any::type_name::<B>()).unwrap().as_c_str()),
        ));
    };
    let scratch_offset_alignment: u32 = allocator
        .device()
        .physical_device()
        .properties()
        .get::<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>()
        .min_acceleration_structure_scratch_offset_alignment;

    if let Some(scratch_buffer) = store.scratch_buffer.as_mut() {
        if scratch_buffer.size() < build_size.build_scratch_size {
            *scratch_buffer = RenderRes::new(
                Buffer::new_resource(
                    allocator.clone(),
                    build_size.build_scratch_size,
                    scratch_offset_alignment as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )
                .unwrap()
                .with_name(cstr!("TLAS scratch buffer")),
            );
        }
    } else {
        store.scratch_buffer = Some(RenderRes::new(
            Buffer::new_resource(
                allocator.clone(),
                build_size.build_scratch_size,
                scratch_offset_alignment as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .unwrap()
            .with_name(cstr!("TLAS scratch buffer")),
        ));
    };
}

fn build_tlas_barrier<B: Send + Sync + 'static>(
    In(mut barriers): In<Barriers>,
    mut store: ResMut<TLASDeviceBuildStore<B>>,
) {
    if store.entity_map.is_empty() {
        return;
    }
    // Inputs from extract may overlap regions from resize copy, so we need a barrier here
    barriers.transition(
        &mut store.static_buffer,
        Access {
            access: vk::AccessFlags2::MEMORY_READ, // TODO: what's the optimal access here?
            stage: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
        },
        true,
        (),
    );
    if let Some(motion) = store.motion_buffer.as_mut() {
        barriers.transition(
            motion,
            Access {
                access: vk::AccessFlags2::MEMORY_READ, // TODO: what's the optimal access here?
                stage: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            },
            true,
            (),
        );
    }
    barriers.transition(
        store.accel_struct.as_mut().unwrap(),
        Access {
            access: vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
            stage: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
        },
        false,
        (),
    ); // TODO: make sure that this is caulling the correct impl
    barriers.transition(
        store.scratch_buffer.as_mut().unwrap(),
        Access {
            access: vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR
                | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
            stage: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
        },
        false,
        (),
    );
}

pub fn build_tlas<B: Send + Sync + 'static>(
    mut commands: RenderCommands<'c'>,
    mut store: ResMut<TLASDeviceBuildStore<B>>,
) {
    if store.entity_map.is_empty() {
        return;
    }
    let geometry = vk::AccelerationStructureGeometryKHR {
        geometry_type: vk::GeometryTypeKHR::INSTANCES,
        geometry: vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR {
                data: vk::DeviceOrHostAddressConstKHR {
                    device_address: store.static_buffer.device_address(),
                },
                ..Default::default()
            },
        },
        ..Default::default()
    };
    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
        ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        mode: vk::BuildAccelerationStructureModeKHR::BUILD,
        geometry_count: 1,
        p_geometries: &geometry,
        dst_acceleration_structure: store.accel_struct.as_ref().unwrap().raw,
        scratch_data: vk::DeviceOrHostAddressKHR {
            device_address: store.scratch_buffer.as_ref().unwrap().device_address(),
        },
        ..Default::default()
    };
    let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR {
        primitive_count: store.entity_map.len() as u32,
        ..Default::default()
    };
    commands.build_acceleration_structure(
        &[build_info],
        std::iter::once([build_range_info].as_slice()),
    );

    store.has_motion = false;
}

/// Building the TLAS on the device.
/// This plugin maintains a device local TLAS input buffer. On updates, it copies the updated entries to the buffer
/// using the staging belt.
///
/// We use a staging belt regardless of the GPU memory architecture because the TLAS input buffer updates need to happen
/// on device timeline.
pub struct TLASBuilderPlugin<T: TLASBuilder> {
    _marker: std::marker::PhantomData<T>,
}
impl<T: TLASBuilder> Default for TLASBuilderPlugin<T> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

#[derive(SystemSet, Hash, PartialEq, Eq, Clone, Copy, Debug)]
pub struct TLASBuilderSet;

impl<T: TLASBuilder> Plugin for TLASBuilderPlugin<T> {
    fn build(&self, _app: &mut App) {}
    fn finish(&self, app: &mut App) {
        if app
            .world
            .get_resource::<TLASDeviceBuildStore<T::TLASType>>()
            .is_none()
        {
            app.init_resource::<TLASDeviceBuildStore<T::TLASType>>();
            app.add_systems(
                PostUpdate,
                (
                    resize_buffer::<T::TLASType>,
                    prepare_tlas::<T::TLASType>,
                    (build_tlas::<T::TLASType>)
                        .with_barriers(build_tlas_barrier::<T::TLASType>)
                        .after(prepare_tlas::<T::TLASType>),
                )
                    .in_set(TLASBuilderSet),
            );
        }

        app.add_systems(
            PostUpdate,
            (
                extract_input::<T>
                    .with_barriers(extract_input_barrier::<T>)
                    .after(resize_buffer::<T::TLASType>)
                    .before(prepare_tlas::<T::TLASType>),
                assign_index::<T>.before(resize_buffer::<T::TLASType>),
            )
                .in_set(TLASBuilderSet),
        );

        // TODO: If supports host build, do those things.
    }
}
