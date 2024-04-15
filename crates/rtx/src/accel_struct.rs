use std::{
    alloc::Layout,
    any::Any,
    borrow::{BorrowMut, Cow},
    collections::{BTreeMap, BTreeSet},
    ops::Deref,
    sync::Arc,
};

use bevy::{
    ecs::{
        component::Component,
        entity::Entity,
        query::{Added, Changed, Or, QueryFilter, QueryItem, ReadOnlyQueryData, With},
        system::{
            Commands, Local, Query, Res, ResMut, Resource, StaticSystemParam, SystemParam,
            SystemParamItem,
        },
    },
    transform,
    utils::petgraph::data,
};

use rhyolite::{
    ash::{extensions::khr, prelude::VkResult, vk},
    commands::{ComputeCommands, TransferCommands},
    task::{AsyncComputeTask, AsyncTaskPool},
    Allocator, Buffer, BufferLike, HasDevice,
};

pub struct AccelStruct {
    buffer: Buffer,
    raw: vk::AccelerationStructureKHR,
    flags: vk::BuildAccelerationStructureFlagsKHR,
    device_address: vk::DeviceAddress,
}
impl Drop for AccelStruct {
    fn drop(&mut self) {
        unsafe {
            self.buffer
                .device()
                .extension::<khr::AccelerationStructure>()
                .destroy_acceleration_structure(self.raw, None);
        }
    }
}
impl AccelStruct {
    pub fn new_blas(allocator: Allocator, size: vk::DeviceSize) -> VkResult<Self> {
        Self::new(
            allocator,
            size,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        )
    }
    pub fn new(
        allocator: Allocator,
        size: vk::DeviceSize,
        ty: vk::AccelerationStructureTypeKHR,
    ) -> VkResult<Self> {
        let buffer = Buffer::new_resource(
            allocator,
            size,
            1,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;
        unsafe {
            let raw = buffer
                .device()
                .extension::<khr::AccelerationStructure>()
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR {
                        ty,
                        size,
                        buffer: buffer.raw_buffer(),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            let device_address = buffer
                .device()
                .extension::<khr::AccelerationStructure>()
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR {
                        acceleration_structure: raw,
                        ..Default::default()
                    },
                );
            Ok(Self {
                buffer,
                raw,
                flags: vk::BuildAccelerationStructureFlagsKHR::empty(),
                device_address,
            })
        }
    }
}

#[derive(Component)]
pub struct BLAS {
    accel_struct: AccelStruct,
}

// Three ways to update:
// Device build, upload buffer immediately
// Device build, buffer already uploaded
// Host build, provide data ptr.

// Three ownership models:
// Data owned as third data structure. Use writer pattern
// Data owned as plain buffer. Prefer providing ptr.
// Data owned as device buffer. Use device address.
pub trait BLASBuildMarker: Send + Sync + 'static {
    /// The marker component. Any entities with this component will have a BLAS built for them.
    type Marker: Component;
    /// Associated entities to be passed.
    type QueryData: ReadOnlyQueryData;
    type QueryFilter: QueryFilter;
    /// Additional system entities to be passed.
    type Params: SystemParam;

    /// If returns true, we will attempt to bulid the BLAS on the host side if possible.
    /// Host side build will occur if:
    /// 1. The GPU driver supports host side builds
    /// 2. This function returns true
    /// Whenever this function returns true, `geometries` must return objects that will support host address.
    fn should_use_host_build(_params: &mut SystemParamItem<Self::Params>) -> bool {
        false
    }

    fn build_flags(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> vk::BuildAccelerationStructureFlagsKHR {
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
            | vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION
    }

    /// BLAS updates will occur if:
    /// 1. The BLAS was initially built with the `ALLOW_UPDATE` flag set. This is set by the `build_flags` function.
    /// 2. This function returns true.
    fn should_update(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> bool {
        false
    }
}


/// Builders with data not owned anywhere. Needing to be generated on the fly.
/// Implementing this trait helps elimitating unnecessary copies by allowing you
/// to write directly into the destination memory.
pub trait BLASStagingBufferBuilder: BLASBuildMarker {
    fn staging_layout(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> Layout;
    type GeometryIterator<'a>: Iterator<Item = BLASBuildGeometry<vk::DeviceSize>> + 'a;
    /// The geometries to be built. The implementation shall write directly into the dst buffer.
    /// The iterator returned shall contain offset values into the dst buffer.
    fn geometries<'a>(
        params: &'a mut SystemParamItem<Self::Params>,
        data: &'a QueryItem<Self::QueryData>,
        dst: &mut [u8],
    ) -> Self::GeometryIterator<'a>;
}
/// Builders with data already located on host memory. Used for host builds.
pub trait BLASHostBufferBuilder: BLASBuildMarker {
    type Data: Deref<Target = [u8]>;
    fn geometries(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> impl Iterator<Item = BLASBuildGeometry<Self::Data>>;
}
/// Builders with data already located on owned device memory
pub trait BLASDeviceBuilder: BLASBuildMarker {
    fn geometries(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> impl Iterator<Item = BLASBuildGeometry<vk::DeviceAddress>>;
}
pub enum BLASBuildGeometry<T> {
    Triangles {
        vertex_format: vk::Format,
        vertex_data: T,
        vertex_stride: vk::DeviceSize,
        max_vertex: u32,
        index_type: vk::IndexType,
        index_data: T,
        transform_data: Option<T>,
        flags: vk::GeometryFlagsKHR,
        /// Number of triangles to be built, where each triangle is treated as 3 vertices
        primitive_count: u32,
    },
    Aabbs {
        buffer: T,
        stride: vk::DeviceSize,
        flags: vk::GeometryFlagsKHR,
        /// Number of AABBs to be built, where each triangle is treated as 3 vertices
        primitive_count: u32,
    },
}
impl<T> BLASBuildGeometry<T> {
    pub fn ty(&self) -> vk::GeometryTypeKHR {
        match self {
            BLASBuildGeometry::Triangles { .. } => vk::GeometryTypeKHR::TRIANGLES,
            BLASBuildGeometry::Aabbs { .. } => vk::GeometryTypeKHR::AABBS,
        }
    }
    pub fn to_device_or_host_address(
        self,
        mapper: impl Fn(T) -> vk::DeviceOrHostAddressConstKHR,
    ) -> BLASBuildGeometry<vk::DeviceOrHostAddressConstKHR>
    {
        match self {
            BLASBuildGeometry::Triangles {
                vertex_format,
                vertex_data,
                vertex_stride,
                max_vertex,
                index_type,
                index_data,
                transform_data,
                flags,
                primitive_count,
            } => BLASBuildGeometry::Triangles {
                vertex_format: vertex_format,
                vertex_data: mapper(vertex_data),
                vertex_stride: vertex_stride,
                max_vertex: max_vertex,
                index_type: index_type,
                index_data: mapper(index_data),
                transform_data: transform_data.map(mapper),
                flags: flags,
                primitive_count: primitive_count,
            },
            BLASBuildGeometry::Aabbs {
                buffer,
                stride,
                flags,
                primitive_count,
            } => BLASBuildGeometry::Aabbs {
                buffer: mapper(buffer),
                stride: stride,
                flags: flags,
                primitive_count: primitive_count,
            },
        }
    }
}

#[derive(Default, Resource)]
struct BLASStore {
    pending_entities: BTreeSet<Entity>,
    pending_task: Option<AsyncComputeTask<BLASBuildPendingTaskInfo>>,
    scratch_buffer: Option<Buffer>,

    pending_builds: PendingBuilds,
}

/// 1. Maintains list of pending_entities and resolve pending entities
fn extract_updated_entities<T: BLASBuildMarker>(
    mut commands: Commands,
    store: ResMut<BLASStore>,
    mut task_pool: ResMut<AsyncTaskPool>,
    mut staging_buffer_store: Option<ResMut<BLASStagingBuilderStore>>,
    changed_entities: Query<Entity, (Or<(Added<T::Marker>, Changed<T::Marker>)>, T::QueryFilter)>,
) {
    let store = store.into_inner();
    if let Some(task) = store.pending_task.as_ref() {
        if !task.is_finished() {
            return;
        }
        let task = store.pending_task.take().unwrap();
        let mut result = task_pool.wait_blocked(task);
        for (entity, accel_struct) in result.entities.drain(..) {
            commands.entity(entity).insert(BLAS { accel_struct });
        }
        // Reuse some resources
        assert!(store.scratch_buffer.is_none());
        store.scratch_buffer = Some(result.scratch_buffer);

        if let Some(staging_buffer_store) = &mut staging_buffer_store {
            assert!(
                staging_buffer_store.input_buffer.is_none()
                    && staging_buffer_store.staging_buffer.is_none()
            );
            staging_buffer_store.input_buffer = result.input_buffer;
            staging_buffer_store.staging_buffer = result.staging_buffer;
            staging_buffer_store.total_buffer_size = 0;
            staging_buffer_store.base_alignment = 1;
            staging_buffer_store.current_write_ptr = 0;
        }
    }

    for entity in changed_entities.iter() {
        store.pending_entities.insert(entity);
    }
}

#[derive(Resource)]
struct BLASStagingBuilderStore {
    total_buffer_size: u64,
    base_alignment: u64,
    input_buffer: Option<Buffer>,
    staging_buffer: Option<Buffer>,

    current_write_ptr: u64,
}
fn staging_builder_calculate_layout<T: BLASStagingBufferBuilder>(
    mut params: StaticSystemParam<T::Params>,
    entities: Query<T::QueryData, (With<T::Marker>, T::QueryFilter)>,
    mut builder_store: ResMut<BLASStagingBuilderStore>,
    store: Res<BLASStore>,
) {
    if store.pending_task.is_some() {
        builder_store.total_buffer_size = 0;
        builder_store.base_alignment = 1;
        return;
    }
    if store.pending_entities.is_empty() {
        builder_store.total_buffer_size = 0;
        builder_store.base_alignment = 1;
        return;
    }
    let store = store.into_inner();
    for entity in store.pending_entities.iter() {
        let data = entities.get(*entity).unwrap();
        let layout = T::staging_layout(&mut params, &data);
        builder_store.total_buffer_size = builder_store
            .total_buffer_size
            .next_multiple_of(layout.align() as u64);
        builder_store.total_buffer_size += layout.size() as u64;
        builder_store.base_alignment = builder_store.base_alignment.max(layout.align() as u64);
    }
}
fn staging_builder_realloc_buffer(
    allocator: Res<Allocator>,
    store: ResMut<BLASStagingBuilderStore>,
) {
    let store = store.into_inner();
    if store.total_buffer_size == 0 {
        return;
    }

    let use_staging_buffer = allocator
        .device()
        .physical_device()
        .properties()
        .memory_model
        .storage_buffer_should_use_staging();

    // Resize input_buffer
    if let Some(buffer) = store.input_buffer.as_ref()
        && buffer.size() >= store.total_buffer_size
    {
        // Preserve old buffer: it's big enough.
    } else {
        store.input_buffer = None;
        store.input_buffer = Some(if use_staging_buffer {
            Buffer::new_resource(
                allocator.clone(),
                store.total_buffer_size,
                store.base_alignment,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .unwrap()
        } else {
            Buffer::new_dynamic(
                allocator.clone(),
                store.total_buffer_size,
                store.base_alignment,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            )
            .unwrap()
        });
    };
}
fn staging_builder_extract_changes<T: BLASStagingBufferBuilder>(
    allocator: Res<Allocator>,
    mut params: StaticSystemParam<T::Params>,
    mut store: ResMut<BLASStore>,
    mut staging_build_store: ResMut<BLASStagingBuilderStore>,
    all_entities: Query<(Option<&mut BLAS>, T::QueryData), (With<T::Marker>, T::QueryFilter)>,
) {
    let staging_build_store = staging_build_store.into_inner();
    let Some(dst_buffer) = staging_build_store
        .staging_buffer
        .as_mut()
        .or(staging_build_store.input_buffer.as_mut())
    else {
        return;
    };
    let base_device_address = dst_buffer.device_address();
    let dst_buffer = dst_buffer.as_slice_mut();

    struct MyCallback<'a> {
        dst_buffer: &'a mut [u8],
        base_device_address: vk::DeviceAddress,
        current_write_ptr: &'a mut u64,
    }
    impl<'a, 'b, T: BLASStagingBufferBuilder> Callback<'a, T> for MyCallback<'b> {
        type Iterator = impl Iterator<Item = BLASBuildGeometry<vk::DeviceOrHostAddressConstKHR>> + 'a;
        fn call(
            &mut self,
            params: &'a mut SystemParamItem<T::Params>,
            data: &'a QueryItem<T::QueryData>,
        ) -> Self::Iterator {
            let size = T::staging_layout(params, data);
            let base_device_address = self.base_device_address + *self.current_write_ptr;
            let dst_buffer = &mut self.dst_buffer[(*self.current_write_ptr) as usize .. (*self.current_write_ptr as usize + size.size())];
            T::geometries(params, data, dst_buffer).map(move |g| g.to_device_or_host_address(|offset|
                vk::DeviceOrHostAddressConstKHR {
                    device_address: base_device_address + offset
                }
            ))
        }
    }
    
    extract_blas_inner::<T, _>(
        allocator,
        params,
        store,
        all_entities,
        MyCallback { dst_buffer, base_device_address, current_write_ptr: &mut staging_build_store.current_write_ptr },
    );
}

struct BLASPendingBuild {
    info: vk::AccelerationStructureBuildGeometryInfoKHR,
    build_sizes: vk::AccelerationStructureBuildSizesInfoKHR,
    geometry_array_index_start: usize,
}

struct PendingBuilds {
    pending_builds: BTreeMap<Entity, BLASPendingBuild>,
    geometries: Vec<vk::AccelerationStructureGeometryKHR>,
    build_ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
}
impl Default for PendingBuilds {
    fn default() -> Self {
        Self {
            pending_builds: BTreeMap::new(),
            geometries: Vec::new(),
            build_ranges: Vec::new(),
        }
    }
}
unsafe impl Send for BLASStore {}
unsafe impl Sync for BLASStore {}

struct BLASBuildPendingTaskInfo {
    entities: Vec<(Entity, AccelStruct)>,
    scratch_buffer: Buffer,
    input_buffer: Option<Buffer>,
    staging_buffer: Option<Buffer>,
}

trait Callback<'a, T: BLASBuildMarker> {
    type Iterator: Iterator<Item = BLASBuildGeometry<vk::DeviceOrHostAddressConstKHR>> + 'a;
    fn call(
        &mut self,
        params: &'a mut SystemParamItem<T::Params>,
        data: &'a QueryItem<T::QueryData>,
    ) -> Self::Iterator;
}

fn extract_blas_inner<T: BLASBuildMarker, F>(
    allocator: Res<Allocator>,
    mut params: StaticSystemParam<T::Params>,
    store: ResMut<BLASStore>,
    all_entities: Query<(Option<&mut BLAS>, T::QueryData), (With<T::Marker>, T::QueryFilter)>,
    mut geometries: F,
) where
    F: for<'a> Callback<'a, T>,
{
    let params = &mut *params;
    let store = store.into_inner();
    let mut primitive_counts: Vec<u32> = Vec::new();

    for entity in store.pending_entities.iter() {
        let Ok((old_blas, query_data)) = all_entities.get(*entity) else {
            return;
        };
        let should_update = old_blas
            .as_ref()
            .map(|a| {
                a.accel_struct
                    .flags
                    .contains(vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE)
            })
            .unwrap_or(false)
            && T::should_update(params, &query_data);
        let mut geometry_count: u32 = 0;
        let geometry_array_index_start = store.pending_builds.geometries.len();
        debug_assert_eq!(
            store.pending_builds.geometries.len(),
            store.pending_builds.build_ranges.len()
        );
        primitive_counts.clear();
        for geometry in geometries.call(params, &query_data) {
            let geometry_type = geometry.ty();
            geometry_count += 1;

            match &geometry {
                BLASBuildGeometry::Triangles {
                    vertex_format,
                    vertex_data,
                    vertex_stride,
                    max_vertex,
                    index_type,
                    index_data,
                    transform_data,
                    flags,
                    primitive_count,
                } => {
                    store
                        .pending_builds
                        .geometries
                        .push(vk::AccelerationStructureGeometryKHR {
                            geometry_type,
                            geometry: vk::AccelerationStructureGeometryDataKHR {
                                triangles: vk::AccelerationStructureGeometryTrianglesDataKHR {
                                    vertex_format: *vertex_format,
                                    vertex_stride: *vertex_stride,
                                    max_vertex: *max_vertex,
                                    index_type: *index_type,
                                    vertex_data: *vertex_data,
                                    index_data: *index_data,
                                    transform_data: transform_data
                                        .as_ref()
                                        .cloned()
                                        .unwrap_or_default(),
                                    ..Default::default()
                                },
                            },
                            flags: *flags,
                            ..Default::default()
                        });
                    primitive_counts.push(*primitive_count);
                    store.pending_builds.build_ranges.push(
                        vk::AccelerationStructureBuildRangeInfoKHR {
                            primitive_count: *primitive_count,
                            ..Default::default()
                        },
                    );
                }
                BLASBuildGeometry::Aabbs {
                    buffer,
                    stride,
                    flags,
                    primitive_count,
                } => {
                    store
                        .pending_builds
                        .geometries
                        .push(vk::AccelerationStructureGeometryKHR {
                            geometry_type,
                            geometry: vk::AccelerationStructureGeometryDataKHR {
                                aabbs: vk::AccelerationStructureGeometryAabbsDataKHR {
                                    stride: *stride,
                                    data: *buffer,
                                    ..Default::default()
                                },
                            },
                            flags: *flags,
                            ..Default::default()
                        });
                    primitive_counts.push(*primitive_count);
                    store.pending_builds.build_ranges.push(
                        vk::AccelerationStructureBuildRangeInfoKHR {
                            primitive_count: *primitive_count,
                            ..Default::default()
                        },
                    );
                }
            };
        }
        let info = vk::AccelerationStructureBuildGeometryInfoKHR {
            ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            flags: T::build_flags(params, &query_data),
            mode: if should_update {
                vk::BuildAccelerationStructureModeKHR::UPDATE
            } else {
                vk::BuildAccelerationStructureModeKHR::BUILD
            },
            src_acceleration_structure: if should_update {
                old_blas.as_ref().unwrap().accel_struct.raw
            } else {
                vk::AccelerationStructureKHR::null()
            },
            geometry_count,
            ..Default::default()
        };
        let build_sizes = unsafe {
            allocator
                .device()
                .extension::<khr::AccelerationStructure>()
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &info,
                    &primitive_counts,
                )
        };
        store.pending_builds.pending_builds.insert(
            *entity,
            BLASPendingBuild {
                info,
                build_sizes,
                geometry_array_index_start,
            },
        );
    }
}

fn device_build_blas_system(
    allocator: Res<Allocator>,
    mut query: Query<&mut BLAS>,
    mut task_pool: ResMut<AsyncTaskPool>,
    mut store: ResMut<BLASStore>,
    mut staging_store: ResMut<BLASStagingBuilderStore>,
) {
    let store = store.into_inner();
    let staging_store = staging_store.into_inner();
    let mut entities: Vec<(Entity, AccelStruct)> = Vec::new();
    let mut scratch_buffer: Option<Buffer> = store.scratch_buffer.take();
    let mut staging_buffer: Option<Buffer> = staging_store.staging_buffer.take();
    let mut build_input_buffer: Option<Buffer> = staging_store.input_buffer.take();

    let mut build = std::mem::take(&mut store.pending_builds);

    let mut total_scratch_size: u64 = 0;
    let scratch_offset_alignment: u32 = allocator
        .device()
        .physical_device()
        .properties()
        .get::<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>()
        .min_acceleration_structure_scratch_offset_alignment;
    for pending in build.pending_builds.values() {
        total_scratch_size = total_scratch_size.next_multiple_of(scratch_offset_alignment as u64);
        total_scratch_size += pending.build_sizes.build_scratch_size;
    }
    // Create scratch buffer
    let scratch_buffer = if let Some(s) = scratch_buffer.as_ref()
        && s.size() >= total_scratch_size
    {
        // Reuse scratch buffer from previous build
        scratch_buffer.take().unwrap()
    } else {
        drop(scratch_buffer);
        Buffer::new_resource(
            allocator.clone(),
            total_scratch_size,
            scratch_offset_alignment as u64,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )
        .unwrap()
    };
    let mut scratch_device_address = scratch_buffer.device_address();

    // Create upload buffer
    let mut recorder = if staging_store.total_buffer_size > 0 {
        let recorder = if let Some(staging_buffer) = &staging_buffer {
            let input_buffer = build_input_buffer.as_ref().unwrap();
            let mut recorder = task_pool.spawn_transfer();
            recorder.copy_buffer(
                staging_buffer.raw_buffer(),
                input_buffer.raw_buffer(),
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: staging_store.total_buffer_size,
                }],
            );
            recorder.commit::<(), 'c'>()
        } else {
            task_pool.spawn_compute()
        };
        recorder
    } else {
        task_pool.spawn_compute()
    };

    for (entity, pending) in build.pending_builds.iter_mut() {
        pending.info.p_geometries = &build.geometries[pending.geometry_array_index_start];

        scratch_device_address =
            scratch_device_address.next_multiple_of(scratch_offset_alignment as u64);
        pending.info.scratch_data.device_address = scratch_device_address;

        if pending.info.mode == vk::BuildAccelerationStructureModeKHR::UPDATE {
            scratch_device_address += pending.build_sizes.update_scratch_size;
        } else {
            scratch_device_address += pending.build_sizes.build_scratch_size;
        }
        scratch_device_address += pending.build_sizes.build_scratch_size;

        // Create acceleration structure
        let accel_struct = AccelStruct::new(
            allocator.clone(),
            pending.build_sizes.acceleration_structure_size,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        )
        .unwrap();
        pending.info.dst_acceleration_structure = accel_struct.raw;
        entities.push((*entity, accel_struct));
    }
    let build_range_infos = build.pending_builds.values().map(|pending| {
        let slice = &build.build_ranges[pending.geometry_array_index_start
            ..(pending.geometry_array_index_start + pending.info.geometry_count as usize)];
        slice
    });
    let build_geometry_info: Vec<_> = build
        .pending_builds
        .values()
        .map(|pending| pending.info.clone())
        .collect();

    recorder.build_acceleration_structure(&build_geometry_info, build_range_infos);
    store.pending_task = Some(recorder.finish(BLASBuildPendingTaskInfo {
        entities,
        scratch_buffer,
        input_buffer: build_input_buffer,
        staging_buffer,
    }));
}
