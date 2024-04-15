use std::{
    any::Any,
    borrow::{BorrowMut, Cow},
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use bevy::{
    ecs::{
        component::Component,
        entity::Entity,
        query::{Added, Changed, Or, QueryFilter, QueryItem, ReadOnlyQueryData, With},
        system::{
            Local, Query, Res, ResMut, Resource, StaticSystemParam, SystemParam, SystemParamItem,
        },
    },
    transform,
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

    type BLASInputBufferType: HasDeviceOrHostAddress;
    fn geometries(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> impl Iterator<Item = BLASBuildGeometry<Self::BLASInputBufferType>>;

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

trait HasDeviceOrHostAddress {
    fn has_device_address(&self) -> bool;
    fn has_host_address(&self) -> bool;

    fn device_address(&mut self) -> vk::DeviceAddress;
    fn host_buffer(&self) -> Vec<u8>;
    fn host_writer(&self, dst: &mut [u8]) {
        dst.copy_from_slice(&self.host_buffer());
    }
}
pub enum DeviceOrHost {}
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
}
struct BLASPendingBuild {
    info: vk::AccelerationStructureBuildGeometryInfoKHR,
    build_sizes: vk::AccelerationStructureBuildSizesInfoKHR,
    geometry_array_index_start: usize,
}

struct PendingBuilds<T> {
    pending_builds: BTreeMap<Entity, BLASPendingBuild>,
    geometries: Vec<vk::AccelerationStructureGeometryKHR>,
    build_ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
    geometry_objs: Vec<BLASBuildGeometry<T>>,
}
impl<T> Default for PendingBuilds<T> {
    fn default() -> Self {
        Self {
            pending_builds: BTreeMap::new(),
            geometries: Vec::new(),
            build_ranges: Vec::new(),
            geometry_objs: Vec::new(),
        }
    }
}
impl<T: HasDeviceOrHostAddress> BLASBuildGeometry<T> {
    pub fn into_vec(self) -> BLASBuildGeometry<Vec<u8>> {
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
                vertex_format,
                vertex_data: vertex_data.host_buffer(),
                vertex_stride,
                max_vertex,
                index_type,
                index_data: index_data.host_buffer(),
                transform_data: transform_data.map(|a| a.host_buffer()),
                flags,
                primitive_count,
            },
            BLASBuildGeometry::Aabbs {
                buffer,
                stride,
                flags,
                primitive_count,
            } => BLASBuildGeometry::Aabbs {
                buffer: buffer.host_buffer(),
                stride,
                flags,
                primitive_count,
            },
        }
    }
    pub fn into_dyn(self) -> BLASBuildGeometry<Box<dyn HasDeviceOrHostAddress>>
    where
        Self: 'static,
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
                vertex_format,
                vertex_data: Box::new(vertex_data),
                vertex_stride,
                max_vertex,
                index_type,
                index_data: Box::new(index_data),
                transform_data: transform_data
                    .map(|a| Box::new(a) as Box<dyn HasDeviceOrHostAddress>),
                flags,
                primitive_count,
            },
            BLASBuildGeometry::Aabbs {
                buffer,
                stride,
                flags,
                primitive_count,
            } => BLASBuildGeometry::Aabbs {
                buffer: Box::new(buffer),
                stride,
                flags,
                primitive_count,
            },
        }
    }
}
#[derive(Default, Resource)]
struct BLASStore {
    pending_task: Option<AsyncComputeTask<BLASBuildPendingTaskInfo>>,

    host_builds: PendingBuilds<Vec<u8>>,
    device_builds: PendingBuilds<Box<dyn HasDeviceOrHostAddress>>,
}
unsafe impl Send for BLASStore {}
unsafe impl Sync for BLASStore {}

struct BLASBuildPendingTaskInfo {
    entities: Vec<(Entity, AccelStruct)>,
    scratch_buffer: Buffer,
}

fn extract_blas_updates<T: BLASBuildMarker>(
    allocator: Res<Allocator>,
    mut params: StaticSystemParam<T::Params>,
    changes: Query<
        (Entity, Option<&BLAS>, T::QueryData),
        Or<(Added<T::Marker>, Changed<T::Marker>)>,
    >,
    mut store: ResMut<BLASStore>,
) {
    let store = store.into_inner();
    let driver_supports_host_build = allocator
        .device()
        .feature::<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>()
        .map(|x| x.acceleration_structure_host_commands == vk::TRUE)
        .unwrap_or(false);
    let mut primitive_counts: Vec<u32> = Vec::new();
    changes.iter().for_each(|(entity, old_blas, query_data)| {
        let build_on_host = driver_supports_host_build && T::should_use_host_build(&mut params);
        let should_update = old_blas
            .as_ref()
            .map(|a| {
                a.accel_struct
                    .flags
                    .contains(vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE)
            })
            .unwrap_or(false)
            && T::should_update(&mut params, &query_data);
        let (geometries, build_ranges, pending_builds) = if build_on_host {
            (
                &mut store.host_builds.geometries,
                &mut store.host_builds.build_ranges,
                &mut store.host_builds.pending_builds,
            )
        } else {
            (
                &mut store.device_builds.geometries,
                &mut store.device_builds.build_ranges,
                &mut store.device_builds.pending_builds,
            )
        };
        let mut geometry_count: u32 = 0;
        let geometry_array_index_start = geometries.len();
        debug_assert_eq!(geometries.len(), build_ranges.len());
        primitive_counts.clear();
        for geometry in T::geometries(&mut params, &query_data) {
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
                    geometries.push(vk::AccelerationStructureGeometryKHR {
                        geometry_type,
                        geometry: vk::AccelerationStructureGeometryDataKHR {
                            triangles: vk::AccelerationStructureGeometryTrianglesDataKHR {
                                vertex_format: *vertex_format,
                                vertex_stride: *vertex_stride,
                                max_vertex: *max_vertex,
                                index_type: *index_type,
                                ..Default::default()
                            },
                        },
                        flags: *flags,
                        ..Default::default()
                    });
                    primitive_counts.push(*primitive_count);
                    build_ranges.push(vk::AccelerationStructureBuildRangeInfoKHR {
                        primitive_count: *primitive_count,
                        ..Default::default()
                    });
                }
                BLASBuildGeometry::Aabbs {
                    buffer,
                    stride,
                    flags,
                    primitive_count,
                } => {
                    geometries.push(vk::AccelerationStructureGeometryKHR {
                        geometry_type,
                        geometry: vk::AccelerationStructureGeometryDataKHR {
                            aabbs: vk::AccelerationStructureGeometryAabbsDataKHR {
                                stride: *stride,
                                ..Default::default()
                            },
                        },
                        flags: *flags,
                        ..Default::default()
                    });
                    primitive_counts.push(*primitive_count);
                    build_ranges.push(vk::AccelerationStructureBuildRangeInfoKHR {
                        primitive_count: *primitive_count,
                        ..Default::default()
                    });
                }
            };
            if build_on_host {
                store.host_builds.geometry_objs.push(geometry.into_vec());
            } else {
                store.device_builds.geometry_objs.push(geometry.into_dyn());
            }
        }
        let info = vk::AccelerationStructureBuildGeometryInfoKHR {
            ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            flags: T::build_flags(&mut params, &query_data),
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
        pending_builds.insert(
            entity,
            BLASPendingBuild {
                info,
                build_sizes,
                geometry_array_index_start,
            },
        );
    });
}

fn device_build_blas_system(
    allocator: Res<Allocator>,
    mut query: Query<&mut BLAS>,
    mut task_pool: ResMut<AsyncTaskPool>,
    mut store: ResMut<BLASStore>,
) {
    let store = store.into_inner();
    let mut entities: Vec<(Entity, AccelStruct)> = Vec::new();
    let mut scratch_buffer: Option<Buffer> = None;
    let mut staging_buffer: Option<Buffer> = None;
    let mut build_input_buffer: Option<Buffer> = None;

    if let Some(task) = store.pending_task.as_ref() {
        if !task.is_finished() {
            return;
        }
        let task = store.pending_task.take().unwrap();
        let mut result = task_pool.wait_blocked(task);
        for (entity, accel_struct) in result.entities.drain(..) {
            let mut blas = query.get_mut(entity).unwrap();
            blas.accel_struct = accel_struct;
        }
        // Reuse some resources
        entities = result.entities;
        scratch_buffer = Some(result.scratch_buffer);
    }

    let mut build = std::mem::take(&mut store.device_builds);

    let mut total_scratch_size: u64 = 0;
    let mut total_upload_buffer_size: u64 = 0;
    let scratch_offset_alignment: u32 = allocator
        .device()
        .physical_device()
        .properties()
        .get::<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>()
        .min_acceleration_structure_scratch_offset_alignment;
    for pending in build.pending_builds.values() {
        total_scratch_size = total_scratch_size.next_multiple_of(scratch_offset_alignment as u64);
        total_scratch_size += pending.build_sizes.build_scratch_size;

        let geometries = &build.geometries[pending.geometry_array_index_start
            ..(pending.geometry_array_index_start + pending.info.geometry_count as usize)];
        let geometry_objs = &build.geometry_objs[pending.geometry_array_index_start
            ..(pending.geometry_array_index_start + pending.info.geometry_count as usize)];
        let build_ranges = &build.build_ranges[pending.geometry_array_index_start
            ..(pending.geometry_array_index_start + pending.info.geometry_count as usize)];
        for ((geometry, obj), build_range) in geometries
            .iter()
            .zip(geometry_objs.iter())
            .zip(build_ranges.iter())
        {
            match obj {
                BLASBuildGeometry::Triangles {
                    vertex_data,
                    index_data,
                    transform_data,
                    ..
                } => {
                    let triangles = unsafe { &geometry.geometry.triangles };
                    if !vertex_data.has_device_address() {
                        total_upload_buffer_size = total_upload_buffer_size
                            .next_multiple_of(triangles.vertex_stride as u64);
                        total_upload_buffer_size +=
                            triangles.vertex_stride as u64 * triangles.max_vertex as u64;
                    }
                    if !index_data.has_device_address()
                        && triangles.index_type != vk::IndexType::NONE_KHR
                    {
                        let size = match triangles.index_type {
                            vk::IndexType::UINT16 => std::mem::size_of::<u16>(),
                            vk::IndexType::UINT32 => std::mem::size_of::<u32>(),
                            _ => 0,
                        } as u64;
                        total_upload_buffer_size = total_upload_buffer_size.next_multiple_of(size);
                        total_upload_buffer_size += size * build_range.primitive_count as u64;
                    }
                    if let Some(transform_data) = transform_data
                        && !transform_data.has_device_address()
                    {
                        total_upload_buffer_size = total_upload_buffer_size
                            .next_multiple_of(std::mem::size_of::<vk::TransformMatrixKHR>() as u64);
                        total_upload_buffer_size +=
                            std::mem::size_of::<vk::TransformMatrixKHR>() as u64;
                    }
                }
                BLASBuildGeometry::Aabbs { buffer, .. } => {
                    let aabbs = unsafe { &geometry.geometry.aabbs };
                    if !buffer.has_device_address() {
                        total_upload_buffer_size =
                            total_upload_buffer_size.next_multiple_of(aabbs.stride as u64);
                        total_upload_buffer_size +=
                            aabbs.stride as u64 * build_range.primitive_count as u64;
                    }
                }
            }
        }
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
    let mut recorder = if total_upload_buffer_size > 0 {
        let mut build_input_buffer = if let Some(s) = build_input_buffer.as_ref()
            && s.size() >= total_upload_buffer_size
        {
            // Reuse scratch buffer from previous build
            build_input_buffer.take().unwrap()
        } else {
            drop(build_input_buffer);
            if allocator
                .device()
                .physical_device()
                .properties()
                .memory_model
                .storage_buffer_should_use_staging()
            {
                Buffer::new_resource(
                    allocator.clone(),
                    total_upload_buffer_size,
                    1,
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .unwrap()
            } else {
                Buffer::new_dynamic(
                    allocator.clone(),
                    total_upload_buffer_size,
                    1,
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                )
                .unwrap()
            }
        };
        let mut copy_dst = &mut build_input_buffer;
        let mut staging = None;
        if allocator
            .device()
            .physical_device()
            .properties()
            .memory_model
            .storage_buffer_should_use_staging()
        {
            let staging_buffer = if let Some(s) = staging_buffer.as_ref()
                && s.size() >= total_upload_buffer_size
            {
                // Reuse scratch buffer from previous build
                staging_buffer.take().unwrap()
            } else {
                drop(staging_buffer);

                Buffer::new_staging(
                    allocator.clone(),
                    total_upload_buffer_size,
                    1,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                )
                .unwrap()
            };
            staging = Some(staging_buffer);
            copy_dst = staging.as_mut().unwrap();
        }

        // Now, copy data into copy_dst
        let mut current_location: u64 = 0;
        for pending in build.pending_builds.values() {
            let geometries = &build.geometries[pending.geometry_array_index_start
                ..(pending.geometry_array_index_start + pending.info.geometry_count as usize)];
            let geometry_objs = &build.geometry_objs[pending.geometry_array_index_start
                ..(pending.geometry_array_index_start + pending.info.geometry_count as usize)];
            let build_ranges = &build.build_ranges[pending.geometry_array_index_start
                ..(pending.geometry_array_index_start + pending.info.geometry_count as usize)];
            for ((geometry, obj), build_range) in geometries
                .iter()
                .zip(geometry_objs.iter())
                .zip(build_ranges.iter())
            {
                match obj {
                    BLASBuildGeometry::Triangles {
                        vertex_data,
                        index_data,
                        transform_data,
                        ..
                    } => {
                        let triangles = unsafe { &geometry.geometry.triangles };
                        if !vertex_data.has_device_address() {
                            current_location =
                                current_location.next_multiple_of(triangles.vertex_stride as u64);
                            let len = triangles.vertex_stride as u64 * triangles.max_vertex as u64;
                            let target_slice = &mut copy_dst.as_slice_mut()
                                [current_location as usize..(current_location + len) as usize];
                            vertex_data.host_writer(target_slice);
                            current_location += len;
                        }
                        if !index_data.has_device_address()
                            && triangles.index_type != vk::IndexType::NONE_KHR
                        {
                            let size = match triangles.index_type {
                                vk::IndexType::UINT16 => std::mem::size_of::<u16>(),
                                vk::IndexType::UINT32 => std::mem::size_of::<u32>(),
                                _ => 0,
                            } as u64;
                            current_location = current_location.next_multiple_of(size);
                            let len = size * build_range.primitive_count as u64;
                            let target_slice = &mut copy_dst.as_slice_mut()
                                [current_location as usize..(current_location + len) as usize];
                            index_data.host_writer(target_slice);
                            current_location += len;
                        }
                        if let Some(transform_data) = transform_data
                            && !transform_data.has_device_address()
                        {
                            let size = std::mem::size_of::<vk::TransformMatrixKHR>() as u64;
                            current_location = current_location.next_multiple_of(size);

                            let target_slice = &mut copy_dst.as_slice_mut()
                                [current_location as usize..(current_location + size) as usize];
                            transform_data.host_writer(target_slice);
                            current_location += size;
                        }
                    }
                    BLASBuildGeometry::Aabbs { buffer, .. } => {
                        let aabbs = unsafe { &geometry.geometry.aabbs };
                        if !buffer.has_device_address() {
                            let len = aabbs.stride as u64 * build_range.primitive_count as u64;
                            let target_slice = &mut copy_dst.as_slice_mut()
                                [current_location as usize..(current_location + len) as usize];
                            buffer.host_writer(target_slice);
                            current_location += len;
                        }
                    }
                }
            }
        }

        if let Some(staging_buffer) = staging {
            let mut recorder = task_pool.spawn_transfer();
            recorder.copy_buffer(
                staging_buffer.raw_buffer(),
                build_input_buffer.raw_buffer(),
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: total_upload_buffer_size,
                }],
            );
            recorder.commit::<(), 'c'>()
        } else {
            task_pool.spawn_compute()
        }
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
    }));
}
