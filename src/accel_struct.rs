use std::{alloc::Layout, collections::{BTreeMap, BTreeSet}, sync::Arc};

use ash::{prelude::VkResult, vk};
use bevy::ecs::{
    component::Component,
    entity::Entity,
    query::{Added, Changed, Or, QueryFilter, QueryItem, ReadOnlyQueryData, With},
    system::{Local, ParamSet, Query, Res, ResMut, StaticSystemParam, SystemParam, SystemParamItem},
};

use crate::{
    commands::ComputeCommands,
    task::{AsyncComputeTask, AsyncComputeTaskPool},
    Allocator, Buffer, BufferLike, Device, HasDevice,
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
                .extension::<ash::extensions::khr::AccelerationStructure>()
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
                .extension::<ash::extensions::khr::AccelerationStructure>()
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
                .extension::<ash::extensions::khr::AccelerationStructure>()
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
pub struct BLAS<Marker: BLASBuildMarker> {
    accel_struct: Option<Arc<AccelStruct>>,
    _marker: std::marker::PhantomData<Marker>,
}

pub trait BLASBuildMarker: Send + Sync + 'static {
    /// The marker component. Any entities with this component will have a BLAS built for them.
    type Marker: Component;
    /// Associated entities to be passed.
    type QueryData: ReadOnlyQueryData;
    type QueryFilter: QueryFilter;
    /// Additional system entities to be passed.
    type Params: SystemParam;
    /// Type to uniquely identify a SBT entry.
    type Key: Send + Sync + Ord + Copy;
    /// The key for a given entity.
    fn blas_key(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> Self::Key;

    fn should_use_host_build(params: &mut SystemParamItem<Self::Params>) -> bool;

    fn geometries(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> impl Iterator<Item = BLASBuildGeometry<Buffer>>;

    fn build_flags(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> vk::BuildAccelerationStructureFlagsKHR {
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
            | vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION
    }

    fn should_update(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> bool {
        false
    }
}
pub enum BLASBuildGeometry<T: BufferLike> {
    Triangles {
        vertex_format: vk::Format,
        vertex_data: T,
        vertex_stride: vk::DeviceSize,
        max_vertex: u32,
        index_type: vk::IndexType,
        index_data: T,
        transform_data: T,
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
impl BLASBuildGeometry<Buffer> {
    pub fn ty(&self) -> vk::GeometryTypeKHR {
        match self {
            BLASBuildGeometry::Triangles { .. } => vk::GeometryTypeKHR::TRIANGLES,
            BLASBuildGeometry::Aabbs { .. } => vk::GeometryTypeKHR::AABBS,
        }
    }
}

#[derive(Default)]
struct BLASStore {
    pending_task: Option<AsyncComputeTask<BLASBuildPendingTaskInfo>>,
    deferred_changes: BTreeSet<Entity>
}

struct BLASBuildPendingTaskInfo {
    entities: Vec<(Entity, AccelStruct)>,
    scratch_buffer: Buffer,
}

fn build_blas_system<T: BLASBuildMarker>(
    allocator: Res<Allocator>,
    mut params: StaticSystemParam<T::Params>,
    mut query: ParamSet<(
        Query<(Entity, T::QueryData, &BLAS<T>), Or<(Added<T::Marker>, Changed<T::Marker>)>>,
        Query<(T::QueryData, &mut BLAS<T>), With<T::Marker>>,
    )>,
    mut async_task_pool: ResMut<AsyncComputeTaskPool>,
    mut store: Local<BLASStore>,
) {
    let mut entities: Vec<(Entity, AccelStruct)> = Vec::new();
    let mut scratch_buffer: Option<Buffer> = None;

    if let Some(task) = store.pending_task.as_ref() {
        if !task.is_finished() {
            store.deferred_changes.extend(query.p0().iter().map(|(entity, _, _)| entity));
            return;
        }
        let task = store.pending_task.take().unwrap();
        let mut result = async_task_pool.wait_blocked(task);
        for (entity, accel_struct) in result.entities.drain(..) {
            let mut all_items = query.p1();
            let (_, mut blas) = all_items.get_mut(entity).unwrap();
            blas.accel_struct = Some(Arc::new(accel_struct));
        }
        // Reuse some resources
        entities = result.entities;
        scratch_buffer = Some(result.scratch_buffer);
    }
    let mut geometries: Vec<vk::AccelerationStructureGeometryKHR> = Vec::new();
    let mut primitive_counts: Vec<u32> = Vec::new();
    let mut build_ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR> = Vec::new();

    let mut total_scratch_size: u64 = 0;
    let scratch_offset_alignment: u32 = allocator
        .device()
        .physical_device()
        .properties()
        .get::<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>()
        .min_acceleration_structure_scratch_offset_alignment;

    let mut build_geometry_info: Vec<vk::AccelerationStructureBuildGeometryInfoKHR> = query
        .p0()
        .iter_mut()
        .map(|(entity, query_data, blas)| {
            let should_update = blas
                .accel_struct
                .as_ref()
                .map(|a| {
                    a.flags
                        .contains(vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE)
                })
                .unwrap_or(false)
                || T::should_update(&mut params, &query_data);
            let mut geometry_count: u32 = 0;
            primitive_counts.clear();
            for geometry in T::geometries(&mut params, &query_data) {
                let geometry_type = geometry.ty();
                geometry_count += 1;
                match geometry {
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
                                    vertex_format,
                                    vertex_data: vk::DeviceOrHostAddressConstKHR {
                                        device_address: vertex_data.device_address(),
                                    },
                                    vertex_stride,
                                    max_vertex,
                                    index_type,
                                    index_data: vk::DeviceOrHostAddressConstKHR {
                                        device_address: index_data.device_address(),
                                    },
                                    transform_data: vk::DeviceOrHostAddressConstKHR {
                                        device_address: transform_data.device_address(),
                                    },
                                    ..Default::default()
                                },
                            },
                            flags,
                            ..Default::default()
                        });
                        primitive_counts.push(primitive_count);
                        build_ranges.push(vk::AccelerationStructureBuildRangeInfoKHR {
                            primitive_count,
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
                                    data: vk::DeviceOrHostAddressConstKHR {
                                        device_address: buffer.device_address(),
                                    },
                                    stride,
                                    ..Default::default()
                                },
                            },
                            flags,
                            ..Default::default()
                        });
                        primitive_counts.push(primitive_count);
                        build_ranges.push(vk::AccelerationStructureBuildRangeInfoKHR {
                            primitive_count,
                            ..Default::default()
                        });
                    }
                }
            }
            let mut info = vk::AccelerationStructureBuildGeometryInfoKHR {
                ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                flags: T::build_flags(&mut params, &query_data),
                mode: if should_update {
                    vk::BuildAccelerationStructureModeKHR::UPDATE
                } else {
                    vk::BuildAccelerationStructureModeKHR::BUILD
                },
                src_acceleration_structure: if should_update {
                    blas.accel_struct.as_ref().unwrap().raw
                } else {
                    vk::AccelerationStructureKHR::null()
                },
                geometry_count,
                ..Default::default()
            };
            let build_sizes = unsafe {
                allocator
                    .device()
                    .extension::<ash::extensions::khr::AccelerationStructure>()
                    .get_acceleration_structure_build_sizes(
                        vk::AccelerationStructureBuildTypeKHR::DEVICE,
                        &info,
                        &primitive_counts,
                    )
            };

            let mut accel_struct =
                AccelStruct::new_blas(allocator.clone(), build_sizes.acceleration_structure_size)
                    .unwrap();
            let scratch_size = if info.mode == vk::BuildAccelerationStructureModeKHR::UPDATE {
                build_sizes.update_scratch_size
            } else {
                build_sizes.build_scratch_size
            };
            total_scratch_size =
                total_scratch_size.next_multiple_of(scratch_offset_alignment as u64);
            info.scratch_data = vk::DeviceOrHostAddressKHR {
                device_address: total_scratch_size,
            };
            total_scratch_size += scratch_size;
            accel_struct.flags = info.flags;
            entities.push((entity, accel_struct));
            info
        })
        .collect();

    // Create scratch buffer
    let scratch_buffer = if let Some(s) = scratch_buffer.as_ref() && s.size() >= total_scratch_size {
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
    let scratch_device_address = scratch_buffer.device_address();
    let mut i = 0;
    for info in build_geometry_info.iter_mut() {
        info.p_geometries = &geometries[i as usize];
        unsafe {
            info.scratch_data.device_address += scratch_device_address;
        }
        i += info.geometry_count;
    }

    let build_range_infos = build_geometry_info.iter().map(|info| {
        let slice = &build_ranges[i as usize..(i + info.geometry_count) as usize];
        i += info.geometry_count;
        slice
    });
    let mut recorder = async_task_pool.spawn();
    recorder.build_acceleration_structure(&build_geometry_info, build_range_infos);
    store.pending_task = Some(recorder.finish(BLASBuildPendingTaskInfo { entities, scratch_buffer }));
}
