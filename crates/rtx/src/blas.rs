use std::{ffi::CString, ops::Deref};

use bevy::{
    app::{App, Plugin, PostUpdate},
    ecs::{
        component::Component,
        entity::Entity,
        query::{ArchetypeFilter, QueryFilter, QueryItem, ReadOnlyQueryData},
        system::{
            Commands, Local, Query, Res, ResMut, StaticSystemParam, SystemParam, SystemParamItem,
        },
    },
    prelude::{FromWorld, IntoSystemConfigs, Resource, Without},
    utils::tracing,
};
use rhyolite::{
    ash::{khr::acceleration_structure::Meta as AccelerationStructureExt, vk}, buffer::BufferLike, debug::DebugObject, Allocator, Device, HasDevice, QueryPool
};

use crate::AccelStruct;
#[derive(Component)]
pub struct BLAS {
    accel_struct: AccelStruct,
}
impl Deref for BLAS {
    type Target = AccelStruct;
    fn deref(&self) -> &Self::Target {
        &self.accel_struct
    }
}

pub trait BLASBuilder: Send + Sync + 'static {
    /// Associated entities to be passed.
    type QueryData: ReadOnlyQueryData;

    /// Note: If the BLAS will never be updated, you may add Without<BLAS> here
    /// to exclude all entities with BLAS already built.
    type QueryFilter: QueryFilter + ArchetypeFilter;
    /// Additional system entities to be passed.
    type Params: SystemParam;

    #[allow(unused_variables)]
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
    #[allow(unused_variables)]
    fn should_update(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> bool {
        false
    }

    type BufferType: BufferLike + Send;
    type GeometryIterator<'a>: Iterator<Item = BLASBuildGeometry<Self::BufferType>> + 'a;
    /// The geometries to be built. The implementation shall write directly into the dst buffer.
    /// The iterator returned shall contain offset values into the dst buffer.
    fn geometries<'a>(
        params: &'a mut SystemParamItem<Self::Params>,
        data: &'a QueryItem<Self::QueryData>,
        commands: &mut impl TransferCommands,
    ) -> Self::GeometryIterator<'a>;
}

pub enum BLASBuildGeometry<T> {
    Triangles {
        vertex_format: vk::Format,
        vertex_data: T,
        vertex_stride: vk::DeviceSize,
        max_vertex: u32,
        index_type: vk::IndexType,
        index_data: T,
        transform_data: Option<vk::TransformMatrixKHR>,
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
    ) -> BLASBuildGeometry<vk::DeviceOrHostAddressConstKHR> {
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
                transform_data,
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

fn build_blas_system<T: BLASBuilder>(
    mut commands: Commands,
    mut task: Local<Option<AsyncComputeTask<BuildTask<T::BufferType>>>>,
    device: Res<Device>,
    allocator: Res<Allocator>,
    entities: Query<(Entity, T::QueryData, Option<&mut BLAS>), T::QueryFilter>,
    mut params: StaticSystemParam<T::Params>,
    mut task_pool: ResMut<AsyncTaskPool>,
) {
    if let Some(task_ref) = task.as_mut() {
        if !task_ref.is_finished() {
            return;
        }
        let task = task_pool.wait_blocked(task.take().unwrap());
        for (entity, blas) in task.built_accel_structs {
            commands.entity(entity).insert(BLAS { accel_struct: blas });
        }
        // Return here so that any new BLAS builds will be scheduled in the next frame.
        // This is important, otherwise we're going to rebuild what we just built.
        // `entities` won't reflect the changes we made to `commands`.
        return;
    }
    if entities.is_empty() {
        return;
    }
    let mut infos: Vec<vk::AccelerationStructureBuildGeometryInfoKHR> = Vec::new();
    let mut info_entities: Vec<Entity> = Vec::new();
    let mut geometries: Vec<vk::AccelerationStructureGeometryKHR> = Vec::new();
    let mut buffers: Vec<T::BufferType> = Vec::new();
    let mut build_ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR> = Vec::new();
    let mut transforms: Vec<vk::TransformMatrixKHR> = Vec::new();

    let mut commands = task_pool.spawn_transfer();

    for (entity, data, blas) in entities.iter() {
        if blas.is_some() && !T::should_update(&mut params, &data) {
            continue;
        }
        let mut info = vk::AccelerationStructureBuildGeometryInfoKHR {
            ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            flags: T::build_flags(&mut params, &data),
            mode: if let Some(blas) = blas
                && blas
                    .flags
                    .contains(vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE)
            {
                vk::BuildAccelerationStructureModeKHR::UPDATE
            } else {
                vk::BuildAccelerationStructureModeKHR::BUILD
            },
            src_acceleration_structure: blas.map(|b| b.raw).unwrap_or_default(),
            ..Default::default()
        };
        for geometry in T::geometries(&mut params, &data, &mut commands) {
            info.geometry_count += 1;
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
                    build_ranges.push(vk::AccelerationStructureBuildRangeInfoKHR {
                        primitive_count: primitive_count,
                        primitive_offset: 0,
                        first_vertex: 0,
                        transform_offset: transforms.len() as u32,
                    });
                    if let Some(transform) = transform_data {
                        transforms.push(transform);
                    }
                    geometries.push(vk::AccelerationStructureGeometryKHR {
                        geometry_type: vk::GeometryTypeKHR::TRIANGLES,
                        geometry: vk::AccelerationStructureGeometryDataKHR {
                            triangles: vk::AccelerationStructureGeometryTrianglesDataKHR {
                                vertex_format: vertex_format,
                                vertex_data: vk::DeviceOrHostAddressConstKHR {
                                    device_address: vertex_data.device_address(),
                                },
                                vertex_stride: vertex_stride,
                                max_vertex: max_vertex,
                                index_type: index_type,
                                index_data: vk::DeviceOrHostAddressConstKHR {
                                    device_address: index_data.device_address(),
                                },
                                ..Default::default()
                            },
                        },
                        flags: flags,
                        ..Default::default()
                    });
                    buffers.push(vertex_data);
                    buffers.push(index_data);
                }
                BLASBuildGeometry::Aabbs {
                    buffer,
                    stride,
                    flags,
                    primitive_count,
                } => {
                    build_ranges.push(vk::AccelerationStructureBuildRangeInfoKHR {
                        primitive_count: primitive_count,
                        primitive_offset: 0,
                        first_vertex: 0,
                        transform_offset: 0,
                    });
                    geometries.push(vk::AccelerationStructureGeometryKHR {
                        geometry_type: vk::GeometryTypeKHR::AABBS,
                        geometry: vk::AccelerationStructureGeometryDataKHR {
                            aabbs: vk::AccelerationStructureGeometryAabbsDataKHR {
                                data: vk::DeviceOrHostAddressConstKHR {
                                    device_address: buffer.device_address(),
                                },
                                stride: stride,
                                ..Default::default()
                            },
                        },
                        flags: flags,
                        ..Default::default()
                    });
                    buffers.push(buffer);
                }
            }
        }
        infos.push(info);
        info_entities.push(entity);
    }
    if infos.is_empty() {
        return;
    }

    let mut cur_geometry_index = 0;
    let mut max_primitive_counts: Vec<u32> = Vec::new();
    let mut scratch_buffers: Vec<Buffer> = Vec::new();
    let mut built_accel_structs: Vec<(Entity, AccelStruct)> = Vec::new();
    let scratch_offset_alignment: u32 = allocator
        .device()
        .physical_device()
        .properties()
        .get::<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>()
        .min_acceleration_structure_scratch_offset_alignment;
    for (info, entity) in infos.iter_mut().zip(info_entities.into_iter()) {
        info.p_geometries = unsafe { geometries.as_ptr().add(cur_geometry_index) };
        max_primitive_counts.clear();
        max_primitive_counts.extend(
            build_ranges
                .iter()
                .skip(cur_geometry_index)
                .map(|r| r.primitive_count)
                .take(info.geometry_count as usize),
        );
        cur_geometry_index += info.geometry_count as usize;
        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            device
                .extension::<AccelerationStructureExt>()
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &info,
                    &max_primitive_counts,
                    &mut size_info,
                )
        };

        let scratch_buffer = Buffer::new_resource(
            allocator.clone(),
            if info.mode == vk::BuildAccelerationStructureModeKHR::UPDATE {
                size_info.update_scratch_size
            } else {
                size_info.build_scratch_size
            },
            scratch_offset_alignment as u64,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .unwrap();
        info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.device_address(),
        };

        let accel_struct = AccelStruct::new(
            allocator.clone(),
            size_info.acceleration_structure_size,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        )
        .unwrap()
        .with_name(
            CString::new(format!("BLAS for entity {:?}", entity))
                .unwrap()
                .as_c_str(),
        );
        info.dst_acceleration_structure = accel_struct.raw;
        scratch_buffers.push(scratch_buffer);
        built_accel_structs.push((entity, accel_struct));
    }

    let mut cmd_recorder = commands.commit::<'c'>(
        vk::PipelineStageFlags2::empty(),
        vk::PipelineStageFlags2::TRANSFER,
    );

    cur_geometry_index = 0;
    let build_range_infos = infos.iter().map(|info| {
        let slice =
            &build_ranges[cur_geometry_index..cur_geometry_index + info.geometry_count as usize];
        cur_geometry_index += info.geometry_count as usize;
        slice
    });
    cmd_recorder.build_acceleration_structure(&infos, build_range_infos);

    *task = Some(cmd_recorder.finish(
        BuildTask {
            _scratch_buffers: scratch_buffers,
            _buffers: buffers,
            built_accel_structs,
        },
        vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
    ));
}

struct BuildTask<T> {
    _scratch_buffers: Vec<Buffer>,
    _buffers: Vec<T>,
    built_accel_structs: Vec<(Entity, AccelStruct)>,
}

pub struct BLASBuilderPlugin<T: BLASBuilder> {
    _marker: std::marker::PhantomData<T>,
}
impl<T: BLASBuilder> Default for BLASBuilderPlugin<T> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: BLASBuilder> Plugin for BLASBuilderPlugin<T> {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            build_blas_system::<T>.before(blas_compaction_system),
        );
    }
}

#[derive(Component)]
pub struct BLASCompacted;

#[derive(Component)]
pub struct BLASProperties {
    compacted_size: vk::DeviceSize,
}

#[derive(Resource, Default)]
pub(crate) struct BLASCompactionTask {
    /// Array of (Entity, original_size) pairs
    queried_entities: Vec<(Entity, u64)>,
    query_pool: Option<RenderObject<QueryPool>>,
    compacted_entities: Vec<(Entity, AccelStruct)>,
    task: Option<AsyncComputeTask<()>>,
}
pub(crate) fn blas_compaction_system(
    mut commands: Commands,
    mut task_pool: ResMut<AsyncTaskPool>,
    mut pending_query_task: ResMut<BLASCompactionTask>,
    mut existing_blas: Query<&mut BLAS>,
    // mut render_commands: RenderCommands<'c'>,
) {
    if let Some(pending_task) = pending_query_task.task.as_ref() {
        // Has pending task
        if !pending_task.is_finished() {
            return;
        }
        // Pending task finished
        let pending_task = pending_query_task.task.take().unwrap();
        task_pool.wait_blocked(pending_task);

        // Insert all compacted BLAS
        for (entity, accel_struct) in pending_query_task.compacted_entities.drain(..) {
            commands.entity(entity).insert(BLASCompacted);
            let old_blas = std::mem::replace(
                &mut existing_blas.get_mut(entity).unwrap().accel_struct,
                accel_struct,
            );
            //RenderObject::new(old_blas).use_on(&mut render_commands); // Defer dropping the BLAS until after the current frame finishes
        }

        // If did query, get query results
        if let Some(pool) = pending_query_task.query_pool.take() {
            assert!(!pending_query_task.queried_entities.is_empty());
            let mut results = vec![0_u64; pending_query_task.queried_entities.len()];
            pool.get().get_results_u64(0, &mut results).unwrap();
            for ((entity, original_size), compacted_size) in
                pending_query_task.queried_entities.drain(..).zip(results)
            {
                commands
                    .entity(entity)
                    .insert(BLASProperties { compacted_size });
            }
        }
    }
}

pub(crate) fn blas_compaction_system_schedule(
    mut commands: Commands,
    query_candidates: Query<(Entity, &BLAS), Without<BLASProperties>>,
    copy_candidates: Query<(Entity, &BLAS, &BLASProperties), Without<BLASCompacted>>,
    mut task_pool: ResMut<AsyncTaskPool>,
    device: Res<Device>,
    allocator: Res<Allocator>,
    mut pending_query_task: ResMut<BLASCompactionTask>,
) {
    if pending_query_task.task.is_some() {
        return;
    }
    if query_candidates.is_empty() && copy_candidates.is_empty() {
        return;
    }
    let mut task = task_pool.spawn_compute();

    let query_accel_structs = query_candidates
        .iter()
        .map(|x| x.1.accel_struct.raw)
        .collect::<Vec<_>>();
    let query_pool = if !query_accel_structs.is_empty() {
        let pool = QueryPool::new(
            device.clone(),
            vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
            query_accel_structs.len() as u32,
        )
        .unwrap();
        let mut pool = RenderObject::new(pool);
        task.reset_query_pool(&mut pool, ..);
        task.write_acceleration_structures_properties(&query_accel_structs, &mut pool, 0);
        Some(pool)
    } else {
        None
    };

    assert!(pending_query_task.compacted_entities.is_empty());

    pending_query_task.compacted_entities = copy_candidates
        .iter()
        .filter_map(|(entity, blas, blas_properties)| {
            if blas.size() * 9 / 10 <= blas_properties.compacted_size {
                tracing::debug!(
                    "Skipped BLAS compaction for {:?}: {} -> {}",
                    entity,
                    blas.size(),
                    blas_properties.compacted_size
                );
                commands.entity(entity).insert(BLASCompacted);
                return None;
            }
            let compacted_blas = AccelStruct::new(
                allocator.clone(),
                blas_properties.compacted_size,
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            )
            .unwrap();
            task.copy_acceleration_structure(&vk::CopyAccelerationStructureInfoKHR {
                src: blas.raw,
                dst: compacted_blas.raw,
                mode: vk::CopyAccelerationStructureModeKHR::COMPACT,
                ..Default::default()
            });
            tracing::debug!(
                "Compacted BLAS for {:?}: {} -> {}",
                entity,
                blas.size(),
                compacted_blas.size()
            );
            Some((entity, compacted_blas))
        })
        .collect();

    pending_query_task
        .task
        .replace(task.finish((), vk::PipelineStageFlags2::empty()));
    pending_query_task.query_pool = query_pool;
    pending_query_task.queried_entities = query_candidates
        .iter()
        .map(|(entity, blas)| (entity, blas.size()))
        .collect();
}
