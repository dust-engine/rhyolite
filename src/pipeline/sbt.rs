use std::{
    alloc::Layout,
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    num::NonZeroU32,
    ops::DerefMut,
    ptr::NonNull,
};

use ash::vk;
use bevy::{
    ecs::{
        component::Component,
        entity::Entity,
        query::{
            Added, ArchetypeFilter, Changed, Or, QueryFilter, QueryItem, ReadOnlyQueryData, With,
        },
        removal_detection::RemovedComponents,
        system::{
            ParamSet, Query, ResMut, Resource, StaticSystemParam, SystemParam, SystemParamItem,
        },
    },
    utils::smallvec::SmallVec,
};
use bytemuck::Pod;
use itertools::Itertools;
use vk_mem::Allocator;

use crate::{
    buffer::BufferLike,
    commands::{ResourceTransitionCommands, TransferCommands},
    dispose::RenderObject,
    ecs::{Barriers, RenderCommands, RenderRes},
    staging::StagingBelt,
    Access, Buffer,
};

use super::{
    ray_tracing::{HitgroupHandle, PipelineGroupManager, RayTracingPipeline},
    HitGroup, PipelineCache,
};

pub struct HitgroupSbtLayout {
    /// The layout for one raytype.
    /// | Raytype 1                                    |
    /// | shader_handles | inline_parameters | padding |
    /// | <--              size           -> | align   |
    pub one_raytype: Layout,

    // The layout for one entry with all its raytypes
    /// | Raytype 1                                    | Raytype 2                                    |
    /// | shader_handles | inline_parameters | padding | shader_handles | inline_parameters | padding |
    /// | <---                                      size                               ---> |  align  |
    pub one_entry: Layout,

    /// The size of the shader group handles, padded.
    /// | Raytype 1                                    |
    /// | shader_handles | inline_parameters | padding |
    /// | <--- size ---> |
    pub handle_size: usize,

    /// The size of the inline params
    /// | Raytype 1                                    |
    /// | shader_handles | inline_parameters | padding |
    ///                  | <---- size -----> |
    pub inline_params_size: usize,
}

pub trait SbtMarker: Send + Sync + 'static {
    /// Type to uniquely identify a SBT entry.
    type HitgroupKey: Send + Sync + Ord + Copy;

    /// The marker component. Any additions of this component will trigger an update of the SBT.
    type Marker: Component;
    type QueryData: ReadOnlyQueryData;
    type QueryFilter: QueryFilter + ArchetypeFilter;
    type Params: SystemParam;

    /// The inline parameters for the hitgroup.
    /// ret is a slice of size inline_params_size * Self::NUM_RAYTYPES.
    fn hitgroup_param(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
        ret: &mut [u8],
    );

    /// Note: For any given entity, this function is assumed to never change.
    /// It will be called only once, when the entity was originally added.
    fn hitgroup_handle(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> HitgroupHandle;

    fn hitgroup_key(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> Self::HitgroupKey;
}

#[derive(Resource)]
pub struct SbtManager<T: SbtMarker, const NUM_RAYTYPES: usize> {
    pipeline_group: PipelineGroupManager<NUM_RAYTYPES>,
    hitgroup_layout: HitgroupSbtLayout,

    allocator: Allocator,
    allocation: Option<RenderRes<Buffer>>,
    capacity: u32,

    /// Number of entries in the SBT.
    free_entries: Vec<u32>,

    /// Mapping from handle and params to sbt_index
    sbt_map_reverse: BTreeMap<T::HitgroupKey, u32>,
    /// Indexed by sbt_index. (num of entities with this SBT index, SbtEntry)
    sbt_map: Vec<Option<(NonZeroU32, HitgroupHandle, Box<[u8]>, T::HitgroupKey)>>,
    /// Mapping from Entity to sbt_index
    entity_map: BTreeMap<Entity, u32>,

    changes: BTreeSet<u32>,

    full_update_required: bool,

    pipelines: Option<SmallVec<[RenderObject<RayTracingPipeline>; NUM_RAYTYPES]>>,
}
// Note: For this to be safe we cannot leak the SbtEntry.
unsafe impl<T: SbtMarker, const NUM_RAYTYPES: usize> Send for SbtManager<T, NUM_RAYTYPES> {}
unsafe impl<T: SbtMarker, const NUM_RAYTYPES: usize> Sync for SbtManager<T, NUM_RAYTYPES> {}

#[derive(Component)]
pub struct SbtHandle<T: SbtMarker> {
    index: u32,
    _marker: PhantomData<T>,
}
impl<T: SbtMarker> Default for SbtHandle<T> {
    fn default() -> Self {
        Self {
            index: u32::MAX,
            _marker: PhantomData,
        }
    }
}

impl<T: SbtMarker, const NUM_RAYTYPES: usize> SbtManager<T, NUM_RAYTYPES>
where
    [(); NUM_RAYTYPES]: Sized,
{
    pub fn hitgroup_layout(&self) -> &HitgroupSbtLayout {
        &self.hitgroup_layout
    }
    pub fn new(
        allocator: Allocator,
        pipeline_group: PipelineGroupManager<NUM_RAYTYPES>,
        hitgroup_layout: HitgroupSbtLayout,
    ) -> Self {
        Self {
            pipeline_group,
            hitgroup_layout,
            allocator,
            allocation: None,
            capacity: 0,
            full_update_required: false,
            free_entries: Vec::new(),
            sbt_map_reverse: BTreeMap::new(),
            sbt_map: Vec::new(),
            entity_map: BTreeMap::new(),
            changes: BTreeSet::new(),
            pipelines: None,
        }
    }
    /// Ensure that the SBT is sufficient for i entries.
    fn resize(&mut self, size: u32) {
        if size == 0 {
            return;
        }
        let size = size.next_power_of_two().max(8);
    }
    fn remove_sbt_index(&mut self, sbt_index: u32) {
        let (num_entities, handle, ptr, key) =
            &mut self.sbt_map[sbt_index as usize].take().unwrap();
        if num_entities.get() == 1 {
            // Last entity with this sbt_index
            self.free_entries.push(sbt_index);
            self.sbt_map_reverse.remove(key).unwrap();
            self.sbt_map[sbt_index as usize] = None;
        } else {
            *num_entities = unsafe { NonZeroU32::new_unchecked(num_entities.get() - 1) };
        }
    }
    fn mark_changes(&mut self, sbt_index: u32) {
        if !self.full_update_required {
            self.changes.insert(sbt_index);
        }
    }

    pub fn extract(
        mut this: ResMut<Self>,
        mut removals: RemovedComponents<T::Marker>,
        mut query: Query<
            (Entity, &mut SbtHandle<T>, T::QueryData),
            (T::QueryFilter, Or<(Added<T::Marker>, Changed<T::Marker>)>),
        >,
        mut params: StaticSystemParam<T::Params>,
    ) {
        let this = &mut *this;
        for entity in removals.read() {
            let sbt_index = this.entity_map.remove(&entity).unwrap();
            this.remove_sbt_index(sbt_index);
        }
        for (entity, handle, data) in query.iter_mut() {
            // For all new entities, allocate their sbt handles.
            let hitgroup_key = T::hitgroup_key(&mut params, &data);

            // Deduplicate
            let sbt_index = if let Some(&sbt_index) = this.sbt_map_reverse.get(&hitgroup_key) {
                this.sbt_map[sbt_index as usize]
                    .as_mut()
                    .unwrap()
                    .0
                    .checked_add(1)
                    .unwrap();
                // Reuse the existing sbt_index
                sbt_index
            } else {
                let hitgroup_handle = T::hitgroup_handle(&mut params, &data);
                let mut hitgroup_params: Vec<u8> =
                    vec![0; this.hitgroup_layout.inline_params_size * NUM_RAYTYPES];
                T::hitgroup_param(&mut params, &data, &mut hitgroup_params);

                // Allocate a new sbt_index
                let sbt_index = this.free_entries.pop().unwrap_or_else(|| {
                    let index = this.sbt_map.len() as u32;
                    this.sbt_map.push(None);
                    index
                });
                assert!(this.sbt_map[sbt_index as usize].is_none());
                unsafe {
                    this.sbt_map[sbt_index as usize] = Some((
                        NonZeroU32::new_unchecked(1),
                        hitgroup_handle,
                        hitgroup_params.into_boxed_slice(),
                        hitgroup_key,
                    ));
                }
                this.sbt_map_reverse.insert(hitgroup_key, sbt_index);
                sbt_index
            };
            this.entity_map.insert(entity, sbt_index);
            let handle = handle.into_inner();
            if handle.index != u32::MAX {
                this.remove_sbt_index(handle.index);
            }
            handle.index = sbt_index;
            this.mark_changes(sbt_index);
        }

        this.resize(this.sbt_map.len() as u32);
    }

    pub fn transfer_barrier(mut this: ResMut<Self>, mut barriers: Barriers) {
        if this.pipelines.is_none() {
            return;
        }
        let full_update_required = this.full_update_required;
        if let Some(allocation) = this.allocation.as_mut() {
            barriers.transition(allocation, Access::COPY_WRITE, !full_update_required, ());
        }
    }

    pub fn transfer(
        mut commands: RenderCommands<'t'>,
        mut this: ResMut<Self>,
        mut staging_belt: ResMut<StagingBelt>,
    ) {
        let this = &mut *this;

        let Some(pipelines) = this.pipelines.as_ref() else {
            return;
        };

        if this.full_update_required {
            this.full_update_required = false;
            let total_size = this
                .hitgroup_layout
                .one_entry
                .repeat(this.sbt_map.len())
                .unwrap()
                .0
                .size() as u64;

            let mut host_buffer = staging_belt
                .start(&mut commands)
                .allocate_buffer(total_size, 1);

            for (i, (_, handle, params, _)) in this
                .sbt_map
                .iter()
                .enumerate()
                .filter_map(|(i, item)| Some((i, item.as_ref()?)))
            {
                this.copy_sbt(&mut host_buffer, i, params, *handle, pipelines);
            }
            let device_buffer = this.allocation.as_mut().unwrap();
            commands.copy_buffer(
                host_buffer.buffer,
                device_buffer.raw_buffer(),
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: total_size,
                }],
            )
        } else if !this.changes.is_empty() {
            let total_size = this
                .hitgroup_layout
                .one_entry
                .repeat(this.changes.len())
                .unwrap()
                .0
                .size() as u64;
            let mut host_buffer = staging_belt
                .start(&mut commands)
                .allocate_buffer(total_size, 1);

            for (i, &sbt_index) in this.changes.iter().enumerate() {
                let (_, handle, params, _) = this.sbt_map[sbt_index as usize].as_ref().unwrap();
                this.copy_sbt(&mut host_buffer, i, params, *handle, pipelines);
            }
            let regions = this
                .changes
                .iter()
                .map(|a| (*a, 1_u32))
                .coalesce(|a, b| {
                    if a.0 + 1 == b.0 {
                        Ok((a.0, a.1 + b.1))
                    } else {
                        Err((a, b))
                    }
                })
                .enumerate()
                .map(|(i, (start, len))| vk::BufferCopy {
                    src_offset: i as u64
                        * this.hitgroup_layout.one_entry.pad_to_align().size() as u64,
                    dst_offset: start as u64
                        * this.hitgroup_layout.one_entry.pad_to_align().size() as u64,
                    size: len as u64 * this.hitgroup_layout.one_entry.pad_to_align().size() as u64,
                })
                .collect::<Vec<_>>();

            let device_buffer = this.allocation.as_mut().unwrap();
            commands.copy_buffer(host_buffer.buffer, device_buffer.raw_buffer(), &regions);
            this.changes.clear();
        }
    }

    fn copy_sbt(
        &self,
        dst_buffer: &mut [u8],
        dst_index: usize,
        raytype_params: &[u8],
        hitgroup_handle: HitgroupHandle,
        pipelines: &[RenderObject<RayTracingPipeline>],
    ) {
        assert_eq!(
            raytype_params.len(),
            self.hitgroup_layout.inline_params_size * NUM_RAYTYPES
        );
        let offset = dst_index * self.hitgroup_layout.one_entry.pad_to_align().size();
        let entry = &mut dst_buffer[offset..offset + self.hitgroup_layout.one_entry.size()];
        for raytype in 0..NUM_RAYTYPES {
            let offset = raytype * self.hitgroup_layout.one_raytype.pad_to_align().size();
            let entry = &mut entry[offset..offset + self.hitgroup_layout.one_raytype.size()];

            let pipeline = &pipelines[self
                .pipeline_group
                .pipeline_index_of_raytype(raytype as u32)
                as usize];
            entry[0..self.hitgroup_layout.handle_size]
                .copy_from_slice(pipeline.get().handles().hitgroup(hitgroup_handle));
            entry[self.hitgroup_layout.handle_size..].copy_from_slice(
                &raytype_params[raytype * self.hitgroup_layout.inline_params_size
                    ..(raytype + 1) * self.hitgroup_layout.inline_params_size],
            );
        }
    }

    pub fn add_hitgroup(
        &mut self,
        hitgroup: HitGroup,
        pipeline_cache: &PipelineCache,
    ) -> HitgroupHandle {
        let handle = self.pipeline_group.add_hitgroup(hitgroup, pipeline_cache);
        self.full_update_required = true; // TODO: Disable this for when pipeline library can be enabled
        handle
    }

    pub fn remove_hitgroup(&mut self, handle: HitgroupHandle) {
        self.pipeline_group.remove_hitgroup(handle);
        self.full_update_required = true; // TODO: Disable this for when pipeline library can be enabled
    }
}
