use std::{
    alloc::Layout, collections::{BTreeMap, BTreeSet}, marker::PhantomData, num::NonZeroU32, ops::DerefMut,
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
        system::{ParamSet, Query, ResMut, Resource},
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
    Access, Buffer, DeviceAddressBuffer,
};

use super::ray_tracing::{HitgroupHandle, PipelineGroupManager, RayTracingPipeline};

pub struct HitgroupSbtLayout {
    /// The layout for one raytype.
    /// | Raytype 1                                    |
    /// | shader_handles | inline_parameters | padding |
    /// | <--              size           -> | align   |
    one_raytype: Layout,

    // The layout for one entry with all its raytypes
    /// | Raytype 1                                    | Raytype 2                                    |
    /// | shader_handles | inline_parameters | padding | shader_handles | inline_parameters | padding |
    /// | <---                                      size                               ---> |  align  |
    one_entry: Layout,

    /// The size of the shader group handles, padded.
    /// | Raytype 1                                    |
    /// | shader_handles | inline_parameters | padding |
    /// | <--- size ---> |
    handle_size: usize,
}

pub trait SbtMarker: Send + Sync + 'static {
    /// Typicall a union.
    type HitgroupParam: Pod + Send + Sync + Ord;
    const NUM_RAYTYPES: usize;

    /// The marker component. Any additions of this component will trigger an update of the SBT.
    type Marker: Component;
    type QueryData: ReadOnlyQueryData;
    type QueryFilter: QueryFilter + ArchetypeFilter;

    fn hitgroup_param(data: &QueryItem<Self::QueryData>, raytype: u32) -> Self::HitgroupParam;

    /// Note: For any given entity, this function is assumed to never change.
    /// It will be called only once, when the entity was originally added.
    fn hitgroup(data: &QueryItem<Self::QueryData>) -> HitgroupHandle;
}

type SbtEntry<T: SbtMarker> = Box<(HitgroupHandle, [T::HitgroupParam; T::NUM_RAYTYPES])>;
type SbtEntryRef<T: SbtMarker> = NonNull<(HitgroupHandle, [T::HitgroupParam; T::NUM_RAYTYPES])>;
#[derive(Resource)]
pub struct SbtManager<T: SbtMarker>
where
    [(); T::NUM_RAYTYPES]: Sized,
{
    pipeline_group: PipelineGroupManager<{ T::NUM_RAYTYPES }>,
    hitgroup_layout: HitgroupSbtLayout,

    allocator: Allocator,
    allocation: Option<RenderRes<DeviceAddressBuffer<Buffer>>>,
    capacity: u32,

    /// Number of entries in the SBT.
    free_entries: Vec<u32>,

    /// Mapping from handle and params to sbt_index
    sbt_map_reverse: BTreeMap<SbtEntry<T>, u32>,
    /// Indexed by sbt_index. (num of entities with this SBT index, SbtEntry)
    sbt_map: Vec<Option<(NonZeroU32, SbtEntryRef<T>)>>,
    /// Mapping from Entity to sbt_index
    entity_map: BTreeMap<Entity, u32>,

    changes: BTreeSet<u32>,

    full_update_required: bool,

    pipelines: Option<SmallVec<[RenderObject<RayTracingPipeline>; T::NUM_RAYTYPES]>>,
}
// Note: For this to be safe we cannot leak the SbtEntry.
unsafe impl<T: SbtMarker> Send for SbtManager<T> where [(); T::NUM_RAYTYPES]: Sized {}
unsafe impl<T: SbtMarker> Sync for SbtManager<T> where [(); T::NUM_RAYTYPES]: Sized {}

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

impl<T: SbtMarker> SbtManager<T>
where
    [(); T::NUM_RAYTYPES]: Sized,
{
    pub fn new(
        allocator: Allocator,
        pipeline_group: PipelineGroupManager<{ T::NUM_RAYTYPES }>,
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
        let (num_entities, ptr) = &mut self.sbt_map[sbt_index as usize].take().unwrap();
        if num_entities.get() == 1 {
            // Last entity with this sbt_index
            self.free_entries.push(sbt_index);
            self.sbt_map_reverse.remove(unsafe { ptr.as_ref() });
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
        mut query: ParamSet<(
            Query<
                (Entity, &mut SbtHandle<T>, T::QueryData),
                (T::QueryFilter, Or<(Added<T::Marker>, Changed<T::Marker>)>),
            >,
        )>,
    ) {
        let this = &mut *this;
        for entity in removals.read() {
            let sbt_index = this.entity_map.remove(&entity).unwrap();
            this.remove_sbt_index(sbt_index);
        }
        for (entity, handle, data) in query.p0().iter_mut() {
            // For all new entities, allocate their sbt handles.
            let hitgroup_handle = T::hitgroup(&data);
            let hitgroup_params = (0..T::NUM_RAYTYPES)
                .map(|i| T::hitgroup_param(&data, i as u32))
                .collect::<SmallVec<[T::HitgroupParam; T::NUM_RAYTYPES]>>()
                .into_inner()
                .ok()
                .unwrap();
            // Deduplicate
            let sbt_index = if let Some(&sbt_index) = this
                .sbt_map_reverse
                .get(&(hitgroup_handle, hitgroup_params))
            {
                this.sbt_map[sbt_index as usize]
                    .as_mut()
                    .unwrap()
                    .0
                    .checked_add(1)
                    .unwrap();
                // Reuse the existing sbt_index
                sbt_index
            } else {
                let mut key = Box::new((hitgroup_handle, hitgroup_params));
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
                        NonNull::new_unchecked(key.as_mut() as *mut _),
                    ));
                }
                this.sbt_map_reverse.insert(key, sbt_index);
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

            for (i, (_, r)) in this
                .sbt_map
                .iter()
                .enumerate()
                .filter_map(|(i, item)| Some((i, item.as_ref()?)))
            {
                let (handle, params) = unsafe { r.as_ref() };
                this.copy_sbt(&mut host_buffer, i, *params, *handle, pipelines);
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
                let r = this.sbt_map[sbt_index as usize].as_ref().unwrap().1;
                let (handle, params) = unsafe { r.as_ref() };
                this.copy_sbt(&mut host_buffer, i, *params, *handle, pipelines);
            }
            let regions = this.changes.iter()
            .map(|a| (*a, 1_u32)).coalesce(|a, b| {
                if a.0 + 1 == b.0 {
                    Ok((a.0, a.1 + b.1))
                } else {
                    Err((a, b))
                }
            })
            .enumerate()
            .map(|(i, (start, len))| {
                vk::BufferCopy {
                    src_offset: i as u64 * this.hitgroup_layout.one_entry.pad_to_align().size() as u64,
                    dst_offset: start as u64 * this.hitgroup_layout.one_entry.pad_to_align().size() as u64,
                    size: len as u64 * this.hitgroup_layout.one_entry.pad_to_align().size() as u64,
                }
            })
            .collect::<Vec<_>>();
            
            let device_buffer = this.allocation.as_mut().unwrap();
            commands.copy_buffer(
                host_buffer.buffer,
                device_buffer.raw_buffer(),
                &regions,
            );
            this.changes.clear();
        }
    }

    
    fn copy_sbt(
        &self,
        dst_buffer: &mut [u8],
        dst_index: usize,
        raytype_params: [T::HitgroupParam; T::NUM_RAYTYPES],
        hitgroup_handle: HitgroupHandle,
        pipelines: &[RenderObject<RayTracingPipeline>],
    ) {
        let offset = dst_index * self.hitgroup_layout.one_entry.pad_to_align().size();
        let entry = &mut dst_buffer
            [offset..offset + self.hitgroup_layout.one_entry.size()];
        for (raytype, params) in raytype_params.iter().enumerate() {
            let offset = raytype * self.hitgroup_layout.one_raytype.pad_to_align().size();
            let entry =
                &mut entry[offset..offset + self.hitgroup_layout.one_raytype.size()];

            let pipeline = &pipelines[
                self.pipeline_group
                .pipeline_index_of_raytype(raytype as u32)
                as usize];
            entry[0..self.hitgroup_layout.handle_size]
                .copy_from_slice(pipeline.get().handles().hitgroup(hitgroup_handle));
            entry[self.hitgroup_layout.handle_size..]
                .copy_from_slice(bytemuck::bytes_of(params));
        }
    }
}

