use std::{
    alloc::Layout,
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    num::NonZeroU32,
    ops::Range,
};

use bevy::{
    app::{App, Plugin, PostUpdate},
    ecs::{
        component::Component,
        entity::Entity,
        query::{Added, ArchetypeFilter, Changed, Or, QueryFilter, QueryItem, ReadOnlyQueryData, Without},
        removal_detection::RemovedComponents,
        schedule::IntoSystemConfigs,
        system::{
            Commands, In, Query, Res, ResMut, Resource, StaticSystemParam, SystemParam,
            SystemParamItem,
        },
    },
    math::UVec3,
    utils::{smallvec::SmallVec, tracing},
};
use bytemuck::{NoUninit, Pod};
use itertools::Itertools;
use rhyolite::{
    ash::{khr::ray_tracing_pipeline::Meta as RayTracingPipelineExt, vk},
    commands::ComputeCommands,
    ecs::IntoRenderSystemConfigs,
    staging::{StagingBeltBatchJob, UniformBelt},
    Allocator, Device, HasDevice,
};

use rhyolite::{
    buffer::BufferLike,
    commands::{ResourceTransitionCommands, TransferCommands},
    dispose::RenderObject,
    ecs::{Barriers, RenderCommands, RenderRes},
    pipeline::PipelineCache,
    staging::StagingBelt,
    Access, Buffer,
};

use crate::pipeline::HitGroup;

use super::pipeline::{HitgroupHandle, PipelineGroupManager, RayTracingPipeline};

pub struct HitgroupSbtLayout {
    num_raytypes: u32,
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
impl HitgroupSbtLayout {
    pub fn extend(&mut self, layout: Layout) {
        let layout = layout.pad_to_align();
        self.inline_params_size += layout.size();
        self.one_raytype = Layout::from_size_align(
            self.handle_size + self.inline_params_size,
            self.one_raytype.align(),
        )
        .unwrap();
        self.one_entry = self
            .one_raytype
            .repeat(self.num_raytypes as usize)
            .unwrap()
            .0;
    }
}

pub trait SBTBuilder: Send + Sync + 'static {
    type QueryData: ReadOnlyQueryData;
    type QueryFilter: QueryFilter + ArchetypeFilter;

    type ChangeFilter: QueryFilter;
    type Params: SystemParam;

    type InlineParam;

    /// The inline parameters for the hitgroup.
    /// ret is a slice of size inline_params_size * Self::NUM_RAYTYPES.
    fn hitgroup_param(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
        raytype: u32,
        ret: &mut Self::InlineParam,
    );

    /// Note: For any given entity, this function is assumed to never change.
    /// It will be called only once, when the entity was originally added.
    fn hitgroup_handle(
        params: &mut SystemParamItem<Self::Params>,
        data: &QueryItem<Self::QueryData>,
    ) -> HitgroupHandle;
}

pub trait PipelineMarker: Resource {
    const NUM_RAYTYPES: usize;
    fn pipelines(&self) -> Option<[&RayTracingPipeline; Self::NUM_RAYTYPES]>;
    fn pipeline_group(&self) -> &PipelineGroupManager<{ Self::NUM_RAYTYPES }>;
}

#[derive(Resource)]
pub struct SbtManager<T: PipelineMarker>
where
    [(); T::NUM_RAYTYPES]: Sized,
{
    hitgroup_layout: HitgroupSbtLayout,

    allocator: Allocator,
    allocation: Option<RenderRes<Buffer>>,

    /// Number of entries in the SBT.
    free_entries: Vec<u32>,

    /// Mapping from Entity to sbt_index
    entity_map: BTreeMap<Entity, u32>,

    full_update_required: bool,
    _marker: PhantomData<*mut T>,
}
unsafe impl<T: PipelineMarker> Send for SbtManager<T> where [(); T::NUM_RAYTYPES]: Sized {}
unsafe impl<T: PipelineMarker> Sync for SbtManager<T> where [(); T::NUM_RAYTYPES]: Sized {}

#[derive(Component)]
pub struct SbtHandle<T> {
    index: u32,
    _marker: PhantomData<*mut T>,
}
unsafe impl<T> Send for SbtHandle<T> {}
unsafe impl<T> Sync for SbtHandle<T> {}
impl<T> Default for SbtHandle<T> {
    fn default() -> Self {
        Self {
            index: u32::MAX,
            _marker: PhantomData,
        }
    }
}

impl<T: PipelineMarker> SbtManager<T>
where
    [(); T::NUM_RAYTYPES]: Sized,
{
    pub fn hitgroup_layout(&self) -> &HitgroupSbtLayout {
        &self.hitgroup_layout
    }
    pub fn new(allocator: Allocator) -> Self {
        let rtx_properties = allocator
            .device()
            .physical_device()
            .properties()
            .get::<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
        let one_raytype_layout = Layout::from_size_align(
            rtx_properties.shader_group_handle_size as usize,
            rtx_properties.shader_group_handle_alignment as usize,
        )
        .unwrap();
        let hitgroup_layout = HitgroupSbtLayout {
            one_raytype: one_raytype_layout,
            one_entry: one_raytype_layout.repeat(T::NUM_RAYTYPES).unwrap().0,
            handle_size: rtx_properties.shader_group_handle_size as usize,
            inline_params_size: 0,
            num_raytypes: T::NUM_RAYTYPES as u32,
        };
        Self {
            hitgroup_layout,
            allocator,
            allocation: None,
            full_update_required: false,
            free_entries: Vec::new(),
            entity_map: BTreeMap::new(),
            _marker: PhantomData,
        }
    }

    // -- Systems
    pub fn assign_index<B: SBTBuilder>(
        mut commands: Commands,
        mut this: ResMut<Self>,
        new_instances: Query<Entity, (B::QueryFilter, Without<SbtHandle<T>>)>,
        mut removed: RemovedComponents<SbtHandle<T>>,
    ) {
        for removed_entity in removed.read() {
            let index = this.entity_map.remove(&removed_entity).unwrap();
            if index != this.entity_map.len() as u32 {
                this.free_entries.push(index);
            }
        }
        for entity in new_instances.iter() {
            assert!(!this.entity_map.contains_key(&entity));
            let index = this.free_entries.pop().unwrap_or(this.entity_map.len() as u32);
            this.entity_map.insert(entity, index);
            commands.entity(entity).insert(SbtHandle::<T> {
                index,
                _marker: PhantomData::default(),
            });
        }
    }
    pub fn resize_buffer(mut this: ResMut<Self>, device: Res<Device>) {
        let total_size = this
            .hitgroup_layout
            .one_entry
            .repeat(this.entity_map.len())
            .unwrap()
            .0
            .size() as u64;
        if total_size == 0 {
            return;
        }
        if let Some(allocation) = this.allocation.as_mut() {
            if allocation.size() <= total_size {
                return;
            }
        }

        let rtx_properties = device
            .physical_device()
            .properties()
            .get::<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
        println!("Total size needed: {}", total_size);
        this.allocation = Some(RenderRes::new(
            Buffer::new_resource(
                this.allocator.clone(),
                total_size,
                rtx_properties.shader_group_base_alignment as u64,
                vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .unwrap(),
        ));
    }

    pub fn transfer_barrier(
        In(mut barriers): In<Barriers>,
        mut this: ResMut<Self>,
        pipelines: Res<T>,
    ) {
        if pipelines.pipelines().is_none() {
            return;
        }
        let full_update_required = this.full_update_required;
        if let Some(allocation) = this.allocation.as_mut() {
            barriers.transition(allocation, Access::COPY_WRITE, !full_update_required, ());
        }
    }

    pub fn transfer<B: SBTBuilder>(
        mut commands: RenderCommands<'t'>,
        mut this: ResMut<Self>,
        mut staging_belt: ResMut<StagingBelt>,
        changed_entries: Query<
            (Entity, &SbtHandle<T>),
            (B::QueryFilter, Or<(B::ChangeFilter, Added<SbtHandle<T>>, Changed<SbtHandle<T>>)>),
        >,
        all_entries: Query<(Entity, B::QueryData, &SbtHandle<T>), B::QueryFilter>,
        mut params: StaticSystemParam<B::Params>,
        pipelines: Res<T>,
    ) {
        let this = &mut *this;
        let pipeline_group = pipelines.pipeline_group();
        let Some(pipelines) = pipelines.pipelines() else {
            return;
        };

        let mut changes;

        if this.full_update_required {
            tracing::info!("{} SBT Full update", std::any::type_name::<T>());
            changes = all_entries
                .iter()
                .map(|(entity, _, handle)| (entity, handle.index))
                .collect::<Vec<_>>();
        } else {
            changes = changed_entries
                .iter()
                .map(|(entity, handle)| (entity, handle.index))
                .collect::<Vec<_>>();
        }
        if changes.is_empty() {
            return;
        }
        tracing::info!(
            "Updating {} SBT for {} entries",
            std::any::type_name::<T>(),
            changes.len()
        );

        changes.sort_unstable_by_key(|n| n.1);

        let total_size = this
            .hitgroup_layout
            .one_entry
            .repeat(changes.len())
            .unwrap()
            .0
            .size() as u64;
        let mut host_buffer = staging_belt
            .start(&mut commands)
            .allocate_buffer(total_size, 1);

        for (i, &(entity, _)) in changes.iter().enumerate() {
            let (_, item, _) = all_entries.get(entity).unwrap();
            let hitgroup_handle = B::hitgroup_handle(&mut params, &item);
            this.copy_sbt(
                &mut host_buffer,
                i,
                hitgroup_handle,
                &pipelines,
                pipeline_group,
                |raytype, dst| {
                    assert_eq!(dst.len(), std::mem::size_of::<B::InlineParam>());
                    B::hitgroup_param(&mut params, &item, raytype, unsafe {
                        &mut *(dst.as_mut_ptr() as *mut B::InlineParam)
                    });
                },
            );
        }
        let regions = changes
            .iter()
            .map(|a| (a.1, 1_u32))
            .coalesce(|a, b| {
                if a.0 + 1 == b.0 {
                    Ok((a.0, a.1 + b.1))
                } else {
                    Err((a, b))
                }
            })
            .enumerate()
            .map(|(i, (start, len))| vk::BufferCopy {
                // Note: this might be problematic. Add StagingBelt base offset.
                src_offset: i as u64 * this.hitgroup_layout.one_entry.pad_to_align().size() as u64,
                dst_offset: start as u64
                    * this.hitgroup_layout.one_entry.pad_to_align().size() as u64,
                size: len as u64 * this.hitgroup_layout.one_entry.pad_to_align().size() as u64,
            })
            .collect::<Vec<_>>();

        let device_buffer = this.allocation.as_mut().unwrap();
        commands.copy_buffer(host_buffer.buffer, device_buffer.raw_buffer(), &regions);
    }

    fn copy_sbt(
        &self,
        dst_buffer: &mut [u8],
        dst_index: usize,
        hitgroup_handle: HitgroupHandle,
        pipelines: &[&RayTracingPipeline],
        pipeline_group: &PipelineGroupManager<{ T::NUM_RAYTYPES }>,
        mut raytype_param_callback: impl FnMut(u32, &mut [u8]),
    ) {
        let offset = dst_index * self.hitgroup_layout.one_entry.pad_to_align().size();
        let entry = &mut dst_buffer[offset..offset + self.hitgroup_layout.one_entry.size()];
        for raytype in 0..T::NUM_RAYTYPES {
            let offset = raytype * self.hitgroup_layout.one_raytype.pad_to_align().size();
            let entry = &mut entry[offset..offset + self.hitgroup_layout.one_raytype.size()];

            let pipeline =
                &pipelines[pipeline_group.pipeline_index_of_raytype(raytype as u32) as usize];
            entry[0..self.hitgroup_layout.handle_size]
                .copy_from_slice(pipeline.handles().hitgroup(hitgroup_handle));
            raytype_param_callback(
                raytype as u32,
                &mut entry[self.hitgroup_layout.handle_size..],
            );
        }
    }
}

pub struct TraceRayBuilder<'a, T: ComputeCommands> {
    pipeline: &'a RayTracingPipeline,
    copy_job: StagingBeltBatchJob<'a>,
    commands: &'a mut T,
    raygen_shader_binding_tables: SmallVec<[vk::StridedDeviceAddressRegionKHR; 1]>,
    miss_shader_binding_tables: vk::StridedDeviceAddressRegionKHR,
    callable_shader_binding_tables: vk::StridedDeviceAddressRegionKHR,
}
impl RayTracingPipeline {
    pub fn trace_rays<'a, T: ComputeCommands>(
        &'a self,
        uniform_belt: &'a mut UniformBelt,
        commands: &'a mut T,
    ) -> TraceRayBuilder<'_, T> {
        TraceRayBuilder {
            pipeline: self,
            copy_job: uniform_belt.start(commands.semaphore_signal()),
            raygen_shader_binding_tables: SmallVec::from_elem(
                vk::StridedDeviceAddressRegionKHR::default(),
                self.handles().num_raygen() as usize,
            ),
            miss_shader_binding_tables: vk::StridedDeviceAddressRegionKHR::default(),
            callable_shader_binding_tables: vk::StridedDeviceAddressRegionKHR::default(),
            commands,
        }
    }
}

impl<T: ComputeCommands> TraceRayBuilder<'_, T> {
    pub fn bind_raygen<D: NoUninit>(&mut self, index: u32, data: &D) -> &mut Self {
        let handle = self.pipeline.handles().rgen(index as usize);
        let data = bytemuck::bytes_of(data);

        let properties = self
            .pipeline
            .device()
            .physical_device()
            .properties()
            .get::<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
        let mut allocation = self.copy_job.allocate_buffer(
            handle.len() as u64 + data.len() as u64,
            properties.shader_group_base_alignment as u64,
        );
        allocation[0..handle.len()].copy_from_slice(handle);
        allocation[handle.len()..].copy_from_slice(data);

        self.raygen_shader_binding_tables[index as usize] = vk::StridedDeviceAddressRegionKHR {
            device_address: allocation.device_address(),
            stride: handle.len() as u64 + data.len() as u64,
            size: handle.len() as u64 + data.len() as u64,
        };
        self
    }
    fn bind_inner<'a, D: NoUninit, I>(&mut self, args: I) -> vk::StridedDeviceAddressRegionKHR
    where
        I: IntoIterator<Item = &'a D>,
        I::IntoIter: ExactSizeIterator,
    {
        let args = args.into_iter();
        let properties = self
            .pipeline
            .device()
            .physical_device()
            .properties()
            .get::<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
        let stride = (properties.shader_group_handle_size + std::mem::size_of::<D>() as u32)
            .next_multiple_of(properties.shader_group_handle_alignment);
        let total_size = stride as u64 * args.len() as u64;
        let mut allocation = self
            .copy_job
            .allocate_buffer(total_size, properties.shader_group_base_alignment as u64);
        for (i, arg) in args.enumerate() {
            let handle = self.pipeline.handles().rmiss(i);
            let arg = bytemuck::bytes_of(arg);
            let entry = &mut allocation[i * stride as usize..(i + 1) * stride as usize];
            entry[0..handle.len()].copy_from_slice(handle);
            entry[handle.len()..].copy_from_slice(arg);
        }
        vk::StridedDeviceAddressRegionKHR {
            device_address: allocation.device_address(),
            stride: stride as u64,
            size: total_size,
        }
    }
    pub fn bind_miss<'a, D: NoUninit, I>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = &'a D>,
        I::IntoIter: ExactSizeIterator,
    {
        self.miss_shader_binding_tables = self.bind_inner(args);
        self
    }
    pub fn bind_callable<'a, D: NoUninit, I>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = &'a D>,
        I::IntoIter: ExactSizeIterator,
    {
        self.callable_shader_binding_tables = self.bind_inner(args);
        self
    }
    pub fn trace(self, raygen_index: usize, extent: UVec3) {
        unsafe {
            let cmd_buf = self.commands.cmd_buf();
            self.pipeline.device().cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline.raw(),
            );
            self.pipeline
                .device()
                .extension::<RayTracingPipelineExt>()
                .cmd_trace_rays(
                    cmd_buf,
                    &self.raygen_shader_binding_tables[raygen_index],
                    &self.miss_shader_binding_tables,
                    &vk::StridedDeviceAddressRegionKHR::default(),
                    &self.callable_shader_binding_tables,
                    extent.x,
                    extent.y,
                    extent.z,
                );
        }
    }

    pub fn trace_indirect(self, raygen_index: usize, indirect_device_address: vk::DeviceAddress) {
        unsafe {
            self.pipeline
                .device()
                .extension::<RayTracingPipelineExt>()
                .cmd_trace_rays_indirect(
                    self.commands.cmd_buf(),
                    &self.raygen_shader_binding_tables[raygen_index],
                    &self.miss_shader_binding_tables,
                    &vk::StridedDeviceAddressRegionKHR::default(),
                    &self.callable_shader_binding_tables,
                    indirect_device_address,
                );
        }
    }
}

pub struct SbtPlugin<T: PipelineMarker, B: SBTBuilder> {
    _marker: PhantomData<(T, B)>,
}
impl<T: PipelineMarker, B: SBTBuilder> Default for SbtPlugin<T, B> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}
impl<T: PipelineMarker, B: SBTBuilder> Plugin for SbtPlugin<T, B>
where
    [(); T::NUM_RAYTYPES]: Sized,
{
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            SbtManager::<T>::assign_index::<B>.before(SbtManager::<T>::resize_buffer),
        );
        app.add_systems(
            PostUpdate,
            SbtManager::<T>::transfer::<B>
                .with_barriers(SbtManager::<T>::transfer_barrier) // TODO: ensure that there's only one instance of this.
                .after(SbtManager::<T>::resize_buffer),
        );
    }
    fn finish(&self, app: &mut App) {
        if app.world.get_resource::<SbtManager<T>>().is_none() {
            app.world.insert_resource(SbtManager::<T>::new(
                app.world.get_resource::<Allocator>().unwrap().clone(),
            ));
            app.add_systems(PostUpdate, SbtManager::<T>::resize_buffer);
        }
        app.world
            .resource_mut::<SbtManager<T>>()
            .hitgroup_layout
            .extend(Layout::new::<B::InlineParam>());
    }
}
