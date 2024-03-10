pub mod queue_cap {
    /// The Q const parameter may take on the following values:
    /// - 'g': Graphics
    /// - 'c': Compute
    /// - 't': Transfer
    /// - 'x': All: Graphics, Compute, Transfer
    pub type QueueCap = char;
    pub trait IsQueueCap<const Q: QueueCap> {}
    impl IsQueueCap<'g'> for () {}
    impl IsQueueCap<'c'> for () {}
    impl IsQueueCap<'t'> for () {}

    pub trait IsGraphicsQueueCap<const Q: QueueCap>: IsQueueCap<Q> {}
    impl IsGraphicsQueueCap<'g'> for () {}

    pub trait IsComputeQueueCap<const Q: QueueCap>: IsQueueCap<Q> {}
    impl IsComputeQueueCap<'c'> for () {}
}

use std::{
    any::Any,
    collections::BTreeMap,
    ffi::CString,
    sync::{atomic::AtomicU64, Arc},
};

use ash::vk::{self, Handle};
use bevy::ecs::{
    archetype::ArchetypeComponentId,
    component::ComponentId,
    system::{lifetimeless::SRes, Res, ResMut, Resource, System, SystemMeta, SystemParam},
    world::World,
};
use itertools::Itertools;
use queue_cap::*;

use crate::{
    command_pool::ManagedCommandPool,
    commands::{CommandRecorder, ImmediateTransitions},
    debug::DebugCommands,
    ecs::Barriers,
    queue::QueueType,
    semaphore::TimelineSemaphore,
    utils::Dispose,
    Device, HasDevice, QueueRef, QueuesRouter,
};

use super::{
    BarrierProducerCell, BoxedBarrierProducer, PerFrameMut, PerFrameResource, PerFrameState,
    RenderSystemConfig,
};
use crate::Access;

#[derive(Resource)]
pub(crate) struct DefaultCommandPool<const Q: char> {
    pool: ManagedCommandPool,
    /// Command buffer for the current stage.
    current_buffer: vk::CommandBuffer,
    /// Command buffer for the next stage, for pre-recording pipeline barriers.
    barrier_buffer: vk::CommandBuffer,
}
impl<const Q: char> DefaultCommandPool<Q> {
    /// Returns the command buffer currently being recorded.
    fn current_buffer(&mut self) -> vk::CommandBuffer {
        if self.current_buffer == vk::CommandBuffer::null() {
            self.current_buffer = self.pool.allocate();
            unsafe {
                self.pool
                    .device()
                    .begin_command_buffer(
                        self.current_buffer,
                        &vk::CommandBufferBeginInfo {
                            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                            ..Default::default()
                        },
                    )
                    .unwrap()
            }
        }
        self.current_buffer
    }
    fn barrier_buffer(&mut self) -> vk::CommandBuffer {
        if self.barrier_buffer == vk::CommandBuffer::null() {
            self.barrier_buffer = self.pool.allocate();
            unsafe {
                self.pool
                    .device()
                    .begin_command_buffer(
                        self.barrier_buffer,
                        &vk::CommandBufferBeginInfo {
                            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                            ..Default::default()
                        },
                    )
                    .unwrap()
            }
        }
        self.barrier_buffer
    }
    fn take_current_buffer(&mut self) -> vk::CommandBuffer {
        let current = self.current_buffer;
        if current != vk::CommandBuffer::null() {
            unsafe {
                self.pool.device().end_command_buffer(current).unwrap();
            }
        }
        current
    }
}
impl<const Q: char> PerFrameResource for DefaultCommandPool<Q> {
    type Params = (SRes<Device>, SRes<QueuesRouter>);

    fn create((device, router): bevy::ecs::system::SystemParamItem<'_, '_, Self::Params>) -> Self {
        let queue_family_index = router.queue_family_of_type(match Q {
            'g' => QueueType::Graphics,
            'c' => QueueType::Compute,
            't' => QueueType::Transfer,
            _ => panic!(),
        });
        let pool = ManagedCommandPool::new(device.clone(), queue_family_index).unwrap();
        DefaultCommandPool {
            pool,
            current_buffer: vk::CommandBuffer::null(),
            barrier_buffer: vk::CommandBuffer::null(),
        }
    }
    fn reset(&mut self, _: bevy::ecs::system::SystemParamItem<'_, '_, Self::Params>) {
        self.pool.reset();
    }
}

/// Command buffers scheduled for submission in the next call to vkQueueSubmit.
#[derive(Resource)]
pub(crate) struct RecordedCommandBuffers<const Q: char>(Vec<vk::CommandBuffer>);

pub struct RenderCommands<'w, 's, const Q: char>
where
    (): IsQueueCap<Q>,
{
    default_cmd_pool: PerFrameMut<'w, DefaultCommandPool<Q>>,
    recorded_command_buffers: ResMut<'w, RecordedCommandBuffers<Q>>,
    retained_objects: &'s mut BTreeMap<u64, Vec<Box<dyn Send + Sync>>>,
    frame_index: u64,
}

impl<'w, 's, const Q: char> RenderCommands<'w, 's, Q>
where
    (): IsQueueCap<Q>,
{
    pub fn retain<T: 'static + Drop + Send + Sync>(&mut self, obj: Dispose<T>) {
        self.retained_objects
            .entry(self.frame_index)
            .or_default()
            .push(Box::new(unsafe { obj.take() }));
    }
    pub fn add_external_command_buffer(&mut self, command_buffer: vk::CommandBuffer) {
        let current = self.default_cmd_pool.take_current_buffer();
        if current != vk::CommandBuffer::null() {
            self.recorded_command_buffers.0.push(current);
        }
        self.recorded_command_buffers.0.push(command_buffer);
    }
}
impl<'w, 's, const Q: char> HasDevice for RenderCommands<'w, 's, Q>
where
    (): IsQueueCap<Q>,
{
    fn device(&self) -> &Device {
        self.default_cmd_pool.pool.device()
    }
}
impl<'w, 's, const Q: char> CommandRecorder for RenderCommands<'w, 's, Q>
where
    (): IsQueueCap<Q>,
{
    const QUEUE_CAP: char = Q;
    fn cmd_buf(&mut self) -> vk::CommandBuffer {
        self.default_cmd_pool.current_buffer()
    }
}

pub struct RenderCommandState<const Q: char> {
    default_cmd_pool_state: PerFrameState<DefaultCommandPool<Q>>,
    recorded_cmd_buf_state: ComponentId,
    retained_objects: BTreeMap<u64, Vec<Box<dyn Send + Sync>>>,
}

unsafe impl<'w, 's, const Q: char> SystemParam for RenderCommands<'w, 's, Q>
where
    (): IsQueueCap<Q>,
{
    type State = RenderCommandState<Q>;

    type Item<'world, 'state> = RenderCommands<'world, 'state, Q>;

    fn init_state(
        world: &mut World,
        system_meta: &mut bevy::ecs::system::SystemMeta,
    ) -> Self::State {
        let default_cmd_pool_state =
            PerFrameMut::<DefaultCommandPool<Q>>::init_state(world, system_meta);
        let recorded_cmd_buf_state =
            ResMut::<RecordedCommandBuffers<Q>>::init_state(world, system_meta);
        world.get_resource_or_insert_with::<RecordedCommandBuffers<Q>>(|| {
            RecordedCommandBuffers(Vec::new())
        });
        RenderCommandState {
            default_cmd_pool_state,
            recorded_cmd_buf_state,
            retained_objects: BTreeMap::new(),
        }
    }
    fn default_configs(config: &mut bevy::utils::ConfigMap) {
        let flags = match Q {
            'g' => QueueType::Graphics,
            'c' => QueueType::Compute,
            't' => QueueType::Transfer,
            _ => unreachable!(),
        };
        let config = config.entry::<RenderSystemConfig>().or_default();
        config.queue = flags;
    }
    fn configurate(state: &mut Self::State, config: &mut dyn Any, world: &mut World) {
        PerFrameMut::configurate(&mut state.default_cmd_pool_state, config, world)
    }
    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &bevy::ecs::system::SystemMeta,
        world: bevy::ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy::ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        let default_cmd_pool = PerFrameMut::<DefaultCommandPool<Q>>::get_param(
            &mut state.default_cmd_pool_state,
            system_meta,
            world,
            change_tick,
        );
        let num_frame_in_flight: u32 = 3;
        if state.default_cmd_pool_state.frame_index > num_frame_in_flight as u64 {
            state
                .retained_objects
                .remove(&(state.default_cmd_pool_state.frame_index - num_frame_in_flight as u64));
        }
        let recorded_command_buffers = ResMut::<RecordedCommandBuffers<Q>>::get_param(
            &mut state.recorded_cmd_buf_state,
            system_meta,
            world,
            change_tick,
        );

        let mut this = RenderCommands {
            frame_index: state.default_cmd_pool_state.frame_index,
            default_cmd_pool,
            retained_objects: &mut state.retained_objects,
            recorded_command_buffers,
        };
        #[cfg(debug_assertions)]
        {
            let str = CString::new(system_meta.name()).unwrap();
            this.begin_debug_label(&str, [0.0, 0.0, 0.0, 1.0]);
        }
        this
    }
}

#[cfg(debug_assertions)]
impl<const Q: char> Drop for RenderCommands<'_, '_, Q>
where
    (): IsQueueCap<Q>,
{
    fn drop(&mut self) {
        self.end_debug_label();
    }
}

#[derive(Debug)]
pub struct BinarySemaphoreOp {
    pub index: u32,
    pub access: Access,
}

#[derive(Clone)]
pub struct TimelineSemaphoreOp {
    pub semaphore: Arc<TimelineSemaphore>,
    pub access: Access,
}

pub struct QueueSystemState {
    pub queue: QueueRef,
    pub frame_index: u64,
    pub binary_signals: Vec<BinarySemaphoreOp>,
    pub binary_waits: Vec<BinarySemaphoreOp>,
    pub timeline_signals: Vec<TimelineSemaphoreOp>,
    pub timeline_waits: Vec<TimelineSemaphoreOp>,
    device: Device,

    /// Map from frame index to retained objects
    pub retained_objects: BTreeMap<u64, Vec<Box<dyn Send + Sync>>>,
    pub fence_to_wait: BTreeMap<u64, vk::Fence>,
}
impl Drop for QueueSystemState {
    fn drop(&mut self) {
        // On destuction, wait for everything to finish execution.
        unsafe {
            if !self.timeline_signals.is_empty() {
                self.timeline_signals[0]
                    .semaphore
                    .wait_blocked(self.frame_index, !0)
                    .unwrap();
            }
            if !self.fence_to_wait.is_empty() {
                let fence_to_wait: Vec<vk::Fence> = self.fence_to_wait.values().cloned().collect();
                self.device
                    .wait_for_fences(&fence_to_wait, true, !0)
                    .unwrap();
            }
            // Now, it's safe to destroy things like `QuerySystemSTate::retained_objects`
            for fence in self.fence_to_wait.values() {
                self.device.destroy_fence(*fence, None);
            }
        }
    }
}

pub struct QueueSystemInitialState {
    pub queue: QueueRef,
    pub timeline_signals: Vec<TimelineSemaphoreOp>,
    pub timeline_waits: Vec<TimelineSemaphoreOp>,
    pub binary_signals: Vec<BinarySemaphoreOp>,
    pub binary_waits: Vec<BinarySemaphoreOp>,
}

pub struct RenderSystemInitialState {
    pub queue: QueueRef,
    pub timeline_signal: Arc<TimelineSemaphore>,
}

pub struct QueueContext<'state, const Q: char>
where
    (): IsQueueCap<Q>,
{
    device: &'state Device,
    pub queue: QueueRef,
    pub frame_index: u64,
    pub binary_signals: &'state [BinarySemaphoreOp],
    pub binary_waits: &'state [BinarySemaphoreOp],
    pub timeline_signals: &'state [TimelineSemaphoreOp],
    pub timeline_waits: &'state [TimelineSemaphoreOp],
    pub(crate) retained_objects: &'state mut Vec<Box<dyn Send + Sync>>,
    fences: &'state mut BTreeMap<u64, vk::Fence>,
}
impl<const Q: char> QueueContext<'_, Q>
where
    (): IsQueueCap<Q>,
{
    pub fn retain<T: 'static + Drop + Send + Sync>(&mut self, obj: Dispose<T>) {
        self.retained_objects.push(unsafe { Box::new(obj.take()) });
    }
    /// Returns a fence that the caller MUST wait on.
    /// Safety:
    /// - A system may either call this method EVERY FRAME, or NEVER
    /// - Once called, the returned fence must be used.
    pub unsafe fn fence_to_wait(&mut self) -> vk::Fence {
        let num_frame_in_flight = 3;

        let mut fence = vk::Fence::null();

        if self.frame_index > num_frame_in_flight {
            let fence_to_recycle = self
                .fences
                .remove(&(self.frame_index - num_frame_in_flight));
            if let Some(fence_to_recycle) = fence_to_recycle {
                unsafe {
                    self.device.reset_fences(&[fence_to_recycle]).unwrap();
                }
                fence = fence_to_recycle;
            }
        }

        if fence == vk::Fence::null() {
            // create a new fence
            unsafe {
                fence = self
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .unwrap()
            }
        };
        let old = self.fences.insert(self.frame_index, fence);
        assert!(old.is_none());
        fence
    }
}

unsafe impl<const Q: char> SystemParam for QueueContext<'_, Q>
where
    (): IsQueueCap<Q>,
{
    type State = QueueSystemState;

    type Item<'world, 'state> = QueueContext<'state, Q>;

    fn init_state(
        world: &mut World,
        _system_meta: &mut bevy::ecs::system::SystemMeta,
    ) -> Self::State {
        QueueSystemState {
            device: world.resource::<Device>().clone(),
            queue: QueueRef::default(),
            binary_signals: Vec::new(),
            binary_waits: Vec::new(),
            timeline_signals: Vec::new(),
            timeline_waits: Vec::new(),
            frame_index: 0,
            retained_objects: BTreeMap::new(),
            fence_to_wait: BTreeMap::new(),
        }
    }

    fn default_configs(config: &mut bevy::utils::ConfigMap) {
        let flags = match Q {
            'g' => QueueType::Graphics,
            'c' => QueueType::Compute,
            't' => QueueType::Transfer,
            _ => unreachable!(),
        };
        let config = config.entry::<RenderSystemConfig>().or_default();
        config.queue = flags;
        config.is_queue_op = true;
    }
    fn configurate(state: &mut Self::State, config: &mut dyn Any, _world: &mut World) {
        if config.is::<QueueSystemInitialState>() {
            let initial_state: &mut QueueSystemInitialState = config.downcast_mut().unwrap();
            state.queue = initial_state.queue;
            state.timeline_signals = std::mem::take(&mut initial_state.timeline_signals);
            state.timeline_waits = std::mem::take(&mut initial_state.timeline_waits);
            state.binary_signals = std::mem::take(&mut initial_state.binary_signals);
            state.binary_waits = std::mem::take(&mut initial_state.binary_waits);
            return;
        }
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        _system_meta: &bevy::ecs::system::SystemMeta,
        _world: bevy::ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        _change_tick: bevy::ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        state.frame_index += 1;

        let num_frame_in_flight = 3;

        if state.frame_index > num_frame_in_flight {
            let wait_value = state.frame_index - num_frame_in_flight;
            if !state.timeline_signals.is_empty() {
                let signaled_semaphore = &state.timeline_signals[0].semaphore;
                // Just waiting on one of those timeline semaphore should be fine, right?
                signaled_semaphore.wait_blocked(wait_value, !0).unwrap();
            }
            if let Some(&fence) = state.fence_to_wait.get(&wait_value) {
                unsafe {
                    state.device.wait_for_fences(&[fence], true, !0).unwrap();
                    // This fence will later be cleaned up by a call to fence_to_wait
                }
            }

            let objs = state.retained_objects.remove(&wait_value).unwrap();
            drop(objs);
        }

        let retained_objects: &mut Vec<Box<dyn Send + Sync>> =
            state.retained_objects.entry(state.frame_index).or_default();

        QueueContext {
            device: &state.device,
            queue: state.queue,
            frame_index: state.frame_index,
            binary_signals: &state.binary_signals,
            binary_waits: &state.binary_waits,
            timeline_signals: &state.timeline_signals,
            timeline_waits: &state.timeline_waits,
            retained_objects,
            fences: &mut state.fence_to_wait,
        }
    }
}

// So, what happens if multiple systems get assigned to the same queue?
// flush_system_graph will only run once for that particular queue.
// If they were assigned to different queues,
// flush_system_graph will run multiple times, once for each queue.
pub(crate) fn flush_system_graph<const Q: char>(
    mut default_cmd_pool: PerFrameMut<DefaultCommandPool<Q>>,
    mut recorded_command_buffers: ResMut<RecordedCommandBuffers<Q>>,
    queue_ctx: QueueContext<Q>,
    binary_semaphore_tracker: Res<RenderSystemsBinarySemaphoreTracker>,
    device: Res<Device>,
) where
    (): IsQueueCap<Q>,
{
    // Take all the command buffers recorded so far, plus the default one.
    let command_buffer: Vec<_> = recorded_command_buffers
        .0
        .iter()
        .cloned()
        .chain(
            std::iter::once(default_cmd_pool.take_current_buffer())
                .filter(|&x| x != vk::CommandBuffer::null()),
        )
        .map(|command_buffer| vk::CommandBufferSubmitInfo {
            command_buffer,
            ..Default::default()
        })
        .collect();
    recorded_command_buffers.0.clear();
    default_cmd_pool.current_buffer = default_cmd_pool.barrier_buffer;
    default_cmd_pool.barrier_buffer = vk::CommandBuffer::null();
    let semaphore_signals = queue_ctx
        .binary_signals
        .iter()
        .map(|op| vk::SemaphoreSubmitInfo {
            semaphore: binary_semaphore_tracker.signal(op.index),
            value: 0,
            stage_mask: op.access.stage,
            ..Default::default()
        })
        .chain(
            queue_ctx
                .timeline_signals
                .iter()
                .map(|op| vk::SemaphoreSubmitInfo {
                    semaphore: op.semaphore.raw(),
                    value: queue_ctx.frame_index,
                    stage_mask: op.access.stage,
                    ..Default::default()
                }),
        )
        .collect::<Vec<_>>();
    let semaphore_waits = queue_ctx
        .binary_waits
        .iter()
        .filter_map(|op| {
            let semaphore = binary_semaphore_tracker.wait(op.index)?;
            let raw = semaphore.raw();
            queue_ctx.retained_objects.push(Box::new(semaphore));
            let info = vk::SemaphoreSubmitInfo {
                semaphore: raw,
                value: 0,
                stage_mask: op.access.stage,
                ..Default::default()
            };
            Some(info)
        })
        .chain(
            queue_ctx
                .timeline_waits
                .iter()
                .map(|op| vk::SemaphoreSubmitInfo {
                    semaphore: op.semaphore.raw(),
                    value: queue_ctx.frame_index,
                    stage_mask: op.access.stage,
                    ..Default::default()
                }),
        )
        .collect::<Vec<_>>();
    unsafe {
        let queue = device.get_raw_queue(queue_ctx.queue);
        device
            .queue_submit2(
                queue,
                &[vk::SubmitInfo2KHR {
                    flags: vk::SubmitFlags::empty(),
                    wait_semaphore_info_count: semaphore_waits.len() as u32,
                    p_wait_semaphore_infos: semaphore_waits.as_ptr(),
                    command_buffer_info_count: command_buffer.len() as u32,
                    p_command_buffer_infos: command_buffer.as_ptr(),
                    signal_semaphore_info_count: semaphore_signals.len() as u32,
                    p_signal_semaphore_infos: semaphore_signals.as_ptr(),
                    ..Default::default()
                }],
                vk::Fence::null(),
            )
            .unwrap();
    }
}

pub(crate) struct InsertPipelineBarrier<const Q: char> {
    barrier_producers: Vec<BoxedBarrierProducer>,
    system_meta: SystemMeta,
    render_command_state: Option<RenderCommandState<Q>>,
    /// If true, will record to `barrier_buffer` instead of `current_buffer`.
    record_to_next: bool,
}
impl<const Q: char> InsertPipelineBarrier<Q> {
    pub(crate) fn new() -> Self {
        InsertPipelineBarrier {
            barrier_producers: Vec::new(),
            system_meta: SystemMeta::new::<Self>(),
            render_command_state: None,
            record_to_next: false,
        }
    }
}

pub(crate) struct InsertPipelineBarrierConfig {
    pub(crate) producers: Vec<BoxedBarrierProducer>,
    pub(crate) record_to_next: bool,
}
impl<const Q: char> System for InsertPipelineBarrier<Q>
where
    (): IsQueueCap<Q>,
{
    type In = ();

    type Out = ();

    fn name(&self) -> std::borrow::Cow<'static, str> {
        if self.barrier_producers.is_empty() {
            return "Empty PipelineBarriers".into();
        }
        let names = self.barrier_producers.iter().map(|x| x.name()).join(",");
        format!("PipelineBarriers({:?})", names).into()
    }

    fn type_id(&self) -> std::any::TypeId {
        std::any::TypeId::of::<InsertPipelineBarrier<Q>>()
    }

    fn component_access(&self) -> &bevy::ecs::query::Access<ComponentId> {
        &self.system_meta.component_access_set.combined_access()
    }

    fn archetype_component_access(&self) -> &bevy::ecs::query::Access<ArchetypeComponentId> {
        &self.system_meta.archetype_component_access
    }

    fn is_send(&self) -> bool {
        self.system_meta.is_send()
    }

    fn is_exclusive(&self) -> bool {
        let mut is_exclusive = false;
        for i in self.barrier_producers.iter() {
            is_exclusive |= i.has_deferred();
        }
        is_exclusive
    }

    fn has_deferred(&self) -> bool {
        self.system_meta.has_deferred()
    }

    unsafe fn run_unsafe(
        &mut self,
        _input: Self::In,
        world: bevy::ecs::world::unsafe_world_cell::UnsafeWorldCell,
    ) -> Self::Out {
        let name = self.name();
        let mut image_barriers = Vec::new();
        let mut buffer_barriers = Vec::new();
        let mut global_barriers = vk::MemoryBarrier2::default();
        let mut dependency_flags = vk::DependencyFlags::empty();
        let change_tick = world.increment_change_tick();
        for i in self.barrier_producers.iter_mut() {
            let mut dropped = false;
            let barrier = Barriers {
                image_barriers: &mut image_barriers,
                buffer_barriers: &mut buffer_barriers,
                global_barriers: &mut global_barriers,
                dependency_flags: &mut dependency_flags,
                dropped: &mut dropped,
            };
            i.run_unsafe(barrier, world);
            assert!(dropped);
        }

        let mut render_commands = RenderCommands::<Q>::get_param(
            self.render_command_state.as_mut().unwrap(),
            &self.system_meta,
            world,
            change_tick,
        );
        let cmd_buf = if self.record_to_next {
            render_commands.default_cmd_pool.barrier_buffer()
        } else {
            render_commands.default_cmd_pool.current_buffer()
        };
        let barriers = ImmediateTransitions {
            cmd_buf,
            device: render_commands.device(),
            global_barriers,
            image_barriers,
            buffer_barriers,
            dependency_flags,
        };
        drop(barriers);
        self.system_meta.last_run = change_tick;
    }

    fn apply_deferred(&mut self, world: &mut World) {
        for i in self.barrier_producers.iter_mut() {
            i.apply_deferred(world);
        }
    }

    fn initialize(&mut self, world: &mut World) {
        for i in self.barrier_producers.iter_mut() {
            i.initialize(world);
        }
        self.render_command_state = Some(RenderCommands::<Q>::init_state(
            world,
            &mut self.system_meta,
        ));
    }

    fn update_archetype_component_access(
        &mut self,
        world: bevy::ecs::world::unsafe_world_cell::UnsafeWorldCell,
    ) {
        for i in self.barrier_producers.iter_mut() {
            i.update_archetype_component_access(world);
        }
        let mut archetype_component_access: bevy::ecs::query::Access<ArchetypeComponentId> =
            bevy::ecs::query::Access::<ArchetypeComponentId>::new();
        for p in self.barrier_producers.iter() {
            archetype_component_access.extend(p.archetype_component_access());
        }
        self.system_meta
            .archetype_component_access
            .extend(&archetype_component_access);
    }

    fn check_change_tick(&mut self, _change_tick: bevy::ecs::component::Tick) {}

    fn get_last_run(&self) -> bevy::ecs::component::Tick {
        self.system_meta.last_run
    }

    fn set_last_run(&mut self, last_run: bevy::ecs::component::Tick) {
        self.system_meta.last_run = last_run;
    }
    fn configurate(&mut self, config: &mut dyn Any, world: &mut World) {
        if let Some(systems) = config.downcast_mut::<InsertPipelineBarrierConfig>() {
            self.record_to_next = systems.record_to_next;
            let systems = std::mem::take(&mut systems.producers);
            assert!(self.barrier_producers.is_empty());
            self.barrier_producers = systems;

            let mut component_access = bevy::ecs::query::Access::<ComponentId>::new();
            let mut archetype_component_access: bevy::ecs::query::Access<ArchetypeComponentId> =
                bevy::ecs::query::Access::<ArchetypeComponentId>::new();
            for p in self.barrier_producers.iter() {
                component_access.extend(p.component_access());
                archetype_component_access.extend(p.archetype_component_access());
            }
            self.system_meta
                .component_access_set
                .extend_combined_access(&component_access);
            self.system_meta.archetype_component_access = archetype_component_access;
            self.system_meta.name = self.name();
        } else {
            RenderCommands::<Q>::configurate(
                self.render_command_state.as_mut().unwrap(),
                config,
                world,
            );
        }
    }
}

pub struct RecycledBinarySemaphore {
    semaphore: vk::Semaphore,
    sender: crossbeam_channel::Sender<vk::Semaphore>,
}
impl RecycledBinarySemaphore {
    pub fn raw(&self) -> vk::Semaphore {
        self.semaphore
    }
}
impl Drop for RecycledBinarySemaphore {
    fn drop(&mut self) {
        self.sender.send(self.semaphore).unwrap();
    }
}
#[derive(Resource)]
pub struct RenderSystemsBinarySemaphoreTracker {
    device: Device,
    sender: crossbeam_channel::Sender<vk::Semaphore>,
    receiver: crossbeam_channel::Receiver<vk::Semaphore>,
    semaphores: Vec<AtomicU64>,
}
impl Drop for RenderSystemsBinarySemaphoreTracker {
    fn drop(&mut self) {
        while let Ok(sem) = self.receiver.try_recv() {
            unsafe {
                self.device.destroy_semaphore(sem, None);
            }
            for sem in self.semaphores.iter_mut() {
                let val = sem.get_mut();
                assert!(*val == 0);
            }
        }
    }
}
impl RenderSystemsBinarySemaphoreTracker {
    pub fn new(device: Device, max_semaphore: usize) -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded();
        Self {
            device,
            sender,
            receiver,
            semaphores: (0..max_semaphore).map(|_| AtomicU64::new(0)).collect(),
        }
    }
    pub fn signal(&self, index: u32) -> vk::Semaphore {
        let semaphore = match self.receiver.try_recv() {
            Ok(semaphore) => semaphore,
            Err(crossbeam_channel::TryRecvError::Empty) => {
                let semaphore = unsafe {
                    self.device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .unwrap()
                };
                semaphore
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => panic!(),
        };
        self.semaphores[index as usize]
            .compare_exchange(
                0,
                semaphore.as_raw(),
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            )
            .expect("Double signal");
        semaphore
    }
    pub fn wait(&self, index: u32) -> Option<RecycledBinarySemaphore> {
        let semaphore =
            self.semaphores[index as usize].swap(0, std::sync::atomic::Ordering::Relaxed);
        let semaphore = vk::Semaphore::from_raw(semaphore);
        if semaphore == vk::Semaphore::null() {
            None
        } else {
            Some(RecycledBinarySemaphore {
                semaphore,
                sender: self.sender.clone(),
            })
        }
    }
}

pub(crate) struct BarrierProducerOutConfig {
    /// This is a pointer from Arc::<BarrierProducerCell<O>>::into_raw()
    pub barrier_producer_output_cell: *const (),
    pub barrier_producer_output_type: std::any::TypeId,
}

pub struct BarrierProducerOut<T>(pub T);
unsafe impl<T: 'static> SystemParam for BarrierProducerOut<T> {
    type State = Option<Arc<BarrierProducerCell<T>>>;

    type Item<'world, 'state> = BarrierProducerOut<T>;

    fn init_state(world: &mut World, system_meta: &mut SystemMeta) -> Self::State {
        None
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &SystemMeta,
        world: bevy::ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy::ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        let state = state.as_mut().unwrap();
        let cell = state.0.get().as_mut().unwrap();
        Self(cell.take().unwrap())
    }

    fn configurate(state: &mut Self::State, config: &mut dyn Any, _world: &mut World) {
        let Some(config) = config.downcast_mut::<Option<BarrierProducerOutConfig>>() else {
            return;
        };
        let config = config.take().unwrap();
        if config.barrier_producer_output_type != std::any::TypeId::of::<T>() {
            panic!("Type mismatch");
        }
        if config.barrier_producer_output_cell.is_null() {
            panic!()
        }
        let cell = unsafe {
            Arc::from_raw(config.barrier_producer_output_cell as *const BarrierProducerCell<T>)
        };
        assert!(state.is_none());
        state.replace(cell);
    }
}
