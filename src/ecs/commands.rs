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
    borrow::Cow,
    collections::BTreeMap,
    ffi::CString,
    sync::{atomic::AtomicU64, Arc, Mutex},
};

use ash::vk::{self, Handle};
use bevy::{
    core::FrameCount,
    ecs::{
        archetype::ArchetypeComponentId,
        component::ComponentId,
        system::{Res, ResMut, Resource, System, SystemMeta, SystemParam},
        world::World,
    },
    utils::smallvec::SmallVec,
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
    BarrierProducerCell, BarriersPrevStage, BoxedBarrierProducer, PerFrame, PerFrameResource,
    RenderSystemConfig,
};
use crate::Access;

#[derive(Resource)]
pub(crate) struct DefaultCommandPool<const Q: char> {
    pool: ManagedCommandPool,
    buffer: vk::CommandBuffer,
}
impl<const Q: char> DefaultCommandPool<Q> {
    /// Returns the command buffer currently being recorded.
    fn current_buffer(&mut self) -> vk::CommandBuffer {
        if self.buffer == vk::CommandBuffer::null() {
            self.buffer = self.pool.allocate();
            unsafe {
                self.pool
                    .device()
                    .begin_command_buffer(
                        self.buffer,
                        &vk::CommandBufferBeginInfo {
                            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                            ..Default::default()
                        },
                    )
                    .unwrap()
            }
        }
        self.buffer
    }
    fn take_current_buffer(&mut self) -> vk::CommandBuffer {
        let current = self.buffer;
        self.buffer = vk::CommandBuffer::null();
        if current != vk::CommandBuffer::null() {
            unsafe {
                self.pool.device().end_command_buffer(current).unwrap();
            }
        }
        current
    }
}
impl<const Q: char> PerFrameResource for DefaultCommandPool<Q> {
    type Param<'a> = (&'a Device, &'a QueuesRouter);

    fn create((device, router): (&Device, &QueuesRouter)) -> Self {
        let queue_family_index = router.queue_family_of_type(match Q {
            'g' => QueueType::Graphics,
            'c' => QueueType::Compute,
            't' => QueueType::Transfer,
            _ => panic!(),
        });
        let pool = ManagedCommandPool::new(device.clone(), queue_family_index).unwrap();
        DefaultCommandPool {
            pool,
            buffer: vk::CommandBuffer::null(),
        }
    }
    fn reset(&mut self, _: (&Device, &QueuesRouter)) {
        self.pool.reset();
        self.buffer = vk::CommandBuffer::null();
    }
}

#[derive(Default)]
pub struct QueueSubmissionInfo {
    pub(crate) cmd_bufs: SmallVec<[vk::CommandBuffer; 4]>,
    pub(crate) last_buf_open: bool,
    pub(crate) trailing_buffer_barriers: Vec<vk::BufferMemoryBarrier2>,
    pub(crate) trailing_image_barriers: Vec<vk::ImageMemoryBarrier2>,

    pub(crate) signal_semaphores:
        SmallVec<[(vk::PipelineStageFlags2, Arc<TimelineSemaphore>, u64); 4]>,
    pub(crate) wait_semaphores:
        SmallVec<[(vk::PipelineStageFlags2, Arc<TimelineSemaphore>, u64); 4]>,
    pub(crate) available_semaphores: Vec<(vk::PipelineStageFlags2, Arc<TimelineSemaphore>, u64)>,
}
unsafe impl Send for QueueSubmissionInfo {}
unsafe impl Sync for QueueSubmissionInfo {}
impl QueueSubmissionInfo {
    pub fn wait_semaphore(
        &mut self,
        semaphore: Cow<Arc<TimelineSemaphore>>,
        value: u64,
        stage: vk::PipelineStageFlags2,
    ) {
        for (_, current_semaphore, current_value) in self.signal_semaphores.iter() {
            if Arc::ptr_eq(current_semaphore, semaphore.as_ref()) {
                if current_value >= &value {
                    return;
                }
                break;
            }
        }
        for (current_stage, current_semaphore, current_value) in self.wait_semaphores.iter_mut() {
            if Arc::ptr_eq(current_semaphore, semaphore.as_ref()) {
                *current_value = (*current_value).max(value);
                *current_stage |= stage;
            }
        }
        self.wait_semaphores
            .push((stage, semaphore.into_owned(), value));
    }

    pub fn signal_semaphore(
        &mut self,
        stage: vk::PipelineStageFlags2,
        device: &Device,
    ) -> (Arc<TimelineSemaphore>, u64) {
        for (current_stage, current_semaphore, current_value) in self.signal_semaphores.iter_mut() {
            if *current_stage == stage {
                return (current_semaphore.clone(), *current_value);
            }
        }
        for (current_stage, current_semaphore, current_value) in
            self.available_semaphores.iter_mut()
        {
            if *current_stage == stage {
                self.signal_semaphores.push((
                    *current_stage,
                    current_semaphore.clone(),
                    *current_value + 1,
                ));
                *current_value += 1;
                return (current_semaphore.clone(), *current_value);
            }
        }
        let semaphore = Arc::new(TimelineSemaphore::new(device.clone()).unwrap());
        self.available_semaphores
            .push((stage, semaphore.clone(), 1));
        self.signal_semaphores.push((stage, semaphore.clone(), 1));
        (semaphore, 1)
    }
}

pub struct RenderCommands<'w, 's, const Q: char>
where
    (): IsQueueCap<Q>,
{
    default_cmd_pool: &'w mut DefaultCommandPool<Q>,
    retained_objects: &'s mut BTreeMap<u64, Vec<Box<dyn Send + Sync>>>,
    pub(crate) frame_index: u64,
    pub(crate) submission_info: &'s Mutex<QueueSubmissionInfo>,
    prev_stage_submission_info: &'s [Option<Arc<Mutex<QueueSubmissionInfo>>>],
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
        let mut submission_info = self.submission_info.lock().unwrap();
        if current != vk::CommandBuffer::null() {
            submission_info.cmd_bufs.push(current);
        }
        submission_info.cmd_bufs.push(command_buffer);
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
    fn current_queue_family(&self) -> (QueueType, u32) {
        (
            match Q {
                'g' => QueueType::Graphics,
                'c' => QueueType::Compute,
                't' => QueueType::Transfer,
                _ => unreachable!(),
            },
            self.default_cmd_pool.pool.queue_family_index(),
        )
    }
    fn cmd_buf(&mut self) -> vk::CommandBuffer {
        self.default_cmd_pool.current_buffer()
    }
}

pub struct RenderCommandState<const Q: char> {
    device: Device,
    default_cmd_pool_state: ComponentId,
    queues_router_state: ComponentId,
    retained_objects: BTreeMap<u64, Vec<Box<dyn Send + Sync>>>,
    submission_info: Option<Arc<Mutex<QueueSubmissionInfo>>>,
    prev_stage_submission_info: SmallVec<[Option<Arc<Mutex<QueueSubmissionInfo>>>; 4]>,
    frame_index_state: ComponentId,
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
            ResMut::<PerFrame<DefaultCommandPool<Q>>>::init_state(world, system_meta);
        let queues_router_state = Res::<QueuesRouter>::init_state(world, system_meta);
        let frame_index_state = Res::<FrameCount>::init_state(world, system_meta);
        RenderCommandState {
            device: world.resource::<Device>().clone(),
            default_cmd_pool_state,
            queues_router_state,
            retained_objects: BTreeMap::new(),
            submission_info: None,
            prev_stage_submission_info: SmallVec::new(),
            frame_index_state,
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
        if let Some(a) = config.downcast_mut::<RenderSystemInitialState>() {
            state.submission_info = Some(a.queue_submission_info.clone());
            state.prev_stage_submission_info = a.prev_stage_queue_submission_info.clone();
            return;
        }
    }
    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &bevy::ecs::system::SystemMeta,
        world: bevy::ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy::ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        let current_frame_index = Res::<FrameCount>::get_param(
            &mut state.frame_index_state,
            system_meta,
            world,
            change_tick,
        )
        .into_inner()
        .0;
        let default_cmd_pool: ResMut<'world, _> =
            ResMut::<PerFrame<DefaultCommandPool<Q>>>::get_param(
                &mut state.default_cmd_pool_state,
                system_meta,
                world,
                change_tick,
            );
        let queues_router = Res::<QueuesRouter>::get_param(
            &mut state.queues_router_state,
            system_meta,
            world,
            change_tick,
        );
        let default_cmd_pool: &'world mut _ = default_cmd_pool.into_inner();
        let default_cmd_pool = default_cmd_pool.on_frame_index(
            current_frame_index,
            state.submission_info.as_ref().unwrap(),
            &state.device,
            (&state.device, &queues_router),
        );
        let num_frame_in_flight: u32 = 3;
        if current_frame_index >= num_frame_in_flight as u64 {
            state
                .retained_objects
                .remove(&(current_frame_index - num_frame_in_flight as u64));
        }

        let mut this = RenderCommands {
            frame_index: current_frame_index,
            default_cmd_pool,
            retained_objects: &mut state.retained_objects,
            submission_info: state.submission_info.as_ref().unwrap(),
            prev_stage_submission_info: &state.prev_stage_submission_info,
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

pub struct SubmissionInfo<'s>(&'s Mutex<QueueSubmissionInfo>);

unsafe impl<'s> SystemParam for SubmissionInfo<'s> {
    type State = Option<Arc<Mutex<QueueSubmissionInfo>>>;

    type Item<'world, 'state> = SubmissionInfo<'state>;

    fn init_state(
        world: &mut World,
        system_meta: &mut bevy::ecs::system::SystemMeta,
    ) -> Self::State {
        None
    }
    fn configurate(state: &mut Self::State, config: &mut dyn Any, world: &mut World) {
        if let Some(a) = config.downcast_mut::<RenderSystemInitialState>() {
            *state = Some(a.queue_submission_info.clone());
            return;
        }
    }
    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &bevy::ecs::system::SystemMeta,
        world: bevy::ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy::ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        SubmissionInfo(state.as_ref().unwrap())
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
    pub binary_signals: Vec<BinarySemaphoreOp>,
    pub binary_waits: Vec<BinarySemaphoreOp>,
    device: Device,

    /// Map from frame index to retained objects
    pub retained_objects: BTreeMap<u64, Vec<Box<dyn Send + Sync>>>,
    pub fence_to_wait: BTreeMap<u64, vk::Fence>,
    pub frame_count_state: ComponentId,
}
impl Drop for QueueSystemState {
    fn drop(&mut self) {
        // On destuction, wait for everything to finish execution.
        unsafe {
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
    pub binary_signals: Vec<BinarySemaphoreOp>,
    pub binary_waits: Vec<BinarySemaphoreOp>,
}

pub struct RenderSystemInitialState {
    pub queue: QueueRef,
    pub queue_submission_info: Arc<Mutex<QueueSubmissionInfo>>,
    pub prev_stage_queue_submission_info: SmallVec<[Option<Arc<Mutex<QueueSubmissionInfo>>>; 4]>,
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
        let frame_count_state: ComponentId = Res::<FrameCount>::init_state(world, _system_meta);
        QueueSystemState {
            device: world.resource::<Device>().clone(),
            queue: QueueRef::default(),
            binary_signals: Vec::new(),
            binary_waits: Vec::new(),
            retained_objects: BTreeMap::new(),
            fence_to_wait: BTreeMap::new(),
            frame_count_state,
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
            state.binary_signals = std::mem::take(&mut initial_state.binary_signals);
            state.binary_waits = std::mem::take(&mut initial_state.binary_waits);
            return;
        }
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &bevy::ecs::system::SystemMeta,
        world: bevy::ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy::ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        let current_frame_index = Res::<FrameCount>::get_param(
            &mut state.frame_count_state,
            system_meta,
            world,
            change_tick,
        )
        .into_inner()
        .0;

        let num_frame_in_flight = 3;

        if current_frame_index >= num_frame_in_flight {
            let wait_value = current_frame_index - num_frame_in_flight;
            if let Some(&fence) = state.fence_to_wait.get(&wait_value) {
                unsafe {
                    state.device.wait_for_fences(&[fence], true, !0).unwrap();
                    // This fence will later be cleaned up by a call to fence_to_wait
                }
            }

            let objs = state.retained_objects.remove(&wait_value).unwrap();
            drop(objs);
        }

        let retained_objects: &mut Vec<Box<dyn Send + Sync>> = state
            .retained_objects
            .entry(current_frame_index)
            .or_default();

        QueueContext {
            device: &state.device,
            queue: state.queue,
            frame_index: current_frame_index,
            binary_signals: &state.binary_signals,
            binary_waits: &state.binary_waits,
            retained_objects,
            fences: &mut state.fence_to_wait,
        }
    }
}

pub(crate) fn flush_system_graph<const Q: char>(
    mut default_cmd_pool: ResMut<PerFrame<DefaultCommandPool<Q>>>,
    frame_index: Res<FrameCount>,
    submission_info: SubmissionInfo,
    device: Res<Device>,
    queues_router: Res<QueuesRouter>,
) where
    (): IsQueueCap<Q>,
{
    let default_cmd_pool = default_cmd_pool.on_frame_index(
        frame_index.0,
        submission_info.0,
        &device,
        (&device, &queues_router),
    );

    let mut submission_info = submission_info.0.lock().unwrap();
    if default_cmd_pool.buffer != vk::CommandBuffer::null() {
        submission_info.cmd_bufs.push(default_cmd_pool.buffer);
        submission_info.last_buf_open = true;
        default_cmd_pool.buffer = vk::CommandBuffer::null();
    } else {
        submission_info.last_buf_open = false;
    }
    // TODO: take care of unused remaining timeline semaphores
}
// So, what happens if multiple systems get assigned to the same queue?
// flush_system_graph will only run once for that particular queue.
// If they were assigned to different queues,
// flush_system_graph will run multiple times, once for each queue.
pub(crate) fn submit_system_graph<const Q: char>(
    mut default_cmd_pool: ResMut<PerFrame<DefaultCommandPool<Q>>>,
    submission_info: SubmissionInfo,
    queue_ctx: QueueContext<Q>,
    binary_semaphore_tracker: Res<RenderSystemsBinarySemaphoreTracker>,
    device: Res<Device>,
    frame_index: Res<FrameCount>,
    queues_router: Res<QueuesRouter>,
) where
    (): IsQueueCap<Q>,
{
    let default_cmd_pool = default_cmd_pool.on_frame_index(
        frame_index.0,
        submission_info.0,
        &device,
        (&device, &queues_router),
    );

    let mut submission_info = submission_info.0.lock().unwrap();
    // Record the trailing pipeline barrier
    if !submission_info.cmd_bufs.is_empty() || !submission_info.cmd_bufs.is_empty() {
        let cmd_buf = if submission_info.last_buf_open {
            submission_info.cmd_bufs.last().unwrap().clone()
        } else {
            let buf = default_cmd_pool.pool.allocate();
            unsafe {
                device
                    .begin_command_buffer(
                        buf,
                        &vk::CommandBufferBeginInfo {
                            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                            ..Default::default()
                        },
                    )
                    .unwrap();
            }
            submission_info.cmd_bufs.push(buf);
            buf
        };
        let buffer_barriers = std::mem::take(&mut submission_info.trailing_buffer_barriers);
        let image_barriers = std::mem::take(&mut submission_info.trailing_image_barriers);
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd_buf,
                &vk::DependencyInfoKHR {
                    p_buffer_memory_barriers: buffer_barriers.as_ptr(),
                    buffer_memory_barrier_count: buffer_barriers.len() as u32,
                    p_image_memory_barriers: image_barriers.as_ptr(),
                    image_memory_barrier_count: image_barriers.len() as u32,
                    ..Default::default()
                },
            );
            drop((buffer_barriers, image_barriers));
            device.end_command_buffer(cmd_buf).unwrap();
        }
    } else if submission_info.last_buf_open {
        let buf = submission_info.cmd_bufs.last().unwrap();
        unsafe {
            device.end_command_buffer(*buf).unwrap();
        }
    }
    // Take all the command buffers recorded so far, plus the default one.
    let command_buffer: Vec<_> = submission_info
        .cmd_bufs
        .iter()
        .cloned()
        .map(|command_buffer| vk::CommandBufferSubmitInfo {
            command_buffer,
            ..Default::default()
        })
        .collect();
    submission_info.cmd_bufs.clear();
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
            submission_info
                .signal_semaphores
                .iter()
                .map(|(stage, semaphore, value)| vk::SemaphoreSubmitInfo {
                    semaphore: semaphore.raw(),
                    value: *value,
                    stage_mask: *stage,
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
            submission_info
                .wait_semaphores
                .iter()
                .map(|(stage, semaphore, value)| vk::SemaphoreSubmitInfo {
                    semaphore: semaphore.raw(),
                    value: *value,
                    stage_mask: *stage,
                    ..Default::default()
                }),
        )
        .collect::<Vec<_>>();
    submission_info.signal_semaphores.clear();
    submission_info.wait_semaphores.clear();
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

    queue_submission_info: Option<Arc<Mutex<QueueSubmissionInfo>>>,
    prev_stage_submission_info: SmallVec<[Option<Arc<Mutex<QueueSubmissionInfo>>>; 4]>,
    device: Option<Device>,
    queue_family_index: u32,
}
impl<const Q: char> InsertPipelineBarrier<Q> {
    pub(crate) fn new() -> Self {
        InsertPipelineBarrier {
            barrier_producers: Vec::new(),
            system_meta: SystemMeta::new::<Self>(),
            render_command_state: None,
            record_to_next: false,
            queue_submission_info: None,
            device: None,
            queue_family_index: 0,
            prev_stage_submission_info: SmallVec::new(),
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
        let change_tick = world.increment_change_tick();

        let mut image_barriers = SmallVec::new();
        let mut buffer_barriers = SmallVec::new();
        let mut global_barriers = vk::MemoryBarrier2::default();
        let mut dependency_flags = vk::DependencyFlags::empty();
        let mut prev_stage_barriers = SmallVec::new();
        for i in self.barrier_producers.iter_mut() {
            let mut dropped = false;
            let barrier = Barriers {
                device: self.device.as_ref().unwrap(),
                image_barriers: &mut image_barriers,
                buffer_barriers: &mut buffer_barriers,
                global_barriers: &mut global_barriers,
                dependency_flags: &mut dependency_flags,
                prev_barriers: &mut prev_stage_barriers,
                submission_info: self
                    .queue_submission_info
                    .as_ref()
                    .map(|a| a.as_ref())
                    .unwrap(),
                dropped: &mut dropped,
                queue_family: (
                    match Q {
                        'g' => QueueType::Graphics,
                        'c' => QueueType::Compute,
                        't' => QueueType::Transfer,
                        _ => panic!(),
                    },
                    self.queue_family_index,
                ),
            };
            i.run_unsafe(barrier, world);
            assert!(dropped);
        }

        if global_barriers.dst_access_mask != vk::AccessFlags2::empty()
            || global_barriers.src_access_mask != vk::AccessFlags2::empty()
            || global_barriers.dst_stage_mask != vk::PipelineStageFlags2KHR::empty()
            || global_barriers.src_stage_mask != vk::PipelineStageFlags2KHR::empty()
            || !image_barriers.is_empty()
            || !buffer_barriers.is_empty()
        {
            let render_commands = RenderCommands::<Q>::get_param(
                self.render_command_state.as_mut().unwrap(),
                &self.system_meta,
                world,
                change_tick,
            );
            let cmd_buf = render_commands.default_cmd_pool.current_buffer();
            let barriers = ImmediateTransitions {
                cmd_buf,
                device: render_commands.device(),
                global_barriers,
                image_barriers,
                buffer_barriers,
                dependency_flags,
                queue_family: (
                    match Q {
                        'g' => QueueType::Graphics,
                        'c' => QueueType::Compute,
                        't' => QueueType::Transfer,
                        _ => panic!(),
                    },
                    render_commands.default_cmd_pool.pool.queue_family_index(),
                ),
            };
            drop(barriers);
        }

        for i in prev_stage_barriers.into_iter() {
            let prev_queue_type = i.prev_queue_type();
            let Some(ref mut prev_submission_info) =
                self.prev_stage_submission_info[prev_queue_type as usize]
            else {
                panic!("Queue family ownership transfers not allowed across frames");
            };
            let mut prev_submission_info = prev_submission_info.lock().unwrap();
            match i {
                BarriersPrevStage::Buffer { barrier, .. } => {
                    prev_submission_info.trailing_buffer_barriers.push(barrier)
                }
                BarriersPrevStage::Image { barrier, .. } => {
                    prev_submission_info.trailing_image_barriers.push(barrier)
                }
            }
        }

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
        self.device = Some(world.resource::<Device>().clone());
        self.queue_family_index = world
            .resource::<QueuesRouter>()
            .queue_family_of_type(match Q {
                'g' => QueueType::Graphics,
                'c' => QueueType::Compute,
                't' => QueueType::Transfer,
                _ => panic!(),
            });
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
        } else if let Some(state) = config.downcast_mut::<RenderSystemInitialState>() {
            self.queue_submission_info = Some(state.queue_submission_info.clone());
            self.prev_stage_submission_info = state.prev_stage_queue_submission_info.clone();
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
