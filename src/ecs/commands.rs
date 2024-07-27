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
    impl IsQueueCap<'u'> for () {}

    pub trait IsGraphicsQueueCap<const Q: QueueCap>: IsQueueCap<Q> {}
    impl IsGraphicsQueueCap<'g'> for () {}

    pub trait IsComputeQueueCap<const Q: QueueCap>: IsQueueCap<Q> {}
    impl IsComputeQueueCap<'c'> for () {}
    impl IsComputeQueueCap<'u'> for () {}
}

use std::{
    any::Any,
    borrow::Cow,
    ffi::CString,
    sync::{Arc, Mutex},
};

use ash::vk;
use bevy::{
    core::FrameCount,
    ecs::{
        archetype::ArchetypeComponentId,
        component::ComponentId,
        system::{Res, Resource, System, SystemMeta, SystemParam},
        world::World,
    },
};
use itertools::Itertools;
use queue_cap::*;
use smallvec::SmallVec;

use crate::{
    command_pool::ManagedCommandPool,
    commands::{CommandRecorder, ImmediateTransitions, SemaphoreSignalCommands},
    debug::DebugCommands,
    ecs::Barriers,
    semaphore::TimelineSemaphore,
    Device, HasDevice, QueueRef, Queues,
};

use super::{
    BarrierProducerCell, BarriersPrevStage, BoxedBarrierProducer, PerFrame, PerFrameReset,
    RenderSystemConfig, ResInstanceMut,
};
use crate::Access;

#[derive(Resource)]
pub(crate) struct DefaultCommandPool {
    pool: ManagedCommandPool,
    buffer: vk::CommandBuffer,
}
impl DefaultCommandPool {
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

    pub fn new(device: Device, queue_family_index: u32) -> Self {
        let pool = ManagedCommandPool::new(device, queue_family_index).unwrap();
        Self {
            pool,
            buffer: vk::CommandBuffer::null(),
        }
    }
}
impl PerFrameReset for DefaultCommandPool {
    fn reset(&mut self) {
        self.pool.reset();
        self.buffer = vk::CommandBuffer::null();
    }
}

#[derive(Default)]
pub struct QueueSubmissionInfo {
    pub(crate) cmd_bufs: SmallVec<[vk::CommandBuffer; 4]>,
    pub(crate) last_buf_open: bool,
    pub(crate) trailing_buffer_barriers: Vec<vk::BufferMemoryBarrier2<'static>>,
    pub(crate) trailing_image_barriers: Vec<vk::ImageMemoryBarrier2<'static>>,

    pub(crate) signal_semaphore: Option<Arc<TimelineSemaphore>>,
    pub(crate) signal_semaphore_value: u64,
    pub(crate) signal_binary_semaphore: vk::Semaphore,
    pub(crate) wait_semaphores:
        SmallVec<[(vk::PipelineStageFlags2, Arc<TimelineSemaphore>, u64); 4]>,
    pub(crate) wait_binary_semaphore: vk::Semaphore,
}
unsafe impl Send for QueueSubmissionInfo {}
unsafe impl Sync for QueueSubmissionInfo {}
impl QueueSubmissionInfo {
    /// Return TRUE if the current queue is going to wait on this semaphore, or FALSE otherwise.
    pub fn wait_semaphore(
        &mut self,
        semaphore: Cow<Arc<TimelineSemaphore>>,
        value: u64,
        stage: vk::PipelineStageFlags2,
    ) -> bool {
        if let Some(signal_semaphore) = self.signal_semaphore.as_ref() {
            if Arc::ptr_eq(signal_semaphore, semaphore.as_ref()) {
                if self.signal_semaphore_value <= value {
                    return false;
                }
            }
        }
        for (current_stage, current_semaphore, current_value) in self.wait_semaphores.iter_mut() {
            if Arc::ptr_eq(current_semaphore, semaphore.as_ref()) {
                *current_value = (*current_value).max(value);
                *current_stage |= stage;
                return true;
            }
        }
        self.wait_semaphores
            .push((stage, semaphore.into_owned(), value));
        true
    }

    pub fn signal_semaphore(
        &mut self,
        _stage: vk::PipelineStageFlags2,
    ) -> (Arc<TimelineSemaphore>, u64) {
        if let Some(signal_semaphore) = self.signal_semaphore.as_ref() {
            return (signal_semaphore.clone(), self.signal_semaphore_value);
        }
        panic!("This queue operation is unable to signal a timeline semaphore");
    }
    pub fn signal_binary_semaphore(
        &mut self,
        semaphore: vk::Semaphore,
        _stage: vk::PipelineStageFlags2,
    ) {
        if self.signal_binary_semaphore != vk::Semaphore::null() {
            panic!()
        }
        self.signal_binary_semaphore = semaphore;
    }
    pub fn wait_binary_semaphore(
        &mut self,
        semaphore: vk::Semaphore,
        _stage: vk::PipelineStageFlags2,
    ) {
        if self.wait_binary_semaphore != vk::Semaphore::null() {
            panic!()
        }
        self.wait_binary_semaphore = semaphore;
    }
}

pub struct RenderCommands<'w, 's, const Q: char>
where
    (): IsQueueCap<Q>,
{
    queue: QueueRef,
    default_cmd_pool: &'w mut DefaultCommandPool,
    pub(crate) frame_index: u64,
    pub(crate) submission_info: &'s Mutex<QueueSubmissionInfo>,
    prev_stage_submission_info: &'s [Option<Arc<Mutex<QueueSubmissionInfo>>>],
}

impl<'w, 's, const Q: char> RenderCommands<'w, 's, Q>
where
    (): IsQueueCap<Q>,
{
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
    fn current_queue(&self) -> QueueRef {
        self.queue
    }
    fn cmd_buf(&mut self) -> vk::CommandBuffer {
        self.default_cmd_pool.current_buffer()
    }
    fn semaphore_signal(&mut self) -> &mut impl SemaphoreSignalCommands {
        self
    }
}
impl<'w, 's, const Q: char> SemaphoreSignalCommands for RenderCommands<'w, 's, Q>
where
    (): IsQueueCap<Q>,
{
    fn wait_semaphore(
        &mut self,
        semaphore: Cow<Arc<TimelineSemaphore>>,
        value: u64,
        stage: vk::PipelineStageFlags2,
    ) -> bool {
        self.submission_info
            .lock()
            .unwrap()
            .wait_semaphore(semaphore, value, stage)
    }

    fn wait_binary_semaphore(&mut self, semaphore: vk::Semaphore, stage: vk::PipelineStageFlags2) {
        self.submission_info
            .lock()
            .unwrap()
            .wait_binary_semaphore(semaphore, stage)
    }

    fn signal_semaphore(
        &mut self,
        stage: vk::PipelineStageFlags2,
    ) -> (Arc<TimelineSemaphore>, u64) {
        self.submission_info.lock().unwrap().signal_semaphore(stage)
    }

    fn signal_binary_semaphore_prev_stage(
        &mut self,
        _semaphore: vk::Semaphore,
        _stage: vk::PipelineStageFlags2,
        _prev_queue: QueueRef,
    ) {
        todo!()
    }

    fn wait_binary_semaphore_prev_stage(
        &mut self,
        _semaphore: vk::Semaphore,
        _stage: vk::PipelineStageFlags2,
        _prev_queue: QueueRef,
    ) {
        todo!()
    }
}

pub struct RenderCommandState {
    default_cmd_pool_state: Option<ComponentId>,
    submission_info: Option<Arc<Mutex<QueueSubmissionInfo>>>,
    prev_stage_submission_info: SmallVec<[Option<Arc<Mutex<QueueSubmissionInfo>>>; 4]>,
    frame_index_state: ComponentId,
    queue: QueueRef,
}

unsafe impl<'w, 's, const Q: char> SystemParam for RenderCommands<'w, 's, Q>
where
    (): IsQueueCap<Q>,
{
    type State = RenderCommandState;

    type Item<'world, 'state> = RenderCommands<'world, 'state, Q>;

    fn init_state(
        world: &mut World,
        system_meta: &mut bevy::ecs::system::SystemMeta,
    ) -> Self::State {
        let default_cmd_pool_state =
            ResInstanceMut::<PerFrame<DefaultCommandPool>>::init_state(world, system_meta);
        let frame_index_state = Res::<FrameCount>::init_state(world, system_meta);
        RenderCommandState {
            default_cmd_pool_state,
            submission_info: None,
            prev_stage_submission_info: SmallVec::new(),
            frame_index_state,
            queue: QueueRef::default(),
        }
    }
    fn default_configs(config: &mut bevy::utils::ConfigMap) {
        let config = config.entry::<RenderSystemConfig>().or_default();
        match Q {
            'g' => {
                config.required_queue_flags = vk::QueueFlags::GRAPHICS;
            }
            'c' => {
                config.required_queue_flags = vk::QueueFlags::COMPUTE;
            }
            't' => {
                config.required_queue_flags = vk::QueueFlags::TRANSFER;
            }
            'u' => {
                config.required_queue_flags = vk::QueueFlags::COMPUTE;
                config.preferred_queue_flags = vk::QueueFlags::GRAPHICS;
            }
            _ => unreachable!(),
        };
    }
    fn configurate(
        state: &mut Self::State,
        config: &mut dyn Any,
        meta: &mut SystemMeta,
        world: &mut World,
    ) {
        if let Some(a) = config.downcast_mut::<RenderSystemInitialState>() {
            state.submission_info = Some(a.queue_submission_info.clone());
            state.prev_stage_submission_info = a.prev_stage_queue_submission_info.clone();
            state.queue = a.queue;
            return;
        }
        ResInstanceMut::<PerFrame<DefaultCommandPool>>::configurate(
            &mut state.default_cmd_pool_state,
            config,
            meta,
            world,
        );
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
        let default_cmd_pool: ResInstanceMut<'world, _> =
            ResInstanceMut::<PerFrame<DefaultCommandPool>>::get_param(
                &mut state.default_cmd_pool_state,
                system_meta,
                world,
                change_tick,
            );
        let default_cmd_pool: &'world mut _ = default_cmd_pool.into_inner();
        let default_cmd_pool = default_cmd_pool
            .on_frame_index(current_frame_index, state.submission_info.as_ref().unwrap());

        let mut this = RenderCommands {
            queue: state.queue,
            frame_index: current_frame_index,
            default_cmd_pool,
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

pub struct SubmissionInfo<'s> {
    info: &'s Mutex<QueueSubmissionInfo>,
    queue: QueueRef,
}

unsafe impl<'s> SystemParam for SubmissionInfo<'s> {
    type State = Option<(Arc<Mutex<QueueSubmissionInfo>>, QueueRef)>;

    type Item<'world, 'state> = SubmissionInfo<'state>;

    fn init_state(
        _world: &mut World,
        _system_meta: &mut bevy::ecs::system::SystemMeta,
    ) -> Self::State {
        None
    }
    fn configurate(
        state: &mut Self::State,
        config: &mut dyn Any,
        _meta: &mut SystemMeta,
        _world: &mut World,
    ) {
        if let Some(a) = config.downcast_mut::<RenderSystemInitialState>() {
            *state = Some((a.queue_submission_info.clone(), a.queue));
            return;
        }
    }
    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        _system_meta: &bevy::ecs::system::SystemMeta,
        _world: bevy::ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        _change_tick: bevy::ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        let (info, queue) = state.as_ref().unwrap();
        SubmissionInfo {
            info,
            queue: *queue,
        }
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

pub struct RenderSystemInitialState {
    pub queue: QueueRef,
    pub queue_submission_info: Arc<Mutex<QueueSubmissionInfo>>,
    pub prev_stage_queue_submission_info: SmallVec<[Option<Arc<Mutex<QueueSubmissionInfo>>>; 4]>,
}

pub(crate) fn flush_system_graph(
    mut default_cmd_pool: ResInstanceMut<PerFrame<DefaultCommandPool>>,
    frame_index: Res<FrameCount>,
    submission_info: SubmissionInfo,
) {
    let default_cmd_pool = default_cmd_pool.on_frame_index(frame_index.0, submission_info.info);

    let mut submission_info = submission_info.info.lock().unwrap();
    if default_cmd_pool.buffer != vk::CommandBuffer::null() {
        submission_info.cmd_bufs.push(default_cmd_pool.buffer);
        submission_info.last_buf_open = true;
        default_cmd_pool.buffer = vk::CommandBuffer::null();
    } else {
        submission_info.last_buf_open = false;
    }
    // TODO: take care of unused remaining timeline semaphores
}

pub(crate) fn submit_system_graph(
    mut default_cmd_pool: ResInstanceMut<PerFrame<DefaultCommandPool>>,
    submission_info: SubmissionInfo,
    queues: Res<Queues>,
    device: Res<Device>,
    frame_index: Res<FrameCount>,
) {
    let default_cmd_pool = default_cmd_pool.on_frame_index(frame_index.0, submission_info.info);
    let queue = submission_info.queue;
    let mut submission_info = submission_info.info.lock().unwrap();
    // Record the trailing pipeline barrier
    if !submission_info.trailing_buffer_barriers.is_empty()
        || !submission_info.trailing_image_barriers.is_empty()
    {
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
    let mut semaphore_signals = submission_info
        .signal_semaphore
        .iter()
        .map(|semaphore| vk::SemaphoreSubmitInfo {
            semaphore: semaphore.raw(),
            value: submission_info.signal_semaphore_value,
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            ..Default::default()
        })
        .collect::<Vec<_>>();
    let mut semaphore_waits = submission_info
        .wait_semaphores
        .iter()
        .map(|(stage, semaphore, value)| vk::SemaphoreSubmitInfo {
            semaphore: semaphore.raw(),
            value: *value,
            stage_mask: *stage,
            ..Default::default()
        })
        .collect::<Vec<_>>();
    if submission_info.signal_binary_semaphore != vk::Semaphore::null() {
        semaphore_signals.push(vk::SemaphoreSubmitInfo {
            semaphore: submission_info.signal_binary_semaphore,
            value: 0,
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            ..Default::default()
        });
    }
    if submission_info.wait_binary_semaphore != vk::Semaphore::null() {
        semaphore_waits.push(vk::SemaphoreSubmitInfo {
            semaphore: submission_info.wait_binary_semaphore,
            value: 0,
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            ..Default::default()
        });
    }

    unsafe {
        let queue = queues.get(queue);
        // TODO: Ensure submission safety on vk::Queue by adding
        // execution dependencies between all instances of submit_system_graph and
        // vkQueuePresentKHR
        device
            .queue_submit2(
                *queue,
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
        drop(queue);
    }

    submission_info.signal_semaphore_value += 1;
    submission_info.wait_semaphores.clear();
    submission_info.signal_binary_semaphore = vk::Semaphore::null();
    submission_info.wait_binary_semaphore = vk::Semaphore::null();
}

pub(crate) struct InsertPipelineBarrier {
    barrier_producers: Vec<BoxedBarrierProducer>,
    system_meta: SystemMeta,
    render_command_state: Option<RenderCommandState>,
    /// If true, will record to `barrier_buffer` instead of `current_buffer`.
    record_to_next: bool,

    queue_submission_info: Option<Arc<Mutex<QueueSubmissionInfo>>>,
    prev_stage_submission_info: SmallVec<[Option<Arc<Mutex<QueueSubmissionInfo>>>; 4]>,
    queue: QueueRef,
}
impl InsertPipelineBarrier {
    pub(crate) fn new() -> Self {
        InsertPipelineBarrier {
            barrier_producers: Vec::new(),
            system_meta: SystemMeta::new::<Self>(),
            render_command_state: None,
            record_to_next: false,
            queue_submission_info: None,
            prev_stage_submission_info: SmallVec::new(),
            queue: QueueRef::default(),
        }
    }
}

pub(crate) struct InsertPipelineBarrierConfig {
    pub(crate) producers: Vec<BoxedBarrierProducer>,
    pub(crate) record_to_next: bool,
}
impl System for InsertPipelineBarrier {
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
        std::any::TypeId::of::<InsertPipelineBarrier>()
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
        let device = world.get_resource::<Device>().unwrap();
        for i in self.barrier_producers.iter_mut() {
            let mut dropped = false;
            let barrier = Barriers {
                device,
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
                queue_family: self.queue,
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
            let mut render_commands = RenderCommands::<'t'>::get_param(
                self.render_command_state.as_mut().unwrap(),
                &self.system_meta,
                world,
                change_tick,
            );
            let cmd_buf = render_commands.default_cmd_pool.current_buffer();
            let barriers = ImmediateTransitions {
                cmd_buf,
                semaphore_signals: &mut render_commands,
                global_barriers,
                image_barriers,
                buffer_barriers,
                dependency_flags,
                queue: self.queue,
            };
            drop(barriers);
        }

        for i in prev_stage_barriers.into_iter() {
            let prev_queue_type = i.prev_queue_type();
            let Some(ref mut prev_submission_info) =
                self.prev_stage_submission_info[prev_queue_type.index as usize]
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
                BarriersPrevStage::SignalBinarySemaphore {
                    stage, semaphore, ..
                } => prev_submission_info.signal_binary_semaphore(semaphore, stage),
                BarriersPrevStage::WaitBinarySemaphore {
                    stage, semaphore, ..
                } => prev_submission_info.wait_binary_semaphore(semaphore, stage),
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
        self.render_command_state = Some(RenderCommands::<'t'>::init_state(
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
        } else if let Some(state) = config.downcast_mut::<RenderSystemInitialState>() {
            self.queue_submission_info = Some(state.queue_submission_info.clone());
            self.prev_stage_submission_info = state.prev_stage_queue_submission_info.clone();
            self.queue = state.queue;
            RenderCommands::<'t'>::configurate(
                self.render_command_state.as_mut().unwrap(),
                config,
                &mut self.system_meta,
                world,
            );
        } else {
            RenderCommands::<'t'>::configurate(
                self.render_command_state.as_mut().unwrap(),
                config,
                &mut self.system_meta,
                world,
            );
        }
    }

    fn queue_deferred(&mut self, _world: bevy::ecs::world::DeferredWorld) {}
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

    fn init_state(_world: &mut World, _system_meta: &mut SystemMeta) -> Self::State {
        None
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        _system_meta: &SystemMeta,
        _world: bevy::ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        _change_tick: bevy::ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        let state = state.as_mut().unwrap();
        let cell = state.0.get().as_mut().unwrap();
        Self(cell.take().unwrap())
    }

    fn configurate(
        state: &mut Self::State,
        config: &mut dyn Any,
        _meta: &mut SystemMeta,
        _world: &mut World,
    ) {
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
