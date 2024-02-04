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

    pub trait IsGraphicsQueueCap<const Q: QueueCap> {}
    impl IsGraphicsQueueCap<'g'> for () {}

    pub trait IsComputeQueueCap<const Q: QueueCap> {}
    impl IsComputeQueueCap<'c'> for () {}
}

use std::{
    any::Any, cell::RefCell, collections::BTreeMap, ops::{Deref, DerefMut}, sync::atomic::AtomicU64
};

use ash::vk::{self, Handle};
use bevy_ecs::{
    component::{ComponentDescriptor, ComponentId, ComponentInfo},
    system::{Res, ResMut, Resource, SystemParam},
    world::{FromWorld, Mut, World},
};
use queue_cap::*;

use crate::{
    command_pool::RecordingCommandBuffer, commands::CommandRecorder, queue::QueueType,
    Device, HasDevice, QueueRef, QueuesRouter,
};

use super::{Access, RenderResRegistry, RenderSystemConfig};

/// A wrapper to produce multiple [`RecordingCommandBuffer`] variants based on the queue type it supports.
#[derive(Resource)]
struct RecordingCommandBufferWrapper<const Q: char>(RecordingCommandBuffer);

pub struct RenderCommands<'w, const Q: char>
where
    (): IsQueueCap<Q>,
{
    recording_cmd_buf: ResMut<'w, RecordingCommandBufferWrapper<Q>>,
}

impl<'w, const Q: char> RenderCommands<'w, Q>
where
    (): IsQueueCap<Q>,
{
    pub fn record_commands(&mut self) -> CommandRecorder<Q> {
        let cmd_buf = self.recording_cmd_buf.0.record();
        CommandRecorder {
            device: self.recording_cmd_buf.0.device(),
            cmd_buf,
        }
    }
}

pub struct RenderCommandState {
    recording_cmd_buf_component_id: ComponentId,
}

unsafe impl<'a, const Q: char> SystemParam for RenderCommands<'a, Q>
where
    (): IsQueueCap<Q>,
{
    type State = RenderCommandState;

    type Item<'world, 'state> = RenderCommands<'world, Q>;

    fn init_state(
        world: &mut World,
        system_meta: &mut bevy_ecs::system::SystemMeta,
    ) -> Self::State {
        let recording_cmd_buf_component_id =
            ResMut::<RecordingCommandBufferWrapper<Q>>::init_state(world, system_meta);
        if world
            .get_resource_by_id(recording_cmd_buf_component_id)
            .is_none()
        {
            let device = world.resource::<Device>().clone();
            let router = world.resource::<QueuesRouter>();
            let queue_family = router.queue_family_of_type(match Q {
                'g' => QueueType::Graphics,
                'c' => QueueType::Compute,
                't' => QueueType::Transfer,
                _ => panic!(),
            });
            let pool = RecordingCommandBuffer::new(device, queue_family);
            world.insert_resource(RecordingCommandBufferWrapper::<Q>(pool));
        }
        RenderCommandState {
            recording_cmd_buf_component_id,
        }
    }
    fn default_configs(config: &mut bevy_utils::ConfigMap) {
        let flags = match Q {
            'g' => QueueType::Graphics,
            'c' => QueueType::Compute,
            't' => QueueType::Transfer,
            _ => unreachable!(),
        };
        let config = config.entry::<RenderSystemConfig>().or_default();
        config.queue = flags;
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &bevy_ecs::system::SystemMeta,
        world: bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy_ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        let recording_cmd_buf = ResMut::<RecordingCommandBufferWrapper<Q>>::get_param(
            &mut state.recording_cmd_buf_component_id,
            system_meta,
            world,
            change_tick,
        );
        RenderCommands { recording_cmd_buf }
    }
}

#[derive(Debug)]
pub struct BinarySemaphoreOp {
    pub index: u32,
    pub access: Access,
}
#[derive(Debug)]
pub struct SemaphoreOp {
    pub semaphore: vk::Semaphore,
    pub access: Access,
}

pub struct QueueSystemState {
    pub queue: QueueRef,
    pub frame_index: u64,
    pub binary_signals: Vec<BinarySemaphoreOp>,
    pub binary_waits: Vec<BinarySemaphoreOp>,
    pub timeline_signals: Vec<SemaphoreOp>,
    pub timeline_waits: Vec<SemaphoreOp>,
    device: Device,

    /// Map from frame index to retained objects
    pub retained_objects: BTreeMap<u64, Vec<Box<dyn Send + Sync>>>,
    pub fence_to_wait: BTreeMap<u64, vk::Fence>,
}
impl Drop for QueueSystemState {
    fn drop(&mut self) {
        // On destuction, wait for everything to finish execution.
        unsafe {
            if !self.timeline_waits.is_empty() {
                let timeline_semaphore_to_wait = self.timeline_waits[0].semaphore;
                self.device.wait_semaphores(&vk::SemaphoreWaitInfo {
                    semaphore_count: 1,
                    p_semaphores: &timeline_semaphore_to_wait,
                    p_values: &self.frame_index,
                    ..Default::default()
                }, !0).unwrap();
            }
            if !self.fence_to_wait.is_empty() {
                let fence_to_wait: Vec<vk::Fence> = self.fence_to_wait.values().cloned().collect();
                self.device.wait_for_fences(&fence_to_wait, true, !0).unwrap();
            }
        }
    }
}
#[derive(Debug)]
pub struct QueueSystemInitialState {
    pub queue: QueueRef,
    pub timeline_signals: Vec<SemaphoreOp>,
    pub timeline_waits: Vec<SemaphoreOp>,
    pub binary_signals: Vec<BinarySemaphoreOp>,
    pub binary_waits: Vec<BinarySemaphoreOp>
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
    pub timeline_signals: &'state [SemaphoreOp],
    pub timeline_waits: &'state [SemaphoreOp],
    pub(crate) retained_objects: &'state mut Vec<Box<dyn Send + Sync>>,
    fences: &'state mut BTreeMap<u64, vk::Fence>,
}
impl<const Q: char> QueueContext<'_, Q>
where
    (): IsQueueCap<Q>,
{
    pub fn retain<T: 'static + Drop + Send + Sync>(&mut self, obj: Box<T>) {
        self.retained_objects.push(obj);
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
        system_meta: &mut bevy_ecs::system::SystemMeta,
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

    fn default_configs(config: &mut bevy_utils::ConfigMap) {
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
    fn set_configs(state: &mut Self::State, config: &mut Option<Box<dyn Any>>) {
        let Some(c) = config else {
            return;
        };
        if c.is::<QueueSystemInitialState>() {
            let config = config.take().unwrap();
            let initial_state: Box<QueueSystemInitialState> = config.downcast().unwrap();
            state.queue = initial_state.queue;
            state.timeline_signals = initial_state.timeline_signals;
            state.timeline_waits = initial_state.timeline_waits;
            state.binary_signals = initial_state.binary_signals;
            state.binary_waits = initial_state.binary_waits;
            return;
        }
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &bevy_ecs::system::SystemMeta,
        world: bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy_ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        state.frame_index += 1;

        let num_frame_in_flight = 3;

        if state.frame_index > num_frame_in_flight {
            let wait_value = state.frame_index - num_frame_in_flight;
            if !state.timeline_signals.is_empty() {
                let signaled_semaphore = state.timeline_signals[0].semaphore;
                // Just waiting on one of those timeline semaphore should be fine, right?
                unsafe {
                    state.device
                        .wait_semaphores(
                            &vk::SemaphoreWaitInfo {
                                semaphore_count: 1,
                                p_semaphores: &signaled_semaphore,
                                p_values: &wait_value, // num of frame in flight
                                ..Default::default()
                            },
                            !0,
                        )
                        .unwrap();
                }
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
            binary_waits:&state.binary_waits,
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
    mut commands: RenderCommands<Q>,
    queue_ctx: QueueContext<Q>,
    binary_semaphore_tracker: Res<RenderSystemsBinarySemaphoreTracker>,
    device: Res<Device>,
) where
    (): IsQueueCap<Q>,
{
    let command_buffer = unsafe { commands.recording_cmd_buf.0.take() };
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
                    semaphore: op.semaphore,
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
                    semaphore: op.semaphore,
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
                    command_buffer_info_count: if command_buffer == vk::CommandBuffer::null() { 0 } else { 1 },
                    p_command_buffer_infos: &vk::CommandBufferSubmitInfoKHR {
                        command_buffer: command_buffer,
                        ..Default::default()
                    },
                    signal_semaphore_info_count: semaphore_signals.len() as u32,
                    p_signal_semaphore_infos: semaphore_signals.as_ptr(),
                    ..Default::default()
                }],
                vk::Fence::null(),
            )
            .unwrap();
    }
}



pub struct RecycledBinarySemaphore {
    semaphore: vk::Semaphore,
    sender: crossbeam_channel::Sender<vk::Semaphore>,
}
pub struct BinarySemaphoreToDestroy {
    device: Device,
    semaphore: vk::Semaphore,
}
impl Drop for BinarySemaphoreToDestroy {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.semaphore, None);
        }
    }
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
    semaphores: Vec<AtomicU64>
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
                let semaphore = unsafe { self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap() };
                semaphore
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => panic!()
        };
        self.semaphores[index as usize].compare_exchange(0, semaphore.as_raw(), std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed).expect("Double signal");
        semaphore
    }
    pub fn wait(&self, index: u32) -> Option<RecycledBinarySemaphore> {
        let semaphore = self.semaphores[index as usize].swap(0, std::sync::atomic::Ordering::Relaxed);
        let semaphore = vk::Semaphore::from_raw(semaphore);
        if semaphore == vk::Semaphore::null() {
            None
        } else {
            Some(RecycledBinarySemaphore { semaphore, sender: self.sender.clone() })
        }
    }
    pub fn destroy(&self, index: u32) -> Option<BinarySemaphoreToDestroy> {
        let semaphore = self.semaphores[index as usize].swap(0, std::sync::atomic::Ordering::Relaxed);
        let semaphore = vk::Semaphore::from_raw(semaphore);
        if semaphore == vk::Semaphore::null() {
            None
        } else {
            Some(BinarySemaphoreToDestroy {
                device: self.device.clone(),
                semaphore,
            })
        }
    }
}
