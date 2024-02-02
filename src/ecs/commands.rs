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

use std::{any::Any, collections::BTreeMap, ops::DerefMut};

use ash::vk;
use bevy_ecs::{
    component::{ComponentDescriptor, ComponentId, ComponentInfo},
    system::{Res, ResMut, Resource, SystemParam},
    world::{FromWorld, Mut, World},
};
use queue_cap::*;

use crate::{
    command_pool::RecordingCommandBuffer, commands::CommandRecorder, queue::QueueType,
    BinarySemaphore, Device, HasDevice, QueueRef, QueuesRouter,
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
pub struct BinarySemaphoreWaitOp {
    pub semaphore: BinarySemaphore,
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
    pub binary_signals: Vec<SemaphoreOp>,
    pub binary_waits: Vec<BinarySemaphoreWaitOp>,
    pub timeline_signals: Vec<SemaphoreOp>,
    pub timeline_waits: Vec<SemaphoreOp>,
    device_component_id: ComponentId,

    /// Map from frame index to retained objects
    pub retained_objects: BTreeMap<u64, Vec<Box<dyn Send + Sync>>>,
}
#[derive(Debug)]
pub struct QueueSystemInitialState {
    pub queue: QueueRef,
    pub timeline_signals: Vec<SemaphoreOp>,
    pub timeline_waits: Vec<SemaphoreOp>,
}
#[derive(Debug)]
pub struct QueueSystemStateUpdate {
    pub frame_index: u64,
    pub binary_signals: Vec<SemaphoreOp>,
    pub binary_waits: Vec<BinarySemaphoreWaitOp>,
}

pub struct QueueContext<'a, const Q: char>
where
    (): IsQueueCap<Q>,
{
    pub queue: QueueRef,
    pub frame_index: u64,
    pub binary_signals: Vec<SemaphoreOp>,
    pub binary_waits: Vec<BinarySemaphoreWaitOp>,
    pub timeline_signals: &'a [SemaphoreOp],
    pub timeline_waits: &'a [SemaphoreOp],
    retained_objects: &'a mut Vec<Box<dyn Send + Sync>>,
}
impl<const Q: char> Drop for QueueContext<'_, Q>
where
    (): IsQueueCap<Q>,
{
    fn drop(&mut self) {
        for op in std::mem::take(&mut self.binary_waits) {
            self.retained_objects.push(Box::new(op));
        }
    }
}

impl<'a, const Q: char> QueueContext<'a, Q>
where
    (): IsQueueCap<Q>,
{
    pub fn retain<T: 'static + Drop + Send + Sync>(&mut self, obj: Box<T>) {
        self.retained_objects.push(obj);
    }
}

unsafe impl<'a, const Q: char> SystemParam for QueueContext<'a, Q>
where
    (): IsQueueCap<Q>,
{
    type State = QueueSystemState;

    type Item<'world, 'state> = QueueContext<'state, Q>;

    fn init_state(
        world: &mut World,
        system_meta: &mut bevy_ecs::system::SystemMeta,
    ) -> Self::State {
        let device_component_id = Res::<Device>::init_state(world, system_meta);
        QueueSystemState {
            device_component_id,
            queue: QueueRef::default(),
            binary_signals: Vec::new(),
            binary_waits: Vec::new(),
            timeline_signals: Vec::new(),
            timeline_waits: Vec::new(),
            frame_index: 0,
            retained_objects: BTreeMap::new(),
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
            return;
        }
        if c.is::<QueueSystemStateUpdate>() {
            let config = config.take().unwrap();
            let update: Box<QueueSystemStateUpdate> = config.downcast().unwrap();
            state.binary_signals = update.binary_signals;
            state.binary_waits = update.binary_waits;
            state.frame_index = update.frame_index;
            return;
        }
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &bevy_ecs::system::SystemMeta,
        world: bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy_ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        let num_frame_in_flight = 3;

        if !state.timeline_signals.is_empty() {
            let wait_value = if state.frame_index <= num_frame_in_flight { 0 } else { state.frame_index - num_frame_in_flight };
            let signaled_semaphore = state.timeline_signals[0].semaphore;
            let device = Res::<Device>::get_param(
                &mut state.device_component_id,
                system_meta,
                world,
                change_tick,
            );
            unsafe {
                device.wait_semaphores(&vk::SemaphoreWaitInfo {
                    semaphore_count: 1,
                    p_semaphores: &signaled_semaphore,
                    p_values: &wait_value, // num of frame in flight
                    ..Default::default()
                }, !0).unwrap();
            }
        }
        if state.frame_index > num_frame_in_flight {
            let objs = state.retained_objects.remove(&(&state.frame_index - &num_frame_in_flight));
            drop(objs.unwrap());
        }

        let retained_objects: &mut Vec<Box<dyn Send + Sync>> = state.retained_objects.entry(state.frame_index).or_default();

        QueueContext {
            queue: state.queue,
            frame_index: state.frame_index,
            binary_signals: std::mem::take(&mut state.binary_signals),
            binary_waits: std::mem::take(&mut state.binary_waits),
            timeline_signals: &state.timeline_signals,
            timeline_waits: &state.timeline_waits,
            retained_objects,
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
    device: Res<Device>,
) where
    (): IsQueueCap<Q>,
{
    let command_buffer = unsafe { commands.recording_cmd_buf.0.take() };
    let semaphore_signals = queue_ctx
        .binary_signals
        .iter()
        .map(|op| vk::SemaphoreSubmitInfo {
            semaphore: op.semaphore,
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
        .map(|op| {
            let info = vk::SemaphoreSubmitInfo {
                semaphore: op.semaphore.raw(),
                value: 0,
                stage_mask: op.access.stage,
                ..Default::default()
            };
            info
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
                    command_buffer_info_count: 1,
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
