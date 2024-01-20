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

use std::any::Any;

use ash::vk;
use bevy_ecs::{system::{SystemParam, Res}, world::World, component::ComponentId};
use queue_cap::*;

use crate::{queue::QueueType, BinarySemaphore, QueueRef};

use super::{QueueAssignment, RenderSystemConfig, Access, RenderResRegistry};

pub struct RenderCommands<const Q: char>
where
    (): IsQueueCap<Q>, {}

unsafe impl<const Q: char> SystemParam for RenderCommands<Q>
where
    (): IsQueueCap<Q>,
{
    type State = ();

    type Item<'world, 'state> = RenderCommands<Q>;

    fn init_state(
        _world: &mut World,
        system_meta: &mut bevy_ecs::system::SystemMeta,
    ) -> Self::State {
    }

    fn default_configs(config: &mut bevy_utils::ConfigMap) {
        let flags = match Q {
            'g' => QueueType::Graphics,
            'c' => QueueType::Compute,
            't' => QueueType::Transfer,
            _ => unreachable!(),
        };
        let config = config.entry::<RenderSystemConfig>().or_default();
        config.queue = QueueAssignment::MinOverhead(flags);
    }

    unsafe fn get_param<'world, 'state>(
        _state: &'state mut Self::State,
        _system_meta: &bevy_ecs::system::SystemMeta,
        _world: bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        _change_tick: bevy_ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        RenderCommands {}
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


#[derive(Debug)]
pub struct QueueSystemState {
    pub queue: QueueRef,
    pub frame_index: u64,
    pub binary_signals: Vec<SemaphoreOp>,
    pub binary_waits: Vec<BinarySemaphoreWaitOp>,
    pub timeline_signals: Vec<SemaphoreOp>,
    pub timeline_waits: Vec<SemaphoreOp>,
    registry_component_id: ComponentId
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
    (): IsQueueCap<Q>, {
    pub queue: QueueRef,
    pub binary_signals: &'a [SemaphoreOp],
    pub binary_waits: &'a [BinarySemaphoreWaitOp],
    pub timeline_signals: &'a [SemaphoreOp],
    pub timeline_waits: &'a [SemaphoreOp],
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
        let component_id = Res::<RenderResRegistry>::init_state(world, system_meta);
        system_meta.set_has_deferred();
        QueueSystemState {
            registry_component_id: component_id,
            queue: QueueRef::default(),
            binary_signals: Vec::new(),
            binary_waits: Vec::new(),
            timeline_signals: Vec::new(),
            timeline_waits: Vec::new(),
            frame_index: 0,
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
        config.queue = QueueAssignment::MinOverhead(flags);
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

    fn apply(state: &mut Self::State, system_meta: &bevy_ecs::system::SystemMeta, world: &mut World) {
        let mut registry = world.resource_mut::<RenderResRegistry>();
        Res::<RenderResRegistry>::apply(&mut state.registry_component_id, system_meta, world);
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &bevy_ecs::system::SystemMeta,
        world: bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy_ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        let registry = Res::<RenderResRegistry>::get_param(&mut state.registry_component_id, system_meta, world, change_tick);

        QueueContext {
            queue: state.queue,
            binary_signals: &state.binary_signals,
            binary_waits: &state.binary_waits,
            timeline_signals: &state.timeline_signals,
            timeline_waits: &state.timeline_waits,
        }
    }
}
