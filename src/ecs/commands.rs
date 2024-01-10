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

use ash::vk;
use bevy_ecs::{system::{SystemParam, Res}, world::World, component::ComponentId};
use queue_cap::*;

use crate::queue::QueueType;

use super::{QueueAssignment, RenderSystemConfig, RenderResRegistry};

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
pub struct SemaphoreOp {
    pub semaphore: vk::Semaphore,
    pub stage: vk::PipelineStageFlags2,
    pub value: u64,
}


#[derive(Debug)]
pub struct QueueSystemState {
    queue: vk::Queue,
    registry_component_id: ComponentId
}
#[derive(Debug)]
pub struct QueueSystemInitialState {
    pub queue: vk::Queue,
}

pub struct QueueContext<'a, const Q: char>
where
    (): IsQueueCap<Q>, {
    pub queue: vk::Queue,
    pub semaphore_waits: &'a [SemaphoreOp],
    pub semaphore_signals: &'a [SemaphoreOp],
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
            queue: vk::Queue::null(),
            registry_component_id: component_id,
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
    fn set_configs(state: &mut Self::State, config: &mut bevy_utils::ConfigMap) {
        let initial_state: QueueSystemInitialState = config.take().unwrap();
        state.queue = initial_state.queue;
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

        println!("system {} has access {:?}", system_meta.name(), system_meta.archetype_component_access());
        println!("list of all rendering archetype components: {:?}", registry.archetype_component_access());
        QueueContext {
            queue: state.queue,
            semaphore_waits: &[],
            semaphore_signals: &[],
        }
    }
}