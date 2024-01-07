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

use bevy_ecs::{system::SystemParam, world::World};
use queue_cap::*;

use crate::queue::QueueType;

use super::{QueueAssignment, RenderSystemConfig};

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
// When can you record two command buffers concurrently?
// - When they do not have any CPU side dependencies - build time enforcement
// - When they do not have CPU side resource conflicts - runtime enforcement
// systems with CPU side dependencies will always be recorded into the same command buffer.

// For each queue node, add a submission node.
// it's told the semaphores to wait and signal.
// semaphores are entirely managed by the pass.
