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
        world: &mut World,
        system_meta: &mut bevy_ecs::system::SystemMeta,
    ) -> Self::State {
        let flags = match Q {
            'g' => QueueType::Graphics,
            'c' => QueueType::Compute,
            't' => QueueType::Transfer,
            _ => unreachable!(),
        };
        system_meta.default_config.insert(RenderSystemConfig {
            queue: QueueAssignment::MinOverhead(flags),
        });
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &bevy_ecs::system::SystemMeta,
        world: bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy_ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        RenderCommands {}
    }
}
