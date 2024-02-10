use bevy_ecs::system::{Local, SystemParam};

pub struct PerFrame<T> {
    item: T
}

unsafe impl<T> SystemParam for PerFrame<T> {
    type State =();

    type Item<'world, 'state>=();

    fn init_state(world: &mut bevy_ecs::world::World, system_meta: &mut bevy_ecs::system::SystemMeta) -> Self::State {
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &bevy_ecs::system::SystemMeta,
        world: bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy_ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        todo!()
    }
}

