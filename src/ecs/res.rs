use std::collections::BTreeSet;

use bevy_ecs::{
    component::ComponentId,
    system::{Res, ResMut, Resource, SystemParam},
};

use super::{RenderResAccess, RenderSystemConfig};

#[derive(Resource, Default)]
pub struct RenderResRegistry {
    component_ids: BTreeSet<ComponentId>,
}

#[derive(Resource)]
#[repr(C)]
struct RenderResInner<T: ?Sized> {
    access: RenderResAccess,
    item: T,
}

#[repr(C)]
pub struct RenderRes<'w, T: Resource + ?Sized> {
    inner: Res<'w, RenderResInner<T>>,
}

#[repr(C)]
pub struct RenderResMut<'w, T: Resource + ?Sized> {
    inner: ResMut<'w, RenderResInner<T>>,
}

unsafe impl<'a, T: Resource> SystemParam for RenderRes<'a, T> {
    type State = ComponentId;

    type Item<'world, 'state> = RenderRes<'world, T>;

    fn init_state(
        world: &mut bevy_ecs::world::World,
        system_meta: &mut bevy_ecs::system::SystemMeta,
    ) -> Self::State {
        if !system_meta.default_config.has::<RenderSystemConfig>() {
            panic!("RenderRes<{}> can only be used in a render system, but {} is not. RenderCommands must be the first parameter of a render system.", std::any::type_name::<T>(), system_meta.name());
        }
        let component_id = Res::<'a, RenderResInner<T>>::init_state(world, system_meta);
        world
            .get_resource_or_insert_with(RenderResRegistry::default)
            .component_ids
            .insert(component_id);
        component_id
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &bevy_ecs::system::SystemMeta,
        world: bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy_ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        let inner = Res::<'a, RenderResInner<T>>::get_param(state, system_meta, world, change_tick);
        RenderRes { inner }
    }
    fn new_archetype(
        state: &mut Self::State,
        archetype: &bevy_ecs::archetype::Archetype,
        system_meta: &mut bevy_ecs::system::SystemMeta,
    ) {
        Res::<'a, RenderResInner<T>>::new_archetype(state, archetype, system_meta);
    }
    fn apply(
        state: &mut Self::State,
        system_meta: &bevy_ecs::system::SystemMeta,
        world: &mut bevy_ecs::world::World,
    ) {
        Res::<'a, RenderResInner<T>>::apply(state, system_meta, world);
    }
}

unsafe impl<'a, T: Resource> SystemParam for RenderResMut<'a, T> {
    type State = ComponentId;

    type Item<'world, 'state> = RenderResMut<'world, T>;

    fn init_state(
        world: &mut bevy_ecs::world::World,
        system_meta: &mut bevy_ecs::system::SystemMeta,
    ) -> Self::State {
        if !system_meta.default_config.has::<RenderSystemConfig>() {
            panic!("RenderResMut<{}> can only be used in a render system, but {} is not. RenderCommands must be the first parameter of a render system.", std::any::type_name::<T>(), system_meta.name());
        }
        let component_id = ResMut::<'a, RenderResInner<T>>::init_state(world, system_meta);
        world
            .get_resource_or_insert_with(RenderResRegistry::default)
            .component_ids
            .insert(component_id);
        component_id
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &bevy_ecs::system::SystemMeta,
        world: bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy_ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        let inner =
            ResMut::<'a, RenderResInner<T>>::get_param(state, system_meta, world, change_tick);
        RenderResMut { inner }
    }

    fn new_archetype(
        state: &mut Self::State,
        archetype: &bevy_ecs::archetype::Archetype,
        system_meta: &mut bevy_ecs::system::SystemMeta,
    ) {
        ResMut::<'a, RenderResInner<T>>::new_archetype(state, archetype, system_meta);
    }
    fn apply(
        state: &mut Self::State,
        system_meta: &bevy_ecs::system::SystemMeta,
        world: &mut bevy_ecs::world::World,
    ) {
        ResMut::<'a, RenderResInner<T>>::apply(state, system_meta, world);
    }
}
