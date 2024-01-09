use std::collections::BTreeSet;

use bevy_ecs::{
    component::{ComponentId, Component},
    system::{Res, ResMut, Resource, SystemParam}, query::{WorldQuery, QueryData, ReadOnlyQueryData}, world::Mut,
};
use bevy_utils::ConfigMap;

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
        let component_id = Res::<'a, RenderResInner<T>>::init_state(world, system_meta);
        world
            .get_resource_or_insert_with(RenderResRegistry::default)
            .component_ids
            .insert(component_id);
        component_id
    }

    fn default_configs(config: &mut ConfigMap) {
        if !config.has::<RenderSystemConfig>() {
            panic!("RenderRes<{}> can only be used in a render system. RenderCommands must be the first parameter of a render system.", std::any::type_name::<T>());
        }
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
        let component_id = ResMut::<'a, RenderResInner<T>>::init_state(world, system_meta);
        world
            .get_resource_or_insert_with(RenderResRegistry::default)
            .component_ids
            .insert(component_id);
        component_id
    }

    fn default_configs(config: &mut ConfigMap) {
        if !config.has::<RenderSystemConfig>() {
            panic!("RenderResMut<{}> can only be used in a render system. RenderCommands must be the first parameter of a render system.", std::any::type_name::<T>());
        }
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


pub struct RenderComponent<'a, T: Component> {
    inner: &'a T,
}
impl<T: Component> RenderComponent<'_, T> {
    pub unsafe fn get_on_host(&self) -> &T {
        self.inner
    }
}


unsafe impl<T: Component> WorldQuery for RenderComponent<'_, T> {
    type Item<'w> = RenderComponent<'w, T>;

    type Fetch<'w> = bevy_ecs::query::ReadFetch<'w, T>;

    type State = ComponentId;

    fn shrink<'wlong: 'wshort, 'wshort>(item: Self::Item<'wlong>) -> Self::Item<'wshort> {
        item
    }

    unsafe fn init_fetch<'w>(
        world: bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell<'w>,
        state: &Self::State,
        last_run: bevy_ecs::component::Tick,
        this_run: bevy_ecs::component::Tick,
    ) -> Self::Fetch<'w> {
        <&T as WorldQuery>::init_fetch(world, state, last_run, this_run)
    }

    const IS_DENSE: bool = <&T as WorldQuery>::IS_DENSE;

    unsafe fn set_archetype<'w>(
        fetch: &mut Self::Fetch<'w>,
        state: &Self::State,
        archetype: &'w bevy_ecs::archetype::Archetype,
        table: &'w bevy_ecs::storage::Table,
    ) {
        <&T as WorldQuery>::set_archetype(fetch, state, archetype, table)
    }

    unsafe fn set_table<'w>(fetch: &mut Self::Fetch<'w>, state: &Self::State, table: &'w bevy_ecs::storage::Table) {
        <&T as WorldQuery>::set_table(fetch, state, table);
    }

    unsafe fn fetch<'w>(
        fetch: &mut Self::Fetch<'w>,
        entity: bevy_ecs::entity::Entity,
        table_row: bevy_ecs::storage::TableRow,
    ) -> Self::Item<'w> {
        let inner = <&T as WorldQuery>::fetch(fetch, entity, table_row);
        RenderComponent { inner }
    }

    fn update_component_access(state: &Self::State, access: &mut bevy_ecs::query::FilteredAccess<ComponentId>) {
        <&T as WorldQuery>::update_component_access(state, access);
    }

    fn update_archetype_component_access(
        state: &Self::State,
        archetype: &bevy_ecs::archetype::Archetype,
        access: &mut bevy_ecs::query::Access<bevy_ecs::archetype::ArchetypeComponentId>,
    ) {
        <&T as WorldQuery>::update_archetype_component_access(state, archetype, access);
    }

    fn init_state(world: &mut bevy_ecs::world::World) -> Self::State {
        <&T as WorldQuery>::init_state(world)
    }

    fn matches_component_set(
        state: &Self::State,
        set_contains_id: &impl Fn(ComponentId) -> bool,
    ) -> bool {
        <&T as WorldQuery>::matches_component_set(state, set_contains_id)
    }
}
unsafe impl<T: Component> ReadOnlyQueryData for RenderComponent<'_, T> {
}
unsafe impl<T: Component> QueryData for RenderComponent<'_, T> {
    type ReadOnly = Self;
}


pub struct RenderComponentMut<'a, T: Component> {
    inner: Mut<'a, T>,
}


unsafe impl<T: Component> WorldQuery for RenderComponentMut<'_, T> {
    type Item<'w> = RenderComponentMut<'w, T>;

    type Fetch<'w> = bevy_ecs::query::WriteFetch<'w, T>;

    type State = ComponentId;

    fn shrink<'wlong: 'wshort, 'wshort>(item: Self::Item<'wlong>) -> Self::Item<'wshort> {
        item
    }

    unsafe fn init_fetch<'w>(
        world: bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell<'w>,
        state: &Self::State,
        last_run: bevy_ecs::component::Tick,
        this_run: bevy_ecs::component::Tick,
    ) -> Self::Fetch<'w> {
        <&mut T as WorldQuery>::init_fetch(world, state, last_run, this_run)
    }

    const IS_DENSE: bool = <&T as WorldQuery>::IS_DENSE;

    unsafe fn set_archetype<'w>(
        fetch: &mut Self::Fetch<'w>,
        state: &Self::State,
        archetype: &'w bevy_ecs::archetype::Archetype,
        table: &'w bevy_ecs::storage::Table,
    ) {
        <&mut T as WorldQuery>::set_archetype(fetch, state, archetype, table)
    }

    unsafe fn set_table<'w>(fetch: &mut Self::Fetch<'w>, state: &Self::State, table: &'w bevy_ecs::storage::Table) {
        <&mut T as WorldQuery>::set_table(fetch, state, table);
    }

    unsafe fn fetch<'w>(
        fetch: &mut Self::Fetch<'w>,
        entity: bevy_ecs::entity::Entity,
        table_row: bevy_ecs::storage::TableRow,
    ) -> Self::Item<'w> {
        let inner = <&mut T as WorldQuery>::fetch(fetch, entity, table_row);
        RenderComponentMut { inner }
    }

    fn update_component_access(state: &Self::State, access: &mut bevy_ecs::query::FilteredAccess<ComponentId>) {
        <&mut T as WorldQuery>::update_component_access(state, access);
    }

    fn update_archetype_component_access(
        state: &Self::State,
        archetype: &bevy_ecs::archetype::Archetype,
        access: &mut bevy_ecs::query::Access<bevy_ecs::archetype::ArchetypeComponentId>,
    ) {
        <&mut T as WorldQuery>::update_archetype_component_access(state, archetype, access);
    }

    fn init_state(world: &mut bevy_ecs::world::World) -> Self::State {
        let component_id = <&mut T as WorldQuery>::init_state(world);
        world
            .get_resource_or_insert_with(RenderResRegistry::default)
            .component_ids
            .insert(component_id);
        component_id
    }

    fn matches_component_set(
        state: &Self::State,
        set_contains_id: &impl Fn(ComponentId) -> bool,
    ) -> bool {
        <&mut T as WorldQuery>::matches_component_set(state, set_contains_id)
    }
}
unsafe impl<'a, T: Component> QueryData for RenderComponentMut<'a, T> {
    type ReadOnly = RenderComponent<'a, T>;
}

impl<T: Component> RenderComponentMut<'_, T> {
    pub unsafe fn get_on_host(&self) -> &T {
        self.inner.as_ref()
    }
    pub unsafe fn get_on_host_mut(&mut self) -> &mut T {
        self.inner.as_mut()
    }
}



