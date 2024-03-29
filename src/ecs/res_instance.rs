use std::{
    any::TypeId,
    ops::{Deref, DerefMut},
};

use bevy::{
    ecs::{
        change_detection::{Ticks, TicksMut},
        component::ComponentId,
        system::{InstancedResource, SystemMeta, SystemParam},
        world::World,
    },
    ptr::UnsafeCellDeref,
};

pub struct ResInstance<'w, T: InstancedResource> {
    pub(crate) value: &'w T,
    pub(crate) ticks: Ticks<'w>,
}

pub struct ResInstanceConfig {
    type_id: TypeId,
    component_id: ComponentId,
}
impl ResInstanceConfig {
    pub fn for_type<T: 'static>(component_id: ComponentId) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            component_id,
        }
    }
}

unsafe impl<T: InstancedResource> SystemParam for ResInstance<'_, T> {
    type State = Option<ComponentId>;

    type Item<'world, 'state> = ResInstance<'world, T>;

    fn init_state(
        world: &mut bevy::prelude::World,
        system_meta: &mut bevy::ecs::system::SystemMeta,
    ) -> Self::State {
        None
    }
    fn configurate(
        state: &mut Self::State,
        config: &mut dyn std::any::Any,
        system_meta: &mut SystemMeta,
        world: &mut World,
    ) {
        if let Some(config) = config.downcast_ref::<ResInstanceConfig>() {
            if config.type_id == TypeId::of::<T>() {
                let component_id = config.component_id;
                *state = Some(component_id);

                let combined_access = system_meta.component_access_set.combined_access();
                assert!(
                    !combined_access.has_write(component_id),
                    "error[B0002]: ResInstance<{}> in system {} conflicts with a previous ResMut<{0}> access. Consider removing the duplicate access.",
                    std::any::type_name::<T>(),
                    system_meta.name,
                );
                system_meta
                    .component_access_set
                    .add_unfiltered_read(component_id);

                let archetype_component_id = world
                    .get_resource_archetype_component_id(component_id)
                    .unwrap();
                system_meta
                    .archetype_component_access
                    .add_read(archetype_component_id);
            }
        }
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &bevy::ecs::system::SystemMeta,
        world: bevy::ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy::ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        let component_id = state.expect("ResInstance must be configured before use.");
        let (value, ticks) = world
            .get_resource_with_ticks(component_id)
            .unwrap_or_else(|| {
                panic!(
                    "Resource requested by {} does not exist: {}",
                    system_meta.name,
                    std::any::type_name::<T>()
                )
            });
        ResInstance {
            value: value.deref::<T>(),
            ticks: Ticks {
                added: ticks.added.deref(),
                changed: ticks.changed.deref(),
                last_run: system_meta.last_run,
                this_run: change_tick,
            },
        }
    }
}

pub struct ResInstanceMut<'w, T: InstancedResource> {
    pub(crate) value: &'w mut T,
    pub(crate) ticks: TicksMut<'w>,
}
impl<'a, T: InstancedResource> ResInstanceMut<'a, T> {
    pub fn into_inner(self) -> &'a mut T {
        *self.ticks.changed = self.ticks.this_run;
        self.value
    }
}
impl<T: InstancedResource> Deref for ResInstanceMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
    }
}
impl<T: InstancedResource> DerefMut for ResInstanceMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        *self.ticks.changed = self.ticks.this_run;
        self.value
    }
}

unsafe impl<T: InstancedResource> SystemParam for ResInstanceMut<'_, T> {
    type State = Option<ComponentId>;

    type Item<'world, 'state> = ResInstanceMut<'world, T>;

    fn init_state(
        _world: &mut bevy::prelude::World,
        _system_meta: &mut bevy::ecs::system::SystemMeta,
    ) -> Self::State {
        None
    }
    fn configurate(
        state: &mut Self::State,
        config: &mut dyn std::any::Any,
        system_meta: &mut SystemMeta,
        world: &mut World,
    ) {
        if let Some(config) = config.downcast_ref::<ResInstanceConfig>() {
            if config.type_id == TypeId::of::<T>() {
                let component_id = config.component_id;
                *state = Some(component_id);
                let combined_access = system_meta.component_access_set.combined_access();
                if combined_access.has_write(component_id) {
                    panic!(
                        "error[B0002]: ResMut<{}> in system {} conflicts with a previous ResMut<{0}> access. Consider removing the duplicate access.",
                        std::any::type_name::<T>(), system_meta.name);
                } else if combined_access.has_read(component_id) {
                    panic!(
                        "error[B0002]: ResMut<{}> in system {} conflicts with a previous Res<{0}> access. Consider removing the duplicate access.",
                        std::any::type_name::<T>(), system_meta.name);
                }
                system_meta
                    .component_access_set
                    .add_unfiltered_write(component_id);

                let archetype_component_id = world
                    .get_resource_archetype_component_id(component_id)
                    .unwrap();
                system_meta
                    .archetype_component_access
                    .add_write(archetype_component_id);
            }
        }
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &bevy::ecs::system::SystemMeta,
        world: bevy::ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy::ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        let component_id = state.expect("ResInstance must be configured before use.");
        let value = world
            .get_resource_mut_by_id(component_id)
            .unwrap_or_else(|| {
                panic!(
                    "Resource requested by {} does not exist: {}",
                    system_meta.name,
                    std::any::type_name::<T>()
                )
            });
        ResInstanceMut {
            value: value.value.deref_mut::<T>(),
            ticks: TicksMut {
                added: value.ticks.added,
                changed: value.ticks.changed,
                last_run: system_meta.last_run,
                this_run: change_tick,
            },
        }
    }
}
