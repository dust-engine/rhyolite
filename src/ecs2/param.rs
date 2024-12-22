use std::{any::Any, marker::PhantomData, sync::Arc};

use bevy::{
    ecs::system::{SystemMeta, SystemParam},
    prelude::World,
    utils::ConfigMap,
};

use crate::{command::Timeline, Device};

pub struct QueueRenderSystem<'state, Q> {
    state: &'state QueueRenderSystemState,
    _marker: PhantomData<Q>,
}
pub struct QueueRenderSystemState {
    dependencies: Vec<Arc<Timeline>>,
    timeline: Arc<Timeline>,
}
pub struct QueueRenderSystemConfigOut {
    timeline: Arc<Timeline>,
}
pub struct QueueRenderSystemConfigIn {
    dependencies: Vec<Arc<Timeline>>,
}
unsafe impl<Q> SystemParam for QueueRenderSystem<'_, Q> {
    type State = QueueRenderSystemState;

    type Item<'world, 'state> = QueueRenderSystem<'state, Q>;

    fn init_state(world: &mut World, _system_meta: &mut SystemMeta) -> Self::State {
        QueueRenderSystemState {
            dependencies: Vec::new(),
            timeline: Arc::new(Timeline::new(world.resource::<Device>().clone()).unwrap()),
        }
    }

    fn default_configs(state: &mut Self::State, config: &mut ConfigMap) {
        config.insert(QueueRenderSystemConfigOut {
            timeline: state.timeline.clone(),
        });
    }
    fn configurate(
        state: &mut Self::State,
        config: &mut dyn Any,
        _meta: &mut SystemMeta,
        world: &mut World,
    ) {
        if let Some(config) = config.downcast_mut::<QueueRenderSystemConfigIn>() {
            state.dependencies = std::mem::take(&mut config.dependencies);
        }
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &SystemMeta,
        world: bevy::ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy::ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        QueueRenderSystem {
            state,
            _marker: PhantomData,
        }
    }
}
