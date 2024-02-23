use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use bevy_ecs::{
    component::ComponentId,
    system::{ResMut, Resource, System, SystemParam, SystemParamItem},
};

use crate::semaphore::TimelineSemaphore;

use super::RenderSystemInitialState;

pub trait PerFrameResource: Resource {
    type Params: SystemParam;
    fn reset(&mut self, _params: SystemParamItem<'_, '_, Self::Params>) {}
    fn create(params: SystemParamItem<'_, '_, Self::Params>) -> Self;
}

pub struct PerFrameMut<'a, T: PerFrameResource> {
    index: usize,
    items: ResMut<'a, PerFrameResourceContainer<T>>,
}

impl<'a, T: PerFrameResource> Deref for PerFrameMut<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match &self.items.frames[self.index] {
            PerFrameResourceFrame::Some {
                frame,
                last_used: _,
            } => frame,
            _ => panic!(),
        }
    }
}
impl<'a, T: PerFrameResource> DerefMut for PerFrameMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match &mut self.items.frames[self.index] {
            PerFrameResourceFrame::Some {
                frame,
                last_used: _,
            } => frame,
            _ => panic!(),
        }
    }
}

enum PerFrameResourceFrame<T> {
    Empty,
    Some { frame: T, last_used: u64 },
}
#[derive(Resource)]
struct PerFrameResourceContainer<T> {
    semaphores: Vec<Arc<TimelineSemaphore>>,
    frames: Vec<PerFrameResourceFrame<T>>,
}

pub struct PerFrameState<T: PerFrameResource> {
    param_state: <T::Params as SystemParam>::State,
    frame_index: u64,
    component_id: ComponentId,
}

// The problem we face rn is that the same resource may be used by different systems in different
// submission passes right?
// So an earlier pass now needs to wait on the later pass. It seems that there's no escape from this.
// If most resources use the same resource, then why don't we just perform the syncronization once
// at the beginning of the frame...

// The resource needs to know all the semaphores it needs to wait on.

// Command pools.
// Intermediate command buffers are stored in command pools. We call this struct "Command Recorder".
// Applications choose the command recorder they'd like      use.
// When flushing, all command recorders must be notified to flush.

// Where to record those pipeline barriers.
// We record them with the default command recorder. We also flush the command recorder.
// Not using the default command recorder, and you'll have to do these things manually.

unsafe impl<'a, T: PerFrameResource> SystemParam for PerFrameMut<'a, T> {
    type State = PerFrameState<T>;

    type Item<'world, 'state> = PerFrameMut<'world, T>;

    fn init_state(
        world: &mut bevy_ecs::world::World,
        system_meta: &mut bevy_ecs::system::SystemMeta,
    ) -> Self::State {
        let component_id = ResMut::<PerFrameResourceContainer<T>>::init_state(world, system_meta);
        let param_state = T::Params::init_state(world, system_meta);
        let num_frame_in_flight = 3;

        if world.get_resource_by_id(component_id).is_none() {
            world.insert_resource(PerFrameResourceContainer::<T> {
                semaphores: Vec::new(),
                frames: (0..num_frame_in_flight)
                    .map(|_| PerFrameResourceFrame::Empty)
                    .collect(),
            });
        }
        PerFrameState {
            frame_index: 0,
            component_id,
            param_state,
        }
    }

    fn configurate(
        state: &mut Self::State,
        config: &mut dyn std::any::Any,
        world: &mut bevy_ecs::world::World,
    ) {
        if config.is::<RenderSystemInitialState>() {
            let initial_state: &mut RenderSystemInitialState = config.downcast_mut().unwrap();
            let res = world.get_resource_mut_by_id(state.component_id).unwrap();
            let mut res = unsafe { res.with_type::<PerFrameResourceContainer<T>>() };
            res.semaphores.push(initial_state.timeline_signal.clone());
        }
        T::Params::configurate(&mut state.param_state, config, world);
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &bevy_ecs::system::SystemMeta,
        world: bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy_ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        state.frame_index += 1;
        let mut res = ResMut::<PerFrameResourceContainer<T>>::get_param(
            &mut state.component_id,
            system_meta,
            world,
            change_tick,
        );
        let num_frame_in_flight = 3;
        if state.frame_index > num_frame_in_flight {
            let value = state.frame_index - 3;
            let semaphores = res.semaphores.iter().map(|s| (s.as_ref(), value));
            TimelineSemaphore::wait_all_blocked(semaphores, !0).unwrap();
        }
        let index = (state.frame_index % num_frame_in_flight) as usize;
        let params = T::Params::get_param(&mut state.param_state, system_meta, world, change_tick);
        match &mut res.frames[index] {
            PerFrameResourceFrame::Empty => {
                res.frames[index] = PerFrameResourceFrame::Some {
                    frame: T::create(params),
                    last_used: state.frame_index,
                };
            }
            PerFrameResourceFrame::Some { frame, last_used } => {
                if *last_used < state.frame_index {
                    frame.reset(params);
                    *last_used = state.frame_index;
                }
            }
        }
        PerFrameMut {
            index: (state.frame_index % num_frame_in_flight) as usize,
            items: res,
        }
    }
}
