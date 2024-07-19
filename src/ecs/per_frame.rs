use std::sync::{Arc, Mutex};

use ash::vk;
use bevy::{ecs::system::{InstancedResource, Resource}, prelude::FromWorld};

use crate::semaphore::TimelineSemaphore;
use smallvec::SmallVec;

use super::{queue_cap::IsQueueCap, QueueSubmissionInfo, RenderCommands};

/// Wraps a host-owned resource, allowing the GPU to borrow it.
#[derive(Resource)]
pub struct PerFrame<T> {
    frame_index: u64,
    items: SmallVec<[PerFrameResourceFrame<T>; 3]>,
}
impl<T: Resource> InstancedResource for PerFrame<T> {}

pub trait PerFrameReset {
    fn reset(&mut self);
}
impl<T> PerFrameReset for T {
    default fn reset(&mut self) {}
}

struct PerFrameResourceFrame<T> {
    frame: T,
    semaphores: SmallVec<[(Arc<TimelineSemaphore>, u64); 4]>,
}
impl<T> Drop for PerFrameResourceFrame<T> {
    fn drop(&mut self) {
        TimelineSemaphore::wait_all_blocked(
            self.semaphores
                .iter()
                .map(|(semaphore, value)| (semaphore.as_ref(), *value)),
            !0,
        )
        .unwrap();
    }
}


impl<T: FromWorld> FromWorld for PerFrame<T> {
    fn from_world(world: &mut bevy::prelude::World) -> Self {
        Self::new(|_| {
            T::from_world(world)
        })
    }
}
impl<T: PerFrameReset> PerFrame<T> {
    pub fn new(mut create_callback: impl FnMut(u32) -> T) -> Self {
        Self {
            frame_index: u64::MAX,
            items: SmallVec::from_buf([0, 1, 2].map(|i| PerFrameResourceFrame {
                frame: create_callback(i),
                semaphores: SmallVec::new(),
            })), // num_frames_in_flight elements
        }
    }
    pub(crate) fn on_frame_index(
        &mut self,
        frame_index: u64,
        submission_info: &Mutex<QueueSubmissionInfo>,
    ) -> &mut T {
        let i = (frame_index % self.items.len() as u64) as usize;
        let PerFrameResourceFrame { frame, semaphores } = &mut self.items[i];
        if frame_index != self.frame_index {
            TimelineSemaphore::wait_all_blocked(
                semaphores.iter().map(|(s, v)| (s.as_ref(), *v)),
                !0,
            )
            .unwrap();
            frame.reset();
            semaphores.clear();
            self.frame_index = frame_index;
        }
        let mut submission_info = submission_info.lock().unwrap();
        let (semaphore, value) =
            submission_info.signal_semaphore(vk::PipelineStageFlags2::ALL_COMMANDS);
        for (current_semaphore, current_value) in semaphores.iter_mut() {
            if Arc::ptr_eq(current_semaphore, &semaphore) {
                *current_value = value;
                return frame;
            }
        }
        semaphores.push((semaphore, value));
        return frame;
    }
    pub fn on_frame<const Q: char>(&mut self, commands: &RenderCommands<Q>) -> &mut T
    where
        (): IsQueueCap<Q>,
    {
        self.on_frame_index(commands.frame_index, &commands.submission_info)
    }
}
