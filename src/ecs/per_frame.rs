use std::{
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
};

use ash::vk;
use bevy::{ecs::{
    component::ComponentId, schedule::IntoSystemSet, system::{ResMut, Resource, System, SystemParam, SystemParamItem}
}, utils::smallvec::SmallVec};

use crate::{semaphore::{self, TimelineSemaphore}, Device, HasDevice};

use super::{queue_cap::IsQueueCap, QueueSubmissionInfo, RenderCommands, RenderRes, RenderSystemInitialState};

pub trait PerFrameResource: Send + Sync + 'static {
    type Param<'a>;
    fn reset(&mut self, _params: Self::Param<'_>) {}
    fn create(params: Self::Param<'_>) -> Self;
}

/// Wraps a host-owned resource, allowing the GPU to borrow it.
#[derive(Resource)]
pub struct PerFrame<T: PerFrameResource> {
    frame_index: u64,
    items: SmallVec<[PerFrameResourceFrame<T>; 3]>,
}
impl<T: PerFrameResource> Default for PerFrame<T> {
    fn default() -> Self {
        Self {
            frame_index: 0,
            items: SmallVec::from_buf([PerFrameResourceFrame::Empty, PerFrameResourceFrame::Empty, PerFrameResourceFrame::Empty])
        }
    }
}

enum PerFrameResourceFrame<T> {
    Empty,
    Some { frame: T, semaphores: SmallVec<[(Arc<TimelineSemaphore>, u64); 4]> },
}

impl<T: PerFrameResource> PerFrame<T> {
    pub fn new() -> Self {
        Self {
            frame_index: 0,
            items: SmallVec::new(),
        }
    }
    pub(crate) fn on_frame_index<const Q: char>(&mut self, frame_index: u64, submission_info: &Mutex<QueueSubmissionInfo>, device: &Device, param: T::Param<'_>) -> &mut T where (): IsQueueCap<Q> {
        let i = (frame_index % self.items.len() as u64) as usize;
        if frame_index != self.frame_index {
            match &mut self.items[i] {
                PerFrameResourceFrame::Empty => {
                    self.items[i] = PerFrameResourceFrame::Some { frame: T::create(param), semaphores: SmallVec::new() };
                },
                PerFrameResourceFrame::Some { frame, semaphores } => {
                    println!("waiting for semaphores: {:?}", semaphores);
                    TimelineSemaphore::wait_all_blocked(semaphores.iter().map(|(s, v)| (s.as_ref(), *v)), !0).unwrap();
                    println!("waited");
                    frame.reset(param);
                    semaphores.clear();
                },
            }
            self.frame_index = frame_index;
        }
        if let PerFrameResourceFrame::Some { frame, semaphores } = &mut self.items[i as usize] {
            let mut submission_info = submission_info.lock().unwrap();
            let (semaphore, value) = submission_info.signal_semaphore(vk::PipelineStageFlags2::ALL_COMMANDS, device);
            for (current_semaphore, current_value) in semaphores.iter_mut() {
                if Arc::ptr_eq(current_semaphore, &semaphore) {
                    *current_value = value;
                    return frame;
                }
            }
            semaphores.push((semaphore, value));
            return frame;
        } else {
            panic!()
        }
    }
    pub fn on_frame<const Q: char>(&mut self, commands: &RenderCommands<Q>, param: T::Param<'_>) -> &mut T where (): IsQueueCap<Q> {
        self.on_frame_index(commands.frame_index, &commands.submission_info, &commands.device(), param)
    }
}
