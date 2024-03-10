use ash::vk;

use bevy::ecs::system::IntoSystem;
use bevy::ecs::{
    schedule::{IntoSystemConfigs, SystemConfigs},
    system::BoxedSystem,
};

use crate::commands::ResourceTransitionCommands;
use crate::{queue::QueueType, Access};

use super::RenderSystemPass;

pub struct RenderSystemConfig {
    /// The render system must be assigned onto a queue supporting these feature flags.
    pub queue: QueueType,
    pub force_binary_semaphore: bool,
    pub is_queue_op: bool,
    pub barrier_producer: Option<BoxedBarrierProducer>,
}
impl Default for RenderSystemConfig {
    fn default() -> Self {
        Self {
            queue: QueueType::Graphics,
            force_binary_semaphore: false,
            is_queue_op: false,
            barrier_producer: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueueSystemDependencyConfig {
    pub wait: Access,
    pub signal: Access,
}
impl Default for QueueSystemDependencyConfig {
    fn default() -> Self {
        Self {
            wait: Access {
                stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                access: vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ,
            },
            signal: Access {
                stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                access: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            },
        }
    }
}

/// GAT limitation. All references contained herein are valid for the duration of the system call.
pub struct Barriers {
    pub(crate) dependency_flags: *mut vk::DependencyFlags,
    pub(crate) image_barriers: *mut Vec<vk::ImageMemoryBarrier2>,
    pub(crate) buffer_barriers: *mut Vec<vk::BufferMemoryBarrier2>,
    pub(crate) global_barriers: *mut vk::MemoryBarrier2,
    pub(crate) dropped: *mut bool,
}
impl Drop for Barriers {
    fn drop(&mut self) {
        unsafe {
            *self.dropped = true;
        }
    }
}
impl ResourceTransitionCommands for Barriers {
    fn add_global_barrier(&mut self, barrier: vk::MemoryBarrier2) -> &mut Self {
        let current = unsafe { &mut *self.global_barriers };
        current.src_stage_mask |= barrier.src_stage_mask;
        current.dst_stage_mask |= barrier.dst_stage_mask;
        current.src_access_mask |= barrier.src_access_mask;
        current.dst_access_mask |= barrier.dst_access_mask;
        self
    }

    fn add_image_barrier(&mut self, barrier: vk::ImageMemoryBarrier2) -> &mut Self {
        let current = unsafe { &mut *self.image_barriers };
        current.push(barrier);
        self
    }

    fn add_buffer_barrier(&mut self, barrier: vk::BufferMemoryBarrier2) -> &mut Self {
        let current = unsafe { &mut *self.buffer_barriers };
        current.push(barrier);
        self
    }

    fn set_dependency_flags(&mut self, flags: vk::DependencyFlags) -> &mut Self {
        unsafe {
            *self.dependency_flags |= flags;
        }
        self
    }
}

pub type BoxedBarrierProducer = BoxedSystem<Barriers, ()>;

pub trait IntoRenderSystemConfigs<Marker>: IntoSystemConfigs<Marker>
where
    Self: Sized,
{
    /// Ensure that this render system will be assigned to a queue supporting the specified queue flags
    /// while minimizing the amount of overhead associated with semaphore syncronizations.
    /// Should be called for most smaller render systems in-between heavier operations.
    fn on_queue(self, queue_type: QueueType) -> SystemConfigs {
        self.with_option::<RenderSystemPass>(|entry| {
            let config = entry.or_default();
            config.queue = queue_type;
        })
    }
    fn with_barriers<M, T: IntoSystem<Barriers, (), M>>(self, barriers: T) -> SystemConfigs {
        let mut barriers: Option<BoxedBarrierProducer> = Some(Box::new(T::into_system(barriers)));
        self.with_option::<RenderSystemPass>(move |entry| {
            let config = entry.or_default();
            if barriers.is_none() {
                unimplemented!();
                // TODO: allow collective barriers for multiple systems
            }
            config.barrier_producer = barriers.take();
        })
    }
}

impl<Marker, T> IntoRenderSystemConfigs<Marker> for T where T: IntoSystemConfigs<Marker> {}
