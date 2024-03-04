use ash::vk;

use bevy::ecs::system::IntoSystem;
use bevy::ecs::{
    schedule::{IntoSystemConfigs, SystemConfigs},
    system::BoxedSystem,
};

use crate::{queue::QueueType, Access, BufferLike, ImageLike};

use super::{RenderImage, RenderRes, RenderSystemPass};

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

impl Barriers {
    pub fn transition_resource<T>(&mut self, res: &mut RenderRes<T>, access: Access) {
        let global_barriers = unsafe { &mut *self.global_barriers };
        let barrier = res.state.transition(access, false);
        global_barriers.src_stage_mask |= barrier.src_stage_mask;
        global_barriers.dst_stage_mask |= barrier.dst_stage_mask;
        global_barriers.src_access_mask |= barrier.src_access_mask;
        global_barriers.dst_access_mask |= barrier.dst_access_mask;
    }
    pub fn transition_image<T: ImageLike>(
        &mut self,
        image: &mut RenderImage<T>,
        access: Access,
        layout: vk::ImageLayout,
        retain_data: bool,
    ) {
        if access.is_readonly() && !retain_data {
            tracing::warn!("Transitioning an image to readonly access without retaining image data. This is likely an error.");
        }
        if access.is_writeonly() && retain_data {
            tracing::warn!("Transitioning an image to writeonly access while retaining image data. This is likely inefficient.");
        }
        let image_barriers = unsafe { &mut *self.image_barriers };
        let global_barriers = unsafe { &mut *self.global_barriers };
        let has_layout_transition = image.layout != layout;
        let barrier = image.res.state.transition(access, has_layout_transition);

        if has_layout_transition {
            image_barriers.push(vk::ImageMemoryBarrier2 {
                src_stage_mask: barrier.src_stage_mask,
                dst_stage_mask: barrier.dst_stage_mask,
                src_access_mask: barrier.src_access_mask,
                dst_access_mask: barrier.dst_access_mask,
                old_layout: if retain_data {
                    image.layout
                } else {
                    vk::ImageLayout::UNDEFINED
                },
                new_layout: layout,
                image: image.raw_image(),
                subresource_range: image.subresource_range(),
                ..Default::default()
            });
            image.layout = layout;
        } else {
            global_barriers.src_stage_mask |= barrier.src_stage_mask;
            global_barriers.dst_stage_mask |= barrier.dst_stage_mask;
            global_barriers.src_access_mask |= barrier.src_access_mask;
            global_barriers.dst_access_mask |= barrier.dst_access_mask;
        }
    }
    pub fn transition_buffer<T: BufferLike>(&mut self, buffer: &mut RenderRes<T>, access: Access) {
        let global_barriers = unsafe { &mut *self.global_barriers };
        let barrier = buffer.state.transition(access, false);
        global_barriers.src_stage_mask |= barrier.src_stage_mask;
        global_barriers.dst_stage_mask |= barrier.dst_stage_mask;
        global_barriers.src_access_mask |= barrier.src_access_mask;
        global_barriers.dst_access_mask |= barrier.dst_access_mask;
        println!("{:?}", global_barriers);
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
