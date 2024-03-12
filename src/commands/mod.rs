use ash::vk;

use crate::{
    buffer::BufferLike,
    ecs::{RenderImage, RenderRes},
    Access, Device, HasDevice, ImageLike, QueueType,
};

mod render;
mod transfer;
pub use render::*;
pub use transfer::*;

pub trait CommandRecorder: HasDevice {
    const QUEUE_CAP: char;
    fn cmd_buf(&mut self) -> vk::CommandBuffer;
    fn current_queue_family(&self) -> (QueueType, u32);
}

pub trait CommonCommands: CommandRecorder {
    fn clear_color_image(
        &mut self,
        image: vk::Image,
        image_layout: vk::ImageLayout,
        clear_color_value: &vk::ClearColorValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device().cmd_clear_color_image(
                cmd_buf,
                image,
                image_layout,
                clear_color_value,
                ranges,
            )
        }
    }
    fn transition_resources(&mut self) -> ImmediateTransitions {
        let cmd_buf = self.cmd_buf();
        ImmediateTransitions {
            queue_family: self.current_queue_family(),
            device: self.device(),
            cmd_buf,
            global_barriers: vk::MemoryBarrier2::default(),
            image_barriers: Vec::new(),
            buffer_barriers: Vec::new(),
            dependency_flags: vk::DependencyFlags::empty(),
        }
    }
}
impl<T> CommonCommands for T where T: CommandRecorder {}

pub struct ImmediateTransitions<'w> {
    pub(crate) device: &'w Device,
    pub(crate) cmd_buf: vk::CommandBuffer,
    pub(crate) global_barriers: vk::MemoryBarrier2,
    pub(crate) image_barriers: Vec<vk::ImageMemoryBarrier2>,
    pub(crate) buffer_barriers: Vec<vk::BufferMemoryBarrier2>,
    pub(crate) dependency_flags: vk::DependencyFlags,
    pub(crate) queue_family: (QueueType, u32),
}

impl Drop for ImmediateTransitions<'_> {
    fn drop(&mut self) {
        if self.global_barriers.src_access_mask == vk::AccessFlags2::empty()
            && self.global_barriers.dst_access_mask == vk::AccessFlags2::empty()
            && self.global_barriers.src_stage_mask == vk::PipelineStageFlags2::empty()
            && self.global_barriers.dst_stage_mask == vk::PipelineStageFlags2::empty()
            && self.image_barriers.is_empty()
            && self.buffer_barriers.is_empty()
        {
            return;
        }
        unsafe {
            self.device.cmd_pipeline_barrier2(
                self.cmd_buf,
                &vk::DependencyInfo {
                    dependency_flags: self.dependency_flags,
                    memory_barrier_count: if self.global_barriers.dst_access_mask
                        == vk::AccessFlags2::empty()
                        && self.global_barriers.src_access_mask == vk::AccessFlags2::empty()
                        && self.global_barriers.src_stage_mask == vk::PipelineStageFlags2::empty()
                        && self.global_barriers.dst_stage_mask == vk::PipelineStageFlags2::empty()
                    {
                        0
                    } else {
                        1
                    },
                    p_memory_barriers: &self.global_barriers,
                    buffer_memory_barrier_count: self.buffer_barriers.len() as u32,
                    p_buffer_memory_barriers: self.buffer_barriers.as_ptr(),
                    image_memory_barrier_count: self.image_barriers.len() as u32,
                    p_image_memory_barriers: self.image_barriers.as_ptr(),
                    ..Default::default()
                },
            )
        }
    }
}
pub trait ResourceTransitionCommands {
    fn add_image_barrier_prev_stage(&mut self, barrier: vk::ImageMemoryBarrier2, prev_queue_type: QueueType) -> &mut Self;
    fn add_buffer_barrier_prev_stage(&mut self, barrier: vk::BufferMemoryBarrier2, prev_queue_type: QueueType) -> &mut Self;

    fn add_global_barrier(&mut self, barrier: vk::MemoryBarrier2) -> &mut Self;
    fn add_image_barrier(&mut self, barrier: vk::ImageMemoryBarrier2) -> &mut Self;
    fn add_buffer_barrier(&mut self, barrier: vk::BufferMemoryBarrier2) -> &mut Self;
    fn set_dependency_flags(&mut self, flags: vk::DependencyFlags) -> &mut Self;

    fn current_queue_family(&self) -> (QueueType, u32);

    fn transition<T>(&mut self, res: &mut RenderRes<T>, access: Access) -> &mut Self {
        let barrier = res.state.transition(access, false);
        self.add_global_barrier(vk::MemoryBarrier2 {
            src_stage_mask: barrier.src_stage_mask,
            dst_stage_mask: barrier.dst_stage_mask,
            src_access_mask: barrier.src_access_mask,
            dst_access_mask: barrier.dst_access_mask,
            ..Default::default()
        });
        self
    }

    fn transition_image<T: ImageLike>(
        &mut self,
        image: &mut RenderImage<T>,
        access: Access,
        layout: vk::ImageLayout,
        retain_data: bool,
    ) -> &mut Self {
        if access.is_readonly() && !retain_data {
            tracing::warn!("Transitioning an image to readonly access without retaining image data. This is likely an error.");
        }
        if access.is_writeonly() && retain_data {
            tracing::warn!("Transitioning an image to writeonly access while retaining image data. This is likely inefficient.");
        }
        let has_layout_transition = image.layout != layout;
        let has_queue_family_ownership_transfer = if let Some(queue_family) = image.res.state.queue_family {
            queue_family != self.current_queue_family()
        } else {
            false
        };
        let barrier = image.res.state.transition(access, has_layout_transition);
        if has_layout_transition || (has_queue_family_ownership_transfer && retain_data) {
            let mut barrier = vk::ImageMemoryBarrier2 {
                src_stage_mask: barrier.src_stage_mask,
                dst_stage_mask: barrier.dst_stage_mask,
                src_access_mask: barrier.src_access_mask,
                dst_access_mask: barrier.dst_access_mask,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: layout,
                image: image.raw_image(),
                subresource_range: image.subresource_range(),
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                ..Default::default()
            };
            if retain_data {
                barrier.old_layout = image.layout;
            }
            if retain_data && has_queue_family_ownership_transfer {
                barrier.src_access_mask = vk::AccessFlags2::empty();
                barrier.src_stage_mask = vk::PipelineStageFlags2::empty();
                barrier.src_queue_family_index = image.res.state.queue_family.unwrap().1;
                barrier.dst_queue_family_index = self.current_queue_family().1;

                
                self.add_image_barrier_prev_stage(vk::ImageMemoryBarrier2 {
                    src_stage_mask: barrier.src_stage_mask,
                    src_access_mask: barrier.src_access_mask,
                    src_queue_family_index: image.state.queue_family.unwrap().1,
                    dst_queue_family_index: self.current_queue_family().1,
                    image: image.raw_image(),
                    subresource_range: image.subresource_range(),
                    ..Default::default()
                }, image.state.queue_family.unwrap().0);
            }
            self.add_image_barrier(barrier);
            image.layout = layout;
        } else {
            self.add_global_barrier(vk::MemoryBarrier2 {
                src_stage_mask: barrier.src_stage_mask,
                dst_stage_mask: barrier.dst_stage_mask,
                src_access_mask: barrier.src_access_mask,
                dst_access_mask: barrier.dst_access_mask,
                ..Default::default()
            });
        }
        image.res.state.queue_family = Some(self.current_queue_family());
        self
    }
    fn transition_buffer<T: BufferLike>(
        &mut self,
        buffer: &mut RenderRes<T>,
        access: Access,
        retain_data: bool,
    ) -> &mut Self {
        let barrier = buffer.state.transition(access, false);
        let has_queue_family_ownership_transfer = if let Some(queue_family) = buffer.state.queue_family {
            queue_family != self.current_queue_family()
        } else {
            false
        };
        if has_queue_family_ownership_transfer && retain_data {
            self.add_buffer_barrier(vk::BufferMemoryBarrier2 {
                dst_stage_mask: barrier.dst_stage_mask,
                dst_access_mask: barrier.dst_access_mask,
                src_queue_family_index: buffer.state.queue_family.unwrap().1,
                dst_queue_family_index: self.current_queue_family().1,
                buffer: buffer.raw_buffer(),
                offset: buffer.offset(),
                size: buffer.size(),
                ..Default::default()
            });
            self.add_buffer_barrier_prev_stage(vk::BufferMemoryBarrier2 {
                src_stage_mask: barrier.src_stage_mask,
                src_access_mask: barrier.src_access_mask,
                src_queue_family_index: buffer.state.queue_family.unwrap().1,
                dst_queue_family_index: self.current_queue_family().1,
                buffer: buffer.raw_buffer(),
                offset: buffer.offset(),
                size: buffer.size(),
                ..Default::default()
            }, buffer.state.queue_family.unwrap().0);
        } else {
            self.add_global_barrier(vk::MemoryBarrier2 {
                src_stage_mask: barrier.src_stage_mask,
                dst_stage_mask: barrier.dst_stage_mask,
                src_access_mask: barrier.src_access_mask,
                dst_access_mask: barrier.dst_access_mask,
                ..Default::default()
            });
        }
        buffer.state.queue_family = Some(self.current_queue_family());
        self
    }
}
impl ResourceTransitionCommands for ImmediateTransitions<'_> {
    fn current_queue_family(&self) -> (QueueType, u32) {
        self.queue_family
    }
    fn add_image_barrier_prev_stage(&mut self, barrier: vk::ImageMemoryBarrier2, queue_type: QueueType) -> &mut Self {
        panic!()
    }
    fn add_buffer_barrier_prev_stage(&mut self, barrier: vk::BufferMemoryBarrier2, queue_type: QueueType) -> &mut Self {
        panic!()
    }
    fn add_global_barrier(&mut self, barrier: vk::MemoryBarrier2) -> &mut Self {
        self.global_barriers.src_stage_mask |= barrier.src_stage_mask;
        self.global_barriers.dst_stage_mask |= barrier.dst_stage_mask;
        self.global_barriers.src_access_mask |= barrier.src_access_mask;
        self.global_barriers.dst_access_mask |= barrier.dst_access_mask;
        self
    }
    fn add_image_barrier(&mut self, barrier: vk::ImageMemoryBarrier2) -> &mut Self {
        self.image_barriers.push(barrier);
        self
    }
    fn add_buffer_barrier(&mut self, barrier: vk::BufferMemoryBarrier2) -> &mut Self {
        self.buffer_barriers.push(barrier);
        self
    }
    fn set_dependency_flags(&mut self, flags: vk::DependencyFlags) -> &mut Self {
        self.dependency_flags = flags;
        self
    }
}
