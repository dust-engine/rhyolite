use std::mem;

use ash::{vk, Device};

use crate::ecs::queue_cap::IsQueueCap;

pub struct CommandRecorder<'w, const Q: char>
where
    (): IsQueueCap<Q>,
{
    pub(crate) device: &'w Device,
    pub(crate) cmd_buf: vk::CommandBuffer,
}

impl<'w, const Q: char> CommandRecorder<'w, Q>
where
    (): IsQueueCap<Q>,
{
    pub fn clear_color_image(
        &mut self,
        image: vk::Image,
        image_layout: vk::ImageLayout,
        clear_color_value: &vk::ClearColorValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        unsafe {
            self.device.cmd_clear_color_image(
                self.cmd_buf,
                image,
                image_layout,
                clear_color_value,
                ranges,
            )
        }
    }

    pub fn pipeline_barrier(
        &mut self,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier],
        buffer_memory_barriers: &[vk::BufferMemoryBarrier],
        image_memory_barriers: &[vk::ImageMemoryBarrier],
    ) {
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.cmd_buf,
                src_stage_mask,
                dst_stage_mask,
                dependency_flags,
                memory_barriers,
                buffer_memory_barriers,
                image_memory_barriers,
            )
        }
    }
}
