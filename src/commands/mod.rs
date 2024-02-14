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
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier2],
        buffer_memory_barriers: &[vk::BufferMemoryBarrier2],
        image_memory_barriers: &[vk::ImageMemoryBarrier2],
    ) {
        unsafe {
            self.device.cmd_pipeline_barrier2(
                self.cmd_buf,
                &vk::DependencyInfo {
                    dependency_flags,
                    memory_barrier_count: memory_barriers.len() as u32,
                    p_memory_barriers: memory_barriers.as_ptr(),
                    buffer_memory_barrier_count: buffer_memory_barriers.len() as u32,
                    p_buffer_memory_barriers: buffer_memory_barriers.as_ptr(),
                    image_memory_barrier_count: image_memory_barriers.len() as u32,
                    p_image_memory_barriers: image_memory_barriers.as_ptr(),
                    ..Default::default()
                },
            )
        }
    }
}
