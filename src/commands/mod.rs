use ash::{vk, Device};

use crate::ecs::queue_cap::IsQueueCap;

pub struct CommandRecorder<'w, const Q: char> where (): IsQueueCap<Q>{
    pub(crate) device: &'w Device,
    pub(crate) cmd_buf: vk::CommandBuffer,
}

impl<'w, const Q: char> CommandRecorder<'w, Q> where (): IsQueueCap<Q> {
    pub fn clear_color_image(&mut self, 
        image: vk::Image,
        image_layout: vk::ImageLayout,
        clear_color_value: &vk::ClearColorValue,
        ranges: &[vk::ImageSubresourceRange]) {
        unsafe {
            self.device.cmd_clear_color_image(
                self.cmd_buf,
                image,
                image_layout,
                clear_color_value,
                ranges
            )
        }
    }
}
