use super::CommandRecorder;
use crate::buffer::BufferLike;
use ash::vk;

pub trait TransferCommands: CommandRecorder {
    fn copy_buffer(&mut self, src: vk::Buffer, dst: vk::Buffer, regions: &[vk::BufferCopy]) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .cmd_copy_buffer(cmd_buf, src.raw_buffer(), dst.raw_buffer(), regions);
        }
    }
    fn copy_buffer_to_image(
        &mut self,
        src: vk::Buffer,
        dst: vk::Image,
        layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .cmd_copy_buffer_to_image(cmd_buf, src.raw_buffer(), dst, layout, regions);
        }
    }
}

impl<T: CommandRecorder> TransferCommands for T {}
