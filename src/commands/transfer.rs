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
}

impl<T: CommandRecorder> TransferCommands for T {}
