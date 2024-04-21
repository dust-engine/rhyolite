use super::CommandRecorder;
use ash::vk;

pub trait TransferCommands: CommandRecorder {
    fn copy_buffer(&mut self, src: vk::Buffer, dst: vk::Buffer, regions: &[vk::BufferCopy]) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device().cmd_copy_buffer(cmd_buf, src, dst, regions);
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
                .cmd_copy_buffer_to_image(cmd_buf, src, dst, layout, regions);
        }
    }
    /// Copying multiple regions of the same buffer
    fn batch_copy(&mut self) -> BatchCopy<'_, Self>
    where
        Self: Sized,
    {
        BatchCopy {
            recorder: self,
            src: vk::Buffer::null(),
            dst: vk::Buffer::null(),
            copies: Vec::new(),
        }
    }
}

impl<T: CommandRecorder> TransferCommands for T {}

pub struct BatchCopy<'a, T: TransferCommands> {
    recorder: &'a mut T,
    src: vk::Buffer,
    dst: vk::Buffer,
    copies: Vec<vk::BufferCopy>,
}
impl<T: TransferCommands> BatchCopy<'_, T> {
    fn flush(&mut self) {
        if !self.copies.is_empty() {
            self.recorder.copy_buffer(self.src, self.dst, &self.copies);
            self.copies.clear();
        }
    }
    pub fn copy_buffer(&mut self, src: vk::Buffer, dst: vk::Buffer, regions: &[vk::BufferCopy]) {
        if self.src != vk::Buffer::null() && self.dst != vk::Buffer::null() {
            if self.src != src || self.dst != dst {
                self.flush();
                self.src = src;
                self.dst = dst;
            }
        }
        self.copies.extend_from_slice(regions);
    }
}
impl<T: TransferCommands> Drop for BatchCopy<'_, T> {
    fn drop(&mut self) {
        self.flush();
    }
}
