use crate::{
    ecs::{queue_cap::IsQueueCap, RenderCommands},
    task::AsyncCommandRecorder,
    HasDevice,
};

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
}

impl<'w, 's, const Q: char> TransferCommands for RenderCommands<'w, 's, Q> where (): IsQueueCap<Q> {}
impl<const Q: char> TransferCommands for AsyncCommandRecorder<'_, Q> where (): IsQueueCap<Q> {}

pub struct BatchCopy<'a, T: TransferCommands> {
    recorder: &'a mut T,
    src: vk::Buffer,
    dst: vk::Buffer,
    copies: Vec<vk::BufferCopy>,
}
impl<'a, T: TransferCommands> BatchCopy<'a, T> {
    pub fn new(recorder: &'a mut T) -> Self {
        Self {
            recorder,
            src: vk::Buffer::null(),
            dst: vk::Buffer::null(),
            copies: Vec::new(),
        }
    }
    fn flush(&mut self) {
        if !self.copies.is_empty() {
            self.recorder.copy_buffer(self.src, self.dst, &self.copies);
            self.copies.clear();
        }
    }
}
impl<T: TransferCommands> HasDevice for BatchCopy<'_, T> {
    fn device(&self) -> &crate::Device {
        self.recorder.device()
    }
}
impl<T: TransferCommands> CommandRecorder for BatchCopy<'_, T> {
    const QUEUE_CAP: char = 't';

    fn cmd_buf(&mut self) -> vk::CommandBuffer {
        self.recorder.cmd_buf()
    }

    fn current_queue(&self) -> crate::QueueRef {
        self.recorder.current_queue()
    }

    fn semaphore_signal(&mut self) -> &mut impl super::SemaphoreSignalCommands {
        self.recorder.semaphore_signal()
    }
}
impl<T: TransferCommands> TransferCommands for BatchCopy<'_, T> {
    fn copy_buffer(&mut self, src: vk::Buffer, dst: vk::Buffer, regions: &[vk::BufferCopy]) {
        if self.src != src || self.dst != dst {
            self.flush();
            self.src = src;
            self.dst = dst;
        }
        if let Some(last_copy) = self.copies.last_mut()
            && let Some(current_copy) = regions.first()
            && (current_copy.src_offset == last_copy.src_offset + last_copy.size
                && current_copy.dst_offset == last_copy.dst_offset + last_copy.size)
        {
            // Optimization: extend the previous buffer copy instead
            last_copy.size += current_copy.size;
        } else {
            self.copies.extend_from_slice(regions);
        }
    }
}
impl<T: TransferCommands> Drop for BatchCopy<'_, T> {
    fn drop(&mut self) {
        self.flush();
    }
}
