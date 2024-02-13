use ash::{prelude::VkResult, vk};

use crate::{Device, HasDevice};

pub struct CommandPool {
    device: Device,
    raw: vk::CommandPool,
}
impl HasDevice for CommandPool {
    fn device(&self) -> &Device {
        &self.device
    }
}
impl CommandPool {
    pub fn new(device: Device, queue_family_index: u32) -> VkResult<Self> {
        let raw = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    flags: vk::CommandPoolCreateFlags::TRANSIENT,
                    queue_family_index,
                    ..Default::default()
                },
                None,
            )?
        };
        Ok(Self { device, raw })
    }
    pub unsafe fn allocate(&mut self) -> VkResult<vk::CommandBuffer> {
        let mut command_buffer = vk::CommandBuffer::null();
        unsafe {
            (self.device.fp_v1_0().allocate_command_buffers)(
                self.device.handle(),
                &vk::CommandBufferAllocateInfo {
                    command_pool: self.raw,
                    level: vk::CommandBufferLevel::PRIMARY,
                    command_buffer_count: 1,
                    ..Default::default()
                },
                &mut command_buffer,
            )
            .result()?;
        }
        Ok(command_buffer)
    }
    pub fn reset(&mut self) -> VkResult<()> {
        unsafe {
            self.device
                .reset_command_pool(self.raw, vk::CommandPoolResetFlags::empty())?;
        }
        Ok(())
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.raw, None);
        }
    }
}

pub struct RecordingCommandBuffer {
    pub(crate) pool: CommandPool,
    /// The currently recording command buffer. May be null.
    pub(crate) command_buffer: vk::CommandBuffer,
}
impl RecordingCommandBuffer {
    pub fn new(device: Device, queue_family_index: u32) -> Self {
        let pool = CommandPool::new(device, queue_family_index).unwrap();
        Self {
            pool,
            command_buffer: vk::CommandBuffer::null(),
        }
    }
    /// Returns the command buffer currently being recorded, or allocates a new one if none is currently recording.
    pub fn record(&mut self) -> vk::CommandBuffer {
        if self.command_buffer == vk::CommandBuffer::null() {
            self.command_buffer = unsafe { self.pool.allocate() }.unwrap();
            unsafe {
                self.pool
                    .device
                    .begin_command_buffer(
                        self.command_buffer,
                        &vk::CommandBufferBeginInfo {
                            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                            ..Default::default()
                        },
                    )
                    .unwrap();
            }
        }
        self.command_buffer
    }
    pub unsafe fn take(&mut self) -> vk::CommandBuffer {
        if self.command_buffer != vk::CommandBuffer::null() {
            self.pool
                .device
                .end_command_buffer(self.command_buffer)
                .unwrap();
        }
        std::mem::take(&mut self.command_buffer)
    }
}
impl HasDevice for RecordingCommandBuffer {
    fn device(&self) -> &Device {
        self.pool.device()
    }
}
