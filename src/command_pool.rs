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
    /// The command pool for managed command buffers.
    pool: CommandPool,
    /// All command buffers allocated from `pool`
    allocated_command_buffers: Vec<vk::CommandBuffer>,
    /// allocated_command_buffers[0..command_buffer_allocation_index] are used in the current frame.
    command_buffer_allocation_index: usize,

    /// The currently recording command buffer. May be null.
    recording_command_buffer: vk::CommandBuffer,

    /// List of all recorded command buffers. May contain external command buffers.
    recorded_command_buffers: Vec<vk::CommandBuffer>,
}
impl RecordingCommandBuffer {
    pub fn new(device: Device, queue_family_index: u32) -> Self {
        let pool = CommandPool::new(device, queue_family_index).unwrap();
        Self {
            pool,
            allocated_command_buffers: Vec::new(),
            command_buffer_allocation_index: 0,
            recording_command_buffer: vk::CommandBuffer::null(),
            recorded_command_buffers: Vec::new(),
        }
    }
    /// Returns new managed command buffer.
    fn allocate_managed_command_buffer(&mut self) -> vk::CommandBuffer {
        if self.command_buffer_allocation_index < self.allocated_command_buffers.len() {
            let allocated = self.allocated_command_buffers[self.command_buffer_allocation_index];
            self.command_buffer_allocation_index += 1;
            allocated
        } else {
            let allocated = unsafe { self.pool.allocate() }.unwrap();
            self.allocated_command_buffers.push(allocated);
            self.command_buffer_allocation_index += 1;
            allocated
        }
    }
    /// Returns the managed command buffer currently being recorded, or allocates a new one if none is currently recording.
    pub fn record(&mut self) -> vk::CommandBuffer {
        if self.recording_command_buffer == vk::CommandBuffer::null() {
            self.recording_command_buffer = self.allocate_managed_command_buffer();
            unsafe {
                self.pool
                    .device
                    .begin_command_buffer(
                        self.recording_command_buffer,
                        &vk::CommandBufferBeginInfo {
                            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                            ..Default::default()
                        },
                    )
                    .unwrap();
            }
        }
        self.recording_command_buffer
    }
    /// Inserts externally managed command buffer
    pub fn insert(&mut self, external: vk::CommandBuffer) {
        if self.recording_command_buffer != vk::CommandBuffer::null() {
            unsafe {
                self.pool
                    .device
                    .end_command_buffer(self.recording_command_buffer)
                    .unwrap();
            }
            self.recorded_command_buffers
                .push(self.recording_command_buffer);
            self.recording_command_buffer = vk::CommandBuffer::null();
        }
        self.recorded_command_buffers.push(external);
    }
    pub fn take(&mut self) -> Vec<vk::CommandBuffer> {
        if self.recording_command_buffer != vk::CommandBuffer::null() {
            unsafe {
                self.pool
                    .device
                    .end_command_buffer(self.recording_command_buffer)
                    .unwrap();
            }
            self.recorded_command_buffers
                .push(self.recording_command_buffer);
            self.recording_command_buffer = vk::CommandBuffer::null();
        }
        std::mem::take(&mut self.recorded_command_buffers)
    }
    pub fn reset(&mut self) {
        assert!(self.recorded_command_buffers.is_empty());
        assert!(self.recording_command_buffer == vk::CommandBuffer::null());
        self.pool.reset().unwrap();
        self.command_buffer_allocation_index = 0;
    }
}
impl HasDevice for RecordingCommandBuffer {
    fn device(&self) -> &Device {
        self.pool.device()
    }
}
