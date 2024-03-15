use ash::{prelude::VkResult, vk};

use crate::{Device, HasDevice};

pub struct CommandPool {
    device: Device,
    raw: vk::CommandPool,
    queue_family_index: u32,
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
        Ok(Self {
            device,
            raw,
            queue_family_index,
        })
    }
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
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

/// Command pool that must be reset as a whole.
/// It maintains a list of command buffers allocated from the pool to reduce the number of calls
/// to `vkAllocateCommandBuffers`.
pub struct ManagedCommandPool {
    /// The command pool for managed command buffers.
    pool: CommandPool,
    /// All command buffers allocated from `pool`
    allocated_command_buffers: Vec<vk::CommandBuffer>,
    /// allocated_command_buffers[0..command_buffer_allocation_index] are used in the current frame.
    command_buffer_allocation_index: usize,
}
impl ManagedCommandPool {
    pub fn new(device: Device, queue_family_index: u32) -> VkResult<Self> {
        let pool = CommandPool::new(device, queue_family_index)?;
        Ok(Self {
            pool,
            allocated_command_buffers: Vec::new(),
            command_buffer_allocation_index: 0,
        })
    }
    pub fn queue_family_index(&self) -> u32 {
        self.pool.queue_family_index()
    }
    /// Returns new command buffer.
    pub fn allocate(&mut self) -> vk::CommandBuffer {
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
    pub fn reset(&mut self) {
        self.pool.reset().unwrap();
        self.command_buffer_allocation_index = 0;
    }
}
impl HasDevice for ManagedCommandPool {
    fn device(&self) -> &Device {
        self.pool.device()
    }
}
