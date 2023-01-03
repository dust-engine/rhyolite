use std::sync::Arc;
use crate::Device;
use ash::vk;

/// An unsafe command pool. Command buffer lifecycles are unmanaged.
pub struct UnsafeCommandPool {
    device: Arc<Device>,
    command_pool: vk::CommandPool,
}
// command pool usage pattern
// Give out references to command buffers.

// transient pool. allocates one-time-use command buffers. free them individually.
// transient pool. allocates one-time-use command buffers. individual reset.
// transient pool. allocates one-time-use command buffers. pool reset.

impl UnsafeCommandPool {
    pub fn new(device: Arc<Device>, queue_family_index: u32, flags: vk::CommandPoolCreateFlags) {
        let command_pool = unsafe {
            device.create_command_pool(&vk::CommandPoolCreateInfo {
                queue_family_index,
                flags,
                ..Default::default()
            }, None);
        };
    }
    /// Marked unsafe because allocated command buffers won't be recycled automatically.
    pub unsafe fn allocate_n<const N: usize>(&mut self, secondary: bool) -> [vk::CommandBuffer; N] {
        unsafe {
            let mut command_buffer = [vk::CommandBuffer::null(); N];
            (self.device.fp_v1_0().allocate_command_buffers)(
                self.device.handle(),
                &vk::CommandBufferAllocateInfo {
                    command_pool: self.command_pool,
                    level: if secondary { vk::CommandBufferLevel::SECONDARY } else { vk::CommandBufferLevel::PRIMARY },
                    command_buffer_count: N as u32,
                    ..Default::default()
                },
                command_buffer.as_mut_ptr()
            ).result().unwrap();
            command_buffer
        }
    }
    /// Marked unsafe because allocated command buffers won't be recycled automatically.
    pub unsafe fn allocate_one(&mut self, secondary: bool) -> vk::CommandBuffer {
        let command_buffer: [vk::CommandBuffer; 1] = self.allocate_n(secondary);
        command_buffer[0]
    }
    pub fn reset(&mut self, release_resources: bool) {
        unsafe {
            self.device.reset_command_pool(self.command_pool, if release_resources {
                vk::CommandPoolResetFlags::RELEASE_RESOURCES
            } else {
                vk::CommandPoolResetFlags::empty()
            }).unwrap();
        }
    }
}

impl Drop for UnsafeCommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}
