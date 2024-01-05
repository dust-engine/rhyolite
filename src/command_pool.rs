use ash::{prelude::VkResult, vk};
use bevy_ecs::system::Resource;
use thread_local::ThreadLocal;

use crate::Device;

pub struct CommandPool {
    device: Device,
    raw: vk::CommandPool,
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

#[derive(Resource)]
pub struct CommandPools {
    pools: ThreadLocal<CommandPool>,
}
