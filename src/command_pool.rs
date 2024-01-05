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

// Command buffer strategies:
// - Primary command buffers
//   - Each stage can have a number of single threaded systems and multi threaded systems.
//     If prev stage and next stage are both single threaded, prev commands, pipeline barrier, and next commands are recorded to the same buffer.
//     If prev stage is multi threaded but next stage is single threaded, on prev stage end, collect all buffers. Pipeline barrier recorded to
//     last buffer in this. next stage commands recorded to this last buffer.
//     If prev stage is single threaded but next stage is multi threaded, on prev stage end, record pipeliene barrier to prev stage buffer.
//     One next stage system gets this command buffer. Others have to allocate new.
// - Secondary command buffers
//   - Parallel recording
//   - Single-threaded recording
