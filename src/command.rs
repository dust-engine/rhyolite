use std::sync::Arc;

use ash::{prelude::VkResult, vk};

use crate::{semaphore::TimelineSemaphore, Device, GPUFutureContext, HasDevice, Queue, QueueSelector};

pub struct CommandPool {
    device: Device,
    raw: vk::CommandPool,
    queue_family_index: u32,
    flags: vk::CommandPoolCreateFlags,
    num_outstanding_buffers: u32,
}
impl Drop for CommandPool {
    fn drop(&mut self) {
        assert!(self.num_outstanding_buffers == 0, "Cannot drop the command pool because there are still outstanding command buffers allocated from this pool.");
        unsafe  {
            self.device.destroy_command_pool(self.raw, None);
        }
    }
}

pub struct CommandEncoder<'a> {
    pool: &'a mut CommandPool,
    pub(crate) command_buffer: CommandBuffer,
}
pub struct CommandBuffer {
    raw: vk::CommandBuffer,
    flags: vk::CommandBufferUsageFlags,
    pub(crate) timeline_semaphore: Arc<TimelineSemaphore>,
    pub(crate) wait_value: u64,
    pub(crate) future_ctx: GPUFutureContext,
    drop_guard: CommandBufferDropGuard,
}
struct CommandBufferDropGuard;
impl Drop for CommandBufferDropGuard {
    fn drop(&mut self) {
        panic!("CommandBuffer must be returned to the CommandPool!");
    }
}


impl CommandPool {
    pub fn new(device: Device, queue_family_index: u32, flags: vk::CommandPoolCreateFlags) -> VkResult<Self> {
        unsafe {
            let raw = device.create_command_pool(&vk::CommandPoolCreateInfo {
                queue_family_index,
                ..Default::default()
            }, None)?;
            Ok(Self {
                raw,
                device,
                num_outstanding_buffers: 0,
                queue_family_index,
                flags,
            })
        }
    }
    pub fn start_encoding(&mut self, on_timeline: &Timeline, flags: vk::CommandBufferUsageFlags) -> VkResult<CommandEncoder> {
        self.num_outstanding_buffers += 1;
        let mut raw = vk::CommandBuffer::null();
        unsafe  {
            (self.device.fp_v1_0().allocate_command_buffers)(self.device.handle(), &vk::CommandBufferAllocateInfo {
                command_pool: self.raw,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            }, &mut raw).result()?;
            self.device.begin_command_buffer(raw, &vk::CommandBufferBeginInfo {
                flags,
                ..Default::default()
            })?;
        }
        let command_buffer = CommandBuffer {
            raw,
            flags,
            timeline_semaphore: on_timeline.semaphore.clone(),
            wait_value: on_timeline.wait_value,
            future_ctx: GPUFutureContext::new(self.device.clone(), raw),
            drop_guard: CommandBufferDropGuard,
        };
        Ok(CommandEncoder {
            pool: self,
            command_buffer,
        })
    }
    pub fn restart_encoding(&mut self, mut command_buffer: CommandBuffer, release_resources: bool) -> VkResult<CommandEncoder> {
        assert!(self.flags.contains(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER));
        let flags = if release_resources {
            vk::CommandBufferResetFlags::RELEASE_RESOURCES
        } else {
            vk::CommandBufferResetFlags::empty()
        };
        unsafe {
            self.device.reset_command_buffer(command_buffer.raw, flags)?;
        }
        command_buffer.future_ctx.clear();
        let encoder = CommandEncoder {
            pool: self,
            command_buffer,
        };
        Ok(encoder)
    }
    pub fn recycle(&mut self, buffer: CommandBuffer) {
        unsafe {
            self.device.free_command_buffers(self.raw, &[buffer.raw]);
            self.num_outstanding_buffers -= 1;

            std::mem::forget(buffer.drop_guard);
        }
    }
    pub fn recycle_many(&mut self, buffers: impl Iterator<Item = CommandBuffer>) {
        let handles: Vec<_> = buffers.map(|x| {
            let buf = x.raw;
            std::mem::forget(x.drop_guard);
            buf
        }).collect();
        unsafe {
            self.device.free_command_buffers(self.raw, &handles);
            self.num_outstanding_buffers -= handles.len() as u32;
        }
    }
}


impl<'a> CommandEncoder<'a> {
    pub fn end(self) -> VkResult<CommandBuffer> {
        unsafe {
            self.pool.device.end_command_buffer(self.command_buffer.raw)?;
        }
        Ok(self.command_buffer)
    }
    pub fn reset(&mut self, release_resources: bool) -> VkResult<()> {
        assert!(self.pool.flags.contains(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER));
        let flags = if release_resources {
            vk::CommandBufferResetFlags::RELEASE_RESOURCES
        } else {
            vk::CommandBufferResetFlags::empty()
        };
        unsafe {
            self.pool.device.reset_command_buffer(self.command_buffer.raw, flags)?;
        }
        Ok(())
    }
}


// Each queue system will have one of this.
// It gets incremented during queue submit.
pub struct Timeline {
    semaphore: Arc<TimelineSemaphore>,
    wait_value: u64,
}

impl<'a, T: QueueSelector> Queue<'a, T> {
    pub fn submit_one(&mut self, command_buffer: CommandBuffer) -> VkResult<()> {
        // TODO: emit syncronization barrier for command buffer future ctx.
        unsafe {
            self.device.queue_submit2(self.queue, &[
                vk::SubmitInfo2 {
                    ..Default::default()
                }.command_buffer_infos(&[
                    vk::CommandBufferSubmitInfo {
                        command_buffer: command_buffer.raw,
                        ..Default::default()
                    }
                ])
                .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo {
                    semaphore: command_buffer.timeline_semaphore.raw(),
                    value: command_buffer.wait_value + 1,
                    // We signal on ALL_COMMANDS because
                    // 1. Timeline semaphore.
                    // 2. Most impl probably cannot take advantage of any other flags.
                    stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                    ..Default::default()
                }])
            ], vk::Fence::null())?;
        }
        Ok(())
    }
}
