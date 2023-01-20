use std::{sync::Arc, ops::{Deref, DerefMut}, marker::PhantomData, cell::Cell};
use crate::{Device, future::{use_state, use_per_frame_state, PerFrameState, PerFrameContainer, use_cached_state}, device, queue, HasDevice};
use ash::vk;
use std::cell::UnsafeCell;

/// An unsafe command pool. Command buffer lifecycles are unmanaged.
pub struct UnsafeCommandPool {
    device: Arc<Device>,
    command_pool: vk::CommandPool,
}
unsafe impl Send for UnsafeCommandPool {}
impl !Sync for UnsafeCommandPool {}



impl UnsafeCommandPool {
    pub fn new(device: Arc<Device>, queue_family_index: u32, flags: vk::CommandPoolCreateFlags) -> Self {
        let command_pool = unsafe {
            device.create_command_pool(&vk::CommandPoolCreateInfo {
                queue_family_index,
                flags,
                ..Default::default()
            }, None)
        }.unwrap();
        Self {
            device,
            command_pool,
        }
    }
    /// Marked unsafe because allocated command buffers won't be recycled automatically.
    pub unsafe fn allocate_n<const N: usize>(&self, secondary: bool) -> [vk::CommandBuffer; N] {
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
    pub unsafe fn allocate_one(&self, secondary: bool) -> vk::CommandBuffer {
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

pub struct SharedCommandPool {
    pool: UnsafeCommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    indice: usize,
}
impl HasDevice for SharedCommandPool {
    fn device(&self) -> &Arc<Device> {
        &self.pool.device
    }
}
impl SharedCommandPool {
    pub fn new(device: Arc<Device>, queue_family_index: u32) -> Self {
        let pool = UnsafeCommandPool::new(device, queue_family_index, vk::CommandPoolCreateFlags::TRANSIENT);
        Self {
            pool,
            command_buffers: Vec::new(),
            indice: 0
        }
    }
    pub fn allocate_one(&mut self) -> vk::CommandBuffer {
        if self.indice >= self.command_buffers.len() {
            let buffer = unsafe {
                self.pool.allocate_one(false)
            };
            self.command_buffers.push(buffer);
            self.indice += 1;
            buffer
        } else {
            let raw_buffer = self.command_buffers[self.indice];
            self.indice += 1;
            raw_buffer
        }
    }
}


pub fn use_command_pool<'recycle>(
    this: &'recycle mut Option<UnsafeCommandPool>,
    device: &Arc<Device>,
    queue_family_index: u32,
) -> &'recycle mut UnsafeCommandPool {
    let pool = this.get_or_insert_with(|| {
        UnsafeCommandPool::new(device.clone(), queue_family_index, vk::CommandPoolCreateFlags::TRANSIENT)
    });
    pool.reset(false);
    pool
    
}


pub struct CommandBuffer<'a> {
    buffer: vk::CommandBuffer,
    _marker: PhantomData<&'a ()>
}
impl<'a> CommandBuffer<'a> {
    pub(crate) unsafe fn new(buffer: vk::CommandBuffer) -> Self {
        Self {
            buffer,
            _marker: PhantomData
        }
    }
}
impl<'a> CommandBufferLike for CommandBuffer<'a> {
    fn raw_command_buffer(&self) -> vk::CommandBuffer {
        self.buffer
    }
}
pub fn use_command_buffer<'recycle>(
    this: &'recycle mut Option<vk::CommandBuffer>,
    pool: &'recycle mut UnsafeCommandPool
) -> CommandBuffer<'recycle> {
    let buffer= *this.get_or_insert_with(|| unsafe {
        pool.allocate_one(false)
    });
    CommandBuffer { buffer, _marker: PhantomData }
}


pub trait CommandBufferLike {
    fn raw_command_buffer(&self) -> vk::CommandBuffer;
}


// It will be safe as long as we don't destroy command buffers.
// CommandPool: Send, but not Sync. RecycledState, reset on fetch.
// CommandBuffer: Do nothing on drop. Also gets preserved in RecycledState. Upon allocation, do nothing.