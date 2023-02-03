use async_ash::{
    future::{GPUCommandFuture, Res},
    BufferLike, SharingMode,
};
use async_ash_core::{
    ash::{prelude::VkResult, vk},
    copy_buffer,
    macros::commands,
    HasDevice, PhysicalDeviceMemoryModel,
};
use vk_mem::Alloc;

use crate::Allocator;

pub struct ResidentBuffer {
    allocator: Allocator,
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    size: vk::DeviceSize,
}

impl ResidentBuffer {
    pub fn contents(&self) -> Option<&[u8]> {
        let info = self.allocator.inner().get_allocation_info(&self.allocation);
        if info.mapped_data.is_null() {
            None
        } else {
            unsafe {
                Some(std::slice::from_raw_parts(
                    info.mapped_data as *mut u8,
                    info.size as usize,
                ))
            }
        }
    }
}

impl BufferLike for ResidentBuffer {
    fn raw_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    fn size(&self) -> vk::DeviceSize {
        self.size
    }
}

impl Drop for ResidentBuffer {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .inner()
                .destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}

#[derive(Default)]
pub struct BufferCreateInfo<'a> {
    pub flags: vk::BufferCreateFlags,
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
    pub sharing_mode: SharingMode<'a>,
}

impl Allocator {
    pub fn create_resident_buffer(
        &self,
        buffer_info: &vk::BufferCreateInfo,
        create_info: &vk_mem::AllocationCreateInfo,
    ) -> VkResult<ResidentBuffer> {
        let staging_buffer = unsafe { self.inner().create_buffer(buffer_info, create_info)? };
        Ok(ResidentBuffer {
            allocator: self.clone(),
            buffer: staging_buffer.0,
            allocation: staging_buffer.1,
            size: buffer_info.size,
        })
    }

    /// Create uninitialized buffer only visible to the GPU.
    pub fn create_device_buffer_uninit(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> VkResult<ResidentBuffer> {
        let buffer_create_info = vk::BufferCreateInfo {
            size,
            usage,
            ..Default::default()
        };
        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };
        self.create_resident_buffer(&buffer_create_info, &alloc_info)
    }

    /// Create uninitialized, cached buffer on the host-side
    pub fn create_readback_buffer(&self, size: vk::DeviceSize) -> VkResult<ResidentBuffer> {
        let buffer_create_info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            ..Default::default()
        };
        let alloc_info = vk_mem::AllocationCreateInfo {
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM
                | vk_mem::AllocationCreateFlags::MAPPED,
            usage: vk_mem::MemoryUsage::AutoPreferHost,
            ..Default::default()
        };
        self.create_resident_buffer(&buffer_create_info, &alloc_info)
    }

    /// Crate a small device-local buffer with pre-populated data, only visible to the GPU.
    /// The data will be directly written to the buffer on ResizableBar, Bar, and UMA memory models.
    /// We will create a temporary staging buffer on all other cases.
    pub fn create_device_buffer_with_data(
        &self,
        data: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> VkResult<impl GPUCommandFuture<Output = Res<ResidentBuffer>>> {
        let create_info = vk::BufferCreateInfo {
            size: data.len() as u64,
            usage,
            ..Default::default()
        };

        let (staging_buffer, dst_buffer) = match self.device().physical_device().memory_model() {
            PhysicalDeviceMemoryModel::UMA
            | PhysicalDeviceMemoryModel::Bar
            | PhysicalDeviceMemoryModel::ResizableBar => {
                let buf = self.create_resident_buffer(
                    &create_info,
                    &vk_mem::AllocationCreateInfo {
                        flags: vk_mem::AllocationCreateFlags::MAPPED
                            | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                        usage: vk_mem::MemoryUsage::AutoPreferDevice,
                        required_flags: vk::MemoryPropertyFlags::empty(),
                        preferred_flags: vk::MemoryPropertyFlags::empty(),
                        memory_type_bits: 0,
                        user_data: 0,
                        priority: 0.0,
                    },
                )?;
                unsafe {
                    let info = self.inner().get_allocation_info(&buf.allocation);
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        info.mapped_data as *mut u8,
                        data.len(),
                    )
                }
                (None, buf)
            }
            PhysicalDeviceMemoryModel::Discrete => {
                let staging_buffer = self.create_resident_buffer(
                    &vk::BufferCreateInfo {
                        size: data.len() as u64,
                        usage: vk::BufferUsageFlags::TRANSFER_SRC,
                        ..Default::default()
                    },
                    &vk_mem::AllocationCreateInfo {
                        flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                        usage: vk_mem::MemoryUsage::AutoPreferHost,
                        ..Default::default()
                    },
                )?;
                unsafe {
                    let info = self.inner().get_allocation_info(&staging_buffer.allocation);
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        info.mapped_data as *mut u8,
                        data.len(),
                    )
                }
                let dst_buffer = self.create_resident_buffer(
                    &create_info,
                    &vk_mem::AllocationCreateInfo {
                        flags: vk_mem::AllocationCreateFlags::empty(),
                        usage: vk_mem::MemoryUsage::AutoPreferDevice,
                        ..Default::default()
                    },
                )?;
                (Some(staging_buffer), dst_buffer)
            }
        };

        Ok(commands! {
            let mut dst_buffer = import!(dst_buffer);
            if let Some(staging_buffer) = staging_buffer {
                let staging_buffer = import!(staging_buffer);
                copy_buffer(&mut dst_buffer, &staging_buffer).await;
            }
            dst_buffer
        })
    }
}
