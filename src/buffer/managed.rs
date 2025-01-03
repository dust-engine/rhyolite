use std::{
    ops::{Deref, DerefMut, Range, RangeBounds},
    ptr::NonNull,
};

use ash::{prelude::VkResult, vk};

use crate::{buffer, commands::TransferCommands, device, Allocator, HasDevice};

use super::BufferLike;

enum ManagedBufferMode {
    Syncronized {
        device_buffer: vk::Buffer,
        device_allocation: vk_mem::Allocation,
        host_buffer: vk::Buffer,
        host_allocation: vk_mem::Allocation,
        flushed_ranges: Vec<vk::BufferCopy>,
        invalidated_ranges: Vec<vk::BufferCopy>,
    },
    DirectWrite {
        device_buffer: vk::Buffer,
        device_allocation: vk_mem::Allocation,
        device_ptr: NonNull<[u8]>,
        host_data: Box<[u8]>,
    },
    Shared {
        buffer: vk::Buffer,
        allocation: vk_mem::Allocation,
    },
}

/// A buffer visible to the GPU and the CPU at the same time.
pub struct ManagedBuffer {
    allocator: Allocator,
    mode: ManagedBufferMode,
    host_coherent: bool,
    ptr: NonNull<[u8]>,
    device_address: vk::DeviceAddress,
}
impl Drop for ManagedBuffer {
    fn drop(&mut self) {
        unsafe {
            match &mut self.mode {
                ManagedBufferMode::Syncronized {
                    device_buffer,
                    device_allocation,
                    host_buffer,
                    host_allocation,
                    ..
                } => {
                    self.allocator
                        .destroy_buffer(*device_buffer, device_allocation);
                    self.allocator.destroy_buffer(*host_buffer, host_allocation);
                }
                ManagedBufferMode::Shared { buffer, allocation }
                | ManagedBufferMode::DirectWrite {
                    device_buffer: buffer,
                    device_allocation: allocation,
                    ..
                } => {
                    self.allocator.destroy_buffer(*buffer, allocation);
                }
            }
        }
    }
}
impl Deref for ManagedBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}
impl DerefMut for ManagedBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ptr.as_mut() }
    }
}
impl BufferLike for ManagedBuffer {
    fn raw_buffer(&self) -> vk::Buffer {
        match &self.mode {
            ManagedBufferMode::Syncronized {
                device_buffer: buffer,
                ..
            }
            | ManagedBufferMode::DirectWrite {
                device_buffer: buffer,
                ..
            }
            | ManagedBufferMode::Shared { buffer, .. } => *buffer,
        }
    }
    fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }
}

impl ManagedBuffer {
    pub fn new(allocator: Allocator, size: u64, usage: vk::BufferUsageFlags) -> VkResult<Self> {
        use vk_mem::Alloc;
        unsafe {
            use crate::PhysicalDeviceMemoryModel::*;
            let memory_model = allocator.physical_device().properties().memory_model;
            match memory_model {
                ReBar => {
                    let (device_buffer, device_allocation) = allocator.create_buffer(
                        &vk::BufferCreateInfo {
                            size,
                            usage,
                            ..Default::default()
                        },
                        &vk_mem::AllocationCreateInfo {
                            usage: vk_mem::MemoryUsage::AutoPreferDevice,
                            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
                                | vk::MemoryPropertyFlags::HOST_VISIBLE,
                            flags: vk_mem::AllocationCreateFlags::MAPPED | vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
                            preferred_flags: vk::MemoryPropertyFlags::HOST_COHERENT,
                            ..Default::default()
                        },
                    )?;
                    let allocation_info = allocator.get_allocation_info(&device_allocation);
                    let host_coherent = allocator.physical_device().properties().memory_types()
                        [allocation_info.memory_type as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
                    let mut host_data = vec![0; size as usize].into_boxed_slice();
                    Ok(Self {
                        device_address: if usage
                            .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        {
                            allocator.device().get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo {
                                    buffer: device_buffer,
                                    ..Default::default()
                                },
                            )
                        } else {
                            0
                        },
                        ptr: NonNull::new(std::ptr::from_raw_parts_mut(
                            host_data.as_mut_ptr(),
                            size as usize,
                        ))
                        .unwrap(),
                        mode: ManagedBufferMode::DirectWrite {
                            device_buffer,
                            device_allocation,
                            device_ptr: NonNull::new(std::ptr::from_raw_parts_mut(
                                allocation_info.mapped_data as *mut u8,
                                size as usize,
                            ))
                            .unwrap(),
                            host_data,
                        },
                        allocator,
                        host_coherent,
                    })
                }
                Discrete | Bar => {
                    let (device_buffer, device_allocation) = allocator.create_buffer(
                        &vk::BufferCreateInfo {
                            size,
                            usage: usage | vk::BufferUsageFlags::TRANSFER_DST,
                            ..Default::default()
                        },
                        &vk_mem::AllocationCreateInfo {
                            usage: vk_mem::MemoryUsage::AutoPreferDevice,
                            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                            ..Default::default()
                        },
                    )?;

                    let (host_buffer, host_allocation) = allocator.create_buffer(
                        &vk::BufferCreateInfo {
                            size,
                            usage: vk::BufferUsageFlags::TRANSFER_SRC,
                            ..Default::default()
                        },
                        &vk_mem::AllocationCreateInfo {
                            usage: vk_mem::MemoryUsage::AutoPreferHost,
                            required_flags: vk::MemoryPropertyFlags::HOST_CACHED
                                | vk::MemoryPropertyFlags::HOST_VISIBLE,
                            flags: vk_mem::AllocationCreateFlags::MAPPED | vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
                            ..Default::default()
                        },
                    )?;
                    let allocation_info = allocator.get_allocation_info(&host_allocation);
                    let host_coherent = allocator.physical_device().properties().memory_types()
                        [allocation_info.memory_type as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
                    Ok(Self {
                        device_address: if usage
                            .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        {
                            allocator.device().get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo {
                                    buffer: device_buffer,
                                    ..Default::default()
                                },
                            )
                        } else {
                            0
                        },
                        ptr: NonNull::new(std::ptr::from_raw_parts_mut(
                            allocation_info.mapped_data as *mut u8,
                            size as usize,
                        ))
                        .unwrap(),
                        mode: ManagedBufferMode::Syncronized {
                            device_buffer,
                            device_allocation,
                            host_buffer,
                            host_allocation,
                            flushed_ranges: Vec::new(),
                            invalidated_ranges: Vec::new(),
                        },
                        host_coherent,
                        allocator,
                    })
                }
                Unified | BiasedUnified => {
                    let (buffer, allocation) = allocator.create_buffer(
                        &vk::BufferCreateInfo {
                            size,
                            usage: usage,
                            ..Default::default()
                        },
                        &vk_mem::AllocationCreateInfo {
                            usage: vk_mem::MemoryUsage::AutoPreferHost,
                            required_flags: vk::MemoryPropertyFlags::HOST_CACHED
                                | vk::MemoryPropertyFlags::HOST_VISIBLE,
                            flags: vk_mem::AllocationCreateFlags::MAPPED | vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
                            ..Default::default()
                        },
                    )?;
                    let allocation_info = allocator.get_allocation_info(&allocation);
                    let host_coherent = allocator.physical_device().properties().memory_types()
                        [allocation_info.memory_type as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
                    Ok(Self {
                        device_address: if usage
                            .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        {
                            allocator.device().get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo {
                                    buffer,
                                    ..Default::default()
                                },
                            )
                        } else {
                            0
                        },
                        ptr: NonNull::new(std::ptr::from_raw_parts_mut(
                            allocation_info.mapped_data as *mut u8,
                            size as usize,
                        ))
                        .unwrap(),
                        mode: ManagedBufferMode::Shared { buffer, allocation },
                        host_coherent,
                        allocator,
                    })
                }
            }
        }
    }
    pub fn len(&self) -> usize {
        self.ptr.len()
    }
    pub fn flush(&mut self, range: impl RangeBounds<u64>) {
        let start = match range.start_bound() {
            std::ops::Bound::Included(&start) => start,
            std::ops::Bound::Excluded(&start) => start + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let size = match range.end_bound() {
            std::ops::Bound::Included(&end) => end + 1 - start,
            std::ops::Bound::Excluded(&end) => end - start,
            std::ops::Bound::Unbounded => self.len() as u64 - start,
        };

        match &mut self.mode {
            ManagedBufferMode::Syncronized { flushed_ranges, .. } => {
                flushed_ranges.push(vk::BufferCopy {
                    src_offset: start,
                    dst_offset: start,
                    size,
                });
            }
            ManagedBufferMode::DirectWrite {
                device_ptr,
                host_data,
                device_allocation,
                ..
            } => unsafe {
                let range = start as usize..(start + size) as usize;
                device_ptr.as_mut()[range.clone()].copy_from_slice(&host_data[range]);
                if !self.host_coherent {
                    self.allocator
                        .flush_allocation(device_allocation, start, size)
                        .unwrap();
                }
            },
            ManagedBufferMode::Shared { allocation, .. } => {
                if self.host_coherent {
                    return;
                }
                self.allocator
                    .flush_allocation(allocation, start, size)
                    .unwrap();
            }
        }
    }
    pub fn invalidate(&mut self, range: impl RangeBounds<u64>) {
        let start = match range.start_bound() {
            std::ops::Bound::Included(&start) => start,
            std::ops::Bound::Excluded(&start) => start + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let size = match range.end_bound() {
            std::ops::Bound::Included(&end) => end + 1 - start,
            std::ops::Bound::Excluded(&end) => end - start,
            std::ops::Bound::Unbounded => vk::WHOLE_SIZE,
        };

        match &mut self.mode {
            ManagedBufferMode::Syncronized { flushed_ranges, .. } => {
                flushed_ranges.push(vk::BufferCopy {
                    src_offset: start,
                    dst_offset: start,
                    size,
                });
            }
            ManagedBufferMode::Shared { allocation, .. } => {
                if self.host_coherent {
                    return;
                }
                self.allocator
                    .invalidate_allocation(allocation, start, size)
                    .unwrap();
            }
            ManagedBufferMode::DirectWrite { .. } => {
                todo!()
            }
        }
    }

    /// Required to be called on Discrete or Bar architectures
    pub fn sync(&mut self, command_encoder: &mut impl TransferCommands) {
        match &mut self.mode {
            ManagedBufferMode::Syncronized {
                flushed_ranges,
                invalidated_ranges,
                device_buffer,
                host_buffer,
                ..
            } => {
                if !flushed_ranges.is_empty() {
                    command_encoder.copy_buffer(*host_buffer, *device_buffer, &flushed_ranges);
                }
                if !invalidated_ranges.is_empty() {
                    command_encoder.copy_buffer(*device_buffer, *host_buffer, &invalidated_ranges);
                }
                flushed_ranges.clear();
                invalidated_ranges.clear();
            }
            _ => {}
        }
    }
}
