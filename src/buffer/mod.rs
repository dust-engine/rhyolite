pub mod staging;

use std::{
    alloc::Layout,
    mem::MaybeUninit,
    ops::{Deref, DerefMut, Index, IndexMut, RangeBounds},
};

use ash::{
    prelude::VkResult,
    vk::{self, Handle},
};

use crate::{utils::SharingMode, Allocator, HasDevice, PhysicalDeviceMemoryModel};
use vk_mem::Alloc;

pub trait BufferLike {
    fn raw_buffer(&self) -> vk::Buffer;
    fn offset(&self) -> vk::DeviceSize {
        0
    }
    fn size(&self) -> vk::DeviceSize {
        vk::WHOLE_SIZE
    }
    fn device_address(&self) -> vk::DeviceAddress {
        panic!()
    }
}
/// A buffer fully bound to a memory allocation.
pub struct Buffer {
    allocator: Allocator,
    allocation: vk_mem::Allocation,
    buffer: vk::Buffer,
    size: vk::DeviceSize,
}
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}
impl crate::utils::AsVkHandle for Buffer {
    fn vk_handle(&self) -> Self::Handle {
        self.buffer
    }
    type Handle = vk::Buffer;
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}
impl Buffer {
    pub fn from_raw(
        allocator: Allocator,
        buffer: vk::Buffer,
        allocation: vk_mem::Allocation,
    ) -> Self {
        let size = allocator.get_allocation_info(&allocation).size;
        Self {
            allocator,
            buffer,
            allocation,
            size,
        }
    }
    /// Create a new buffer with DEVICE_LOCAL, HOST_VISIBLE memory.
    pub fn new_dynamic(
        allocator: Allocator,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> VkResult<Self> {
        unsafe {
            let (buffer, allocation) = allocator.create_buffer_with_alignment(
                &vk::BufferCreateInfo {
                    size,
                    usage,
                    ..Default::default()
                },
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferDevice,
                    flags: vk_mem::AllocationCreateFlags::MAPPED
                        | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                alignment,
            )?;
            Ok(Self {
                allocator,
                buffer,
                allocation,
                size,
            })
        }
    }
    /// Create a new buffer with DEVICE_LOCAL memory.
    pub fn new_staging(
        allocator: Allocator,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> VkResult<Self> {
        unsafe {
            let (buffer, allocation) = allocator.create_buffer_with_alignment(
                &vk::BufferCreateInfo {
                    size,
                    usage,
                    ..Default::default()
                },
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferHost,
                    flags: vk_mem::AllocationCreateFlags::MAPPED
                        | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                alignment,
            )?;
            Ok(Self {
                allocator,
                buffer,
                allocation,
                size,
            })
        }
    }
    /// Create a new buffer with DEVICE_LOCAL memory.
    pub fn new_resource(
        allocator: Allocator,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> VkResult<Self> {
        unsafe {
            let (buffer, allocation) = allocator.create_buffer_with_alignment(
                &vk::BufferCreateInfo {
                    size,
                    usage,
                    ..Default::default()
                },
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferDevice,
                    required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    preferred_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    ..Default::default()
                },
                alignment,
            )?;
            Ok(Self {
                allocator,
                buffer,
                allocation,
                size,
            })
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.allocator
                    .get_allocation_info(&self.allocation)
                    .mapped_data as *const u8,
                self.size as usize,
            )
        }
    }
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.allocator
                    .get_allocation_info(&self.allocation)
                    .mapped_data as *mut u8,
                self.size as usize,
            )
        }
    }
}
impl HasDevice for Buffer {
    fn device(&self) -> &crate::Device {
        self.allocator.device()
    }
}
impl BufferLike for Buffer {
    fn raw_buffer(&self) -> vk::Buffer {
        self.buffer
    }
    fn size(&self) -> vk::DeviceSize {
        self.size
    }
    fn device_address(&self) -> vk::DeviceAddress {
        unsafe {
            self.allocator
                .device()
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo {
                    buffer: self.buffer,
                    ..Default::default()
                })
        }
    }
}

pub struct BufferArray<T> {
    allocator: Allocator,
    allocation: Option<vk_mem::Allocation>,
    buffer: vk::Buffer,
    len: usize,
    ptr: *mut MaybeUninit<T>,
    _marker: std::marker::PhantomData<T>,
    usage: vk::BufferUsageFlags,
    sharing_mode: SharingMode<Vec<u32>>,
    allocation_info: vk_mem::AllocationCreateInfo,
}
unsafe impl<T: Send> Send for BufferArray<T> {}
unsafe impl<T: Sync> Sync for BufferArray<T> {}
impl<T> Drop for BufferArray<T> {
    fn drop(&mut self) {
        unsafe {
            if let Some(allocation) = &mut self.allocation {
                self.allocator.destroy_buffer(self.buffer, allocation);
            }
        }
    }
}
impl<T> HasDevice for BufferArray<T> {
    fn device(&self) -> &crate::Device {
        self.allocator.device()
    }
}
impl<T> BufferLike for BufferArray<T> {
    fn raw_buffer(&self) -> vk::Buffer {
        self.buffer
    }
    fn device_address(&self) -> vk::DeviceAddress {
        unsafe {
            self.allocator
                .device()
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo {
                    buffer: self.buffer,
                    ..Default::default()
                })
        }
    }
    fn size(&self) -> vk::DeviceSize {
        Layout::new::<T>()
            .repeat(self.len)
            .unwrap()
            .0
            .pad_to_align()
            .size() as vk::DeviceSize
    }
}

impl<T> BufferArray<T> {
    pub fn len(&self) -> usize {
        self.len
    }
    /// Ensure that the array is at least `new_len` elements long.
    pub fn realloc(&mut self, new_len: usize) -> VkResult<Option<Buffer>> {
        let new_capacity = new_len.next_power_of_two().max(8);
        if new_capacity <= self.len {
            return Ok(None);
        }
        unsafe {
            let old_buffer = self
                .allocation
                .take()
                .map(|a| Buffer::from_raw(self.allocator.clone(), self.buffer, a));
            let (buffer, allocation) = self.allocator.create_buffer(
                &vk::BufferCreateInfo {
                    size: Layout::new::<T>()
                        .repeat(new_capacity)
                        .unwrap()
                        .0
                        .pad_to_align()
                        .size() as vk::DeviceSize,
                    usage: self.usage,
                    sharing_mode: self.sharing_mode.as_raw(),
                    queue_family_index_count: self.sharing_mode.queue_family_indices().len() as u32,
                    p_queue_family_indices: self.sharing_mode.queue_family_indices().as_ptr(),
                    ..Default::default()
                },
                &self.allocation_info,
            )?;

            let info = self.allocator.get_allocation_info(&allocation);
            self.ptr = info.mapped_data as *mut MaybeUninit<T>;
            self.buffer = buffer;
            self.allocation = Some(allocation);
            self.len = new_capacity;
            Ok(old_buffer)
        }
    }
    /// Create a new HOST_VISIBLE upload buffer for sequential write.
    /// On integrated GPUs and GPUs with SAM, the upload buffer may be used directly by the device.
    /// On discrete GPUs, the upload buffer serves as the staging buffer. The user will have to create backing
    /// DEVICE_LOCAL buffer and schedule transfer.
    pub fn new_upload(allocator: Allocator, mut usage: vk::BufferUsageFlags) -> Self {
        let memory_model = allocator
            .device()
            .physical_device()
            .properties()
            .memory_model;
        let memory_usage = if matches!(memory_model, PhysicalDeviceMemoryModel::ReBar) {
            vk_mem::MemoryUsage::AutoPreferDevice
        } else {
            vk_mem::MemoryUsage::AutoPreferHost
        };
        if memory_model.storage_buffer_should_use_staging() {
            usage |= vk::BufferUsageFlags::TRANSFER_SRC;
        };
        Self {
            allocator,
            allocation: None,
            buffer: vk::Buffer::null(),
            ptr: std::ptr::null_mut(),
            _marker: std::marker::PhantomData,
            usage,
            sharing_mode: SharingMode::Exclusive,
            allocation_info: vk_mem::AllocationCreateInfo {
                usage: memory_usage,
                flags: vk_mem::AllocationCreateFlags::MAPPED
                    | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            len: 0,
        }
    }

    /// Create GPU-owned buffer.
    pub fn new_resource(allocator: Allocator, usage: vk::BufferUsageFlags) -> Self {
        let res = Self {
            allocator,
            allocation: None,
            buffer: vk::Buffer::null(),
            ptr: std::ptr::null_mut(),
            _marker: std::marker::PhantomData,
            usage,
            sharing_mode: SharingMode::Exclusive,
            allocation_info: vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                flags: vk_mem::AllocationCreateFlags::empty(),
                required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                preferred_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                ..Default::default()
            },
            len: 0,
        };
        res
    }

    pub fn flush(&mut self, range: impl RangeBounds<usize>) -> VkResult<()> {
        if let Some(allocation) = &self.allocation {
            let item_size = Layout::new::<T>().pad_to_align().size();
            let start = match range.start_bound() {
                std::ops::Bound::Included(&start) => start * item_size,
                std::ops::Bound::Excluded(&start) => (start + 1) * item_size,
                std::ops::Bound::Unbounded => 0,
            };
            let len = match range.end_bound() {
                std::ops::Bound::Included(&end) => (end + 1) * item_size - start,
                std::ops::Bound::Excluded(&end) => end * item_size - start,
                std::ops::Bound::Unbounded => vk::WHOLE_SIZE as usize,
            };
            return self
                .allocator
                .flush_allocation(allocation, start as u64, len as u64);
        } else {
            return Ok(());
        }
    }
}

impl<T> Index<usize> for BufferArray<T> {
    type Output = MaybeUninit<T>;

    fn index(&self, index: usize) -> &Self::Output {
        if self
            .allocation_info
            .flags
            .contains(vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE)
        {
            tracing::warn!("Reading from a HOST_VISIBLE buffer that is mapped for sequential write is likely inefficient");
        }
        assert!(index < self.len);
        unsafe {
            let ptr = self.ptr.add(index as usize);
            &*ptr
        }
    }
}
impl<T> IndexMut<usize> for BufferArray<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.len);
        unsafe {
            let ptr = self.ptr.add(index as usize);
            &mut *ptr
        }
    }
}
impl<T> Deref for BufferArray<T> {
    type Target = [MaybeUninit<T>];

    fn deref(&self) -> &Self::Target {
        if self
            .allocation_info
            .flags
            .contains(vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE)
        {
            tracing::warn!("Reading from a HOST_VISIBLE buffer that is mapped for sequential write is likely inefficient");
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len as usize) }
    }
}
impl<T> DerefMut for BufferArray<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len as usize) }
    }
}
