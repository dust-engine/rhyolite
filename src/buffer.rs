use std::{mem::MaybeUninit, ops::{Deref, DerefMut, Index, IndexMut}, ptr::NonNull, sync::Barrier};

use ash::{prelude::VkResult, vk};
use bevy_ecs::system::{ResMut, Resource};

use crate::{ecs::{queue_cap::IsQueueCap, Barriers, PerFrameMut, PerFrameResource, RenderCommands, RenderRes}, utils::SharingMode, Access, Allocator, HasDevice, PhysicalDeviceMemoryModel};
use vk_mem::Alloc;

pub trait BufferLike {
    fn raw_buffer(&self) -> vk::Buffer;
    fn offset(&self) -> vk::DeviceSize {
        0
    }
    fn size(&self) -> vk::DeviceSize;
    fn device_address(&self) -> vk::DeviceAddress;
    /// If the buffer is host visible and mapped, this function returns the host-side address.
    fn as_mut_ptr(&mut self) -> Option<NonNull<u8>>;
}

impl BufferLike for vk::Buffer {
    fn raw_buffer(&self) -> vk::Buffer {
        *self
    }
    fn size(&self) -> vk::DeviceSize {
        vk::WHOLE_SIZE
    }
    fn device_address(&self) -> vk::DeviceAddress {
        panic!()
    }
    fn as_mut_ptr(&mut self) -> Option<NonNull<u8>> {
        panic!()
    }
}

pub struct BufferArray<T> {
    allocator: Allocator,
    allocation: Option<vk_mem::Allocation>,
    buffer: vk::Buffer,
    capacity: u64,
    ptr: *mut MaybeUninit<T>,
    _marker: std::marker::PhantomData<T>,
    usage: vk::BufferUsageFlags,
    sharing_mode: SharingMode<Vec<u32>>,
    allocation_info: vk_mem::AllocationCreateInfo,
}
unsafe impl<T: Send> Send for BufferArray<T> {}
unsafe impl<T: Sync> Sync for BufferArray<T> {}
impl<T> BufferLike for BufferArray<T> {
    fn raw_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    fn size(&self) -> vk::DeviceSize {
        self.capacity * std::mem::size_of::<T>() as u64
    }

    fn device_address(&self) -> vk::DeviceAddress {
        0
    }

    fn as_mut_ptr(&mut self) -> Option<NonNull<u8>> {
        todo!()
    }
}

impl<T> BufferArray<T> {
    /// Ensure that the array is at least `new_capacity` elements long.
    pub fn realloc(&mut self, new_capacity: u64) -> VkResult<()> {
        let new_capacity = new_capacity.next_power_of_two().max(8);
        if new_capacity <= self.capacity {
            return Ok(());
        }
        unsafe {
            if let Some(allocation) = &mut self.allocation {
                self.allocator.destroy_buffer(self.buffer, allocation);
            }
            let (buffer, allocation) = self.allocator.create_buffer(&vk::BufferCreateInfo {
                size: new_capacity * std::mem::size_of::<T>() as u64,
                usage: self.usage,
                sharing_mode: self.sharing_mode.as_raw(),
                queue_family_index_count: self.sharing_mode.queue_family_indices().len() as u32,
                p_queue_family_indices: self.sharing_mode.queue_family_indices().as_ptr(),
                ..Default::default()
            }, &self.allocation_info)?;
            
            let info = self.allocator.get_allocation_info(&allocation);
            self.ptr = info.mapped_data as *mut MaybeUninit<T>;
            self.buffer = buffer;
            self.allocation = Some(allocation);
        }
        Ok(())
    }
    /// Create a new HOST_VISIBLE upload buffer for sequential write.
    /// On integrated GPUs and GPUs with SAM, the upload buffer may be used directly by the device.
    /// On discrete GPUs, the upload buffer serves as the staging buffer. The user will have to create backing
    /// DEVICE_LOCAL buffer and schedule transfer.
    pub fn new_upload(allocator: Allocator, mut usage: vk::BufferUsageFlags) -> VkResult<Self> {
        let memory_model = allocator.device().physical_device().properties().memory_model;
        let memory_usage = if matches!(memory_model, PhysicalDeviceMemoryModel::ReBar) {
            vk_mem::MemoryUsage::AutoPreferDevice
        } else {
            vk_mem::MemoryUsage::AutoPreferHost
        };
        if matches!(memory_model, PhysicalDeviceMemoryModel::Bar | PhysicalDeviceMemoryModel::Discrete) {
            usage |= vk::BufferUsageFlags::TRANSFER_SRC;
        };
        Ok(Self {
            allocator,
            allocation: None,
            buffer: vk::Buffer::null(),
            capacity: 0,
            ptr: std::ptr::null_mut(),
            _marker: std::marker::PhantomData,
            usage,
            sharing_mode: SharingMode::Exclusive,
            allocation_info: vk_mem::AllocationCreateInfo {
                usage: memory_usage,
                flags: vk_mem::AllocationCreateFlags::MAPPED | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        })
    }
    
    pub fn new_resource(allocator: Allocator, mut usage: vk::BufferUsageFlags) -> VkResult<Self> {
        usage |= vk::BufferUsageFlags::TRANSFER_DST;
        Ok(Self {
            allocator,
            allocation: None,
            buffer: vk::Buffer::null(),
            capacity: 0,
            ptr: std::ptr::null_mut(),
            _marker: std::marker::PhantomData,
            usage,
            sharing_mode: SharingMode::Exclusive,
            allocation_info: vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                flags: vk_mem::AllocationCreateFlags::empty(),
                ..Default::default()
            },
        })
    }
}

impl<T> Index<u64> for BufferArray<T> {
    type Output = MaybeUninit<T>;

    fn index(&self, index: u64) -> &Self::Output {
        if self.allocation_info.flags.contains(vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE) {
            tracing::warn!("Reading from a HOST_VISIBLE buffer that is mapped for sequential write is likely inefficient");
        }
        assert!(index < self.capacity);
        unsafe {
            let ptr = self.ptr.add(index as usize);
            &*ptr
        }
    }
}
impl<T> IndexMut<u64> for BufferArray<T> {
    fn index_mut(&mut self, index: u64) -> &mut Self::Output {
        assert!(index < self.capacity);
        unsafe {
            let ptr = self.ptr.add(index as usize);
            &mut *ptr
        }
    }
}
impl<T> Deref for BufferArray<T> {
    type Target = [MaybeUninit<T>];
    
    fn deref(&self) -> &Self::Target {
        if self.allocation_info.flags.contains(vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE) {
            tracing::warn!("Reading from a HOST_VISIBLE buffer that is mapped for sequential write is likely inefficient");
        }
        unsafe {
            std::slice::from_raw_parts(self.ptr, self.capacity as usize)
        }
    }
}
impl<T> DerefMut for BufferArray<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr, self.capacity as usize)
        }
    }
}
