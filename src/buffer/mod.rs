pub(crate) mod staging;

use std::{
    alloc::Layout,
    cell::UnsafeCell,
    ops::{Deref, DerefMut, RangeBounds},
    ptr::NonNull,
};

use ash::{prelude::VkResult, vk};

use crate::{Allocator, HasDevice};
pub use staging::{StagingBelt, StagingBeltSuballocation, UniformBelt};
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
    device_address: u64,
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
        usage: vk::BufferUsageFlags,
    ) -> Self {
        let info = allocator.get_allocation_info(&allocation);
        let device_address = if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            unsafe {
                allocator
                    .device()
                    .get_buffer_device_address(&vk::BufferDeviceAddressInfo {
                        buffer,
                        ..Default::default()
                    })
            }
        } else {
            0
        };

        Self {
            allocator,
            buffer,
            allocation,
            size: info.size,
            device_address,
        }
    }
    /// Create a buffer for small amount of host -> device dataflow.
    /// On Discrete devices:
    ///
    /// Integrated, ReBar: Create a new buffer with HOST_VISIBLE, preferably DEVICE_LOCAL memory.
    /// Bar: Create a new buffer on the 256MB Bar.
    /// Discrete: Create a new buffer on device memory. Application to use StagingBelt for updates.
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
            Ok(Self::from_raw(allocator, buffer, allocation, usage))
        }
    }
    /// Discrete, Bar: Create a new buffer on host memory.
    /// Integrated, ReBar: Not applicable.
    pub fn new_host(
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
            Ok(Self::from_raw(allocator, buffer, allocation, usage))
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
            Ok(Self::from_raw(allocator, buffer, allocation, usage))
        }
    }

    /*
    pub fn new_resource_init(
        allocator: Allocator,
        staging_belt: &mut StagingBelt,
        data: &[u8],
        alignment: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        commands: &mut impl TransferCommands,
    ) -> VkResult<Self> {
        Self::new_resource_init_with(
            allocator,
            staging_belt,
            data.len() as vk::DeviceSize,
            alignment,
            usage,
            commands,
            |slice| {
                slice.copy_from_slice(data);
            },
        )
    }
    pub fn new_resource_init_with(
        allocator: Allocator,
        staging_belt: &mut StagingBelt,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        commands: &mut impl TransferCommands,
        initializer: impl FnOnce(&mut [u8]),
    ) -> VkResult<Self> {
        let memory_model = allocator
            .device()
            .physical_device()
            .properties()
            .memory_model;
        match memory_model {
            PhysicalDeviceMemoryModel::Bar | PhysicalDeviceMemoryModel::Discrete => {
                let mut batch = staging_belt.start(commands.semaphore_signal());
                let mut staging = batch.allocate_buffer(size);
                unsafe {
                    initializer(&mut staging);
                    let (buffer, allocation) = allocator.create_buffer_with_alignment(
                        &vk::BufferCreateInfo {
                            size,
                            usage: usage | vk::BufferUsageFlags::TRANSFER_DST,
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
                    commands.copy_buffer(
                        staging.buffer,
                        buffer,
                        &[vk::BufferCopy {
                            src_offset: staging.offset(),
                            dst_offset: 0,
                            size,
                        }],
                    );
                    Ok(Self {
                        allocator,
                        buffer,
                        allocation,
                        size,
                    })
                }
            }
            _ => {
                // Direct write
                unsafe {
                    let (buffer, mut allocation) = allocator.create_buffer_with_alignment(
                        &vk::BufferCreateInfo {
                            size,
                            usage,
                            ..Default::default()
                        },
                        &vk_mem::AllocationCreateInfo {
                            usage: vk_mem::MemoryUsage::AutoPreferDevice,
                            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                            preferred_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                            ..Default::default()
                        },
                        alignment,
                    )?;
                    let ptr = allocator.map_memory(&mut allocation)?;
                    initializer(std::slice::from_raw_parts_mut(
                        ptr as *mut u8,
                        size as usize,
                    ));
                    allocator.unmap_memory(&mut allocation);
                    Ok(Self {
                        allocator,
                        buffer,
                        allocation,
                        size,
                    })
                }
            }
        }
    }
    */

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
        self.device_address
    }
}

/// A special allocator designed to be used for [`BufferVec`].
/// Holds one allocation at any given time.
pub struct BufferAllocator {
    allocator: Allocator,
    buffer: UnsafeCell<Option<Buffer>>,
    usage: vk::BufferUsageFlags,
    create_info: vk_mem::AllocationCreateInfo,
}
unsafe impl std::alloc::Allocator for BufferAllocator {
    fn allocate(&self, layout: Layout) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        unsafe {
            let (buffer, allocation) = self
                .allocator
                .create_buffer_with_alignment(
                    &vk::BufferCreateInfo {
                        size: layout.size() as u64,
                        usage: self.usage,
                        ..Default::default()
                    },
                    &self.create_info,
                    layout.align() as u64,
                )
                .map_err(|_| std::alloc::AllocError)?;
            let info = self.allocator.get_allocation_info(&allocation);
            let buffer = Buffer::from_raw(self.allocator.clone(), buffer, allocation, self.usage);
            let old_buffer = (&mut *self.buffer.get()).replace(buffer);
            assert!(old_buffer.is_none());
            let ptr = NonNull::new(info.mapped_data as *mut u8).unwrap();
            Ok(NonNull::slice_from_raw_parts(ptr, info.size as usize))
        }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        let (buffer, allocation) = self
            .allocator
            .create_buffer_with_alignment(
                &vk::BufferCreateInfo {
                    size: new_layout.size() as u64,
                    usage: self.usage,
                    ..Default::default()
                },
                &self.create_info,
                new_layout.align() as u64,
            )
            .map_err(|_| std::alloc::AllocError)?;
        let info = self.allocator.get_allocation_info(&allocation);
        let buffer = Buffer::from_raw(self.allocator.clone(), buffer, allocation, self.usage);
        let old_buffer = (&mut *self.buffer.get()).replace(buffer);
        drop(old_buffer);
        let ptr = NonNull::new(info.mapped_data as *mut u8).unwrap();
        Ok(NonNull::slice_from_raw_parts(ptr, info.size as usize))
    }

    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: Layout) {
        let buffer = (&mut *self.buffer.get()).take();
        assert!(buffer.is_some());
        drop(buffer);
    }
}
impl<T> BufferLike for BufferVec<T> {
    fn raw_buffer(&self) -> vk::Buffer {
        unsafe { (&*self.0.allocator().buffer.get()).as_ref().unwrap().buffer }
    }
    fn size(&self) -> vk::DeviceSize {
        unsafe { (&*self.0.allocator().buffer.get()).as_ref().unwrap().size }
    }
    fn device_address(&self) -> vk::DeviceAddress {
        unsafe {
            (&*self.0.allocator().buffer.get())
                .as_ref()
                .unwrap()
                .device_address
        }
    }
}
impl<T> BufferVec<T> {
    /// Create a buffer for small amount of host -> device dataflow.
    /// On Discrete devices:
    ///
    /// Integrated, ReBar: Create a new buffer with HOST_VISIBLE, preferably DEVICE_LOCAL memory.
    /// Bar: Create a new buffer on the 256MB Bar.
    /// Discrete: Create a new buffer on device memory. Application to use StagingBelt for updates.
    pub fn new_dynamic(
        allocator: Allocator,
        alignment: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        Self(Vec::new_in(BufferAllocator {
            allocator,
            buffer: UnsafeCell::new(None),
            usage,
            create_info: vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                flags: vk_mem::AllocationCreateFlags::MAPPED
                    | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        }))
    }
    /// Create a buffer for large amount of host -> device dataflow.
    /// Only difference with `new_dynamic` is that on GPUs with Bar, this creates the buffer on the host.
    pub fn new_upload(
        allocator: Allocator,
        alignment: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        Self(Vec::new_in(BufferAllocator {
            allocator,
            buffer: UnsafeCell::new(None),
            usage,
            create_info: vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                flags: vk_mem::AllocationCreateFlags::MAPPED
                    | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        }))
    }
    pub fn new_host(
        allocator: Allocator,
        alignment: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        Self(Vec::new_in(BufferAllocator {
            allocator,
            buffer: UnsafeCell::new(None),
            usage,
            create_info: vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferHost,
                flags: vk_mem::AllocationCreateFlags::MAPPED
                    | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        }))
    }
    /// Create a new buffer with DEVICE_LOCAL memory.
    pub fn new_resource(
        allocator: Allocator,
        alignment: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        Self(Vec::new_in(BufferAllocator {
            allocator,
            buffer: UnsafeCell::new(None),
            usage,
            create_info: vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                preferred_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                ..Default::default()
            },
        }))
    }

    pub fn flush(&mut self) -> VkResult<()> {
        let Some(buffer) = &unsafe { &mut *self.0.allocator().buffer.get() }.as_ref() else {
            return Ok(());
        };
        self.0
            .allocator()
            .allocator
            .flush_allocation(&buffer.allocation, 0, self.0.len() as u64)
    }
}

pub struct BufferVec<T>(Vec<T, BufferAllocator>);
unsafe impl<T: Send> Send for BufferVec<T> {}
unsafe impl<T: Sync> Sync for BufferVec<T> {}
impl<T> Deref for BufferVec<T> {
    type Target = Vec<T, BufferAllocator>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> DerefMut for BufferVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
