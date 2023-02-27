use std::{
    ops::{Deref, DerefMut},
    pin::Pin,
    task::Poll,
};

use ash::prelude::VkResult;
use ash::vk;
use pin_project::pin_project;

use crate::{
    future::{CommandBufferRecordContext, GPUCommandFuture, RenderRes, StageContext},
    macros::commands,
    Allocator, HasDevice, PhysicalDeviceMemoryModel, SharingMode,
};
use vk_mem::Alloc;

pub trait BufferLike {
    fn raw_buffer(&self) -> vk::Buffer;
    fn offset(&self) -> vk::DeviceSize {
        0
    }
    fn size(&self) -> vk::DeviceSize;
}
impl BufferLike for vk::Buffer {
    fn raw_buffer(&self) -> vk::Buffer {
        *self
    }
    fn size(&self) -> vk::DeviceSize {
        u64::MAX
    }
}
impl<T> BufferLike for &T
where
    T: BufferLike,
{
    fn raw_buffer(&self) -> vk::Buffer {
        let this: &T = self;
        this.raw_buffer()
    }
    fn offset(&self) -> vk::DeviceSize {
        let this: &T = self;
        this.offset()
    }
    fn size(&self) -> vk::DeviceSize {
        let this: &T = self;
        this.size()
    }
}
impl<T> BufferLike for &mut T
where
    T: BufferLike,
{
    fn raw_buffer(&self) -> vk::Buffer {
        let this: &T = self;
        this.raw_buffer()
    }
    fn offset(&self) -> vk::DeviceSize {
        let this: &T = self;
        this.offset()
    }
    fn size(&self) -> vk::DeviceSize {
        let this: &T = self;
        this.size()
    }
}

// Everyone wants a mutable refence to outer.
// Some people wants a mutable reference to inner.
// In the case of Fork. Each fork gets a & of the container. Container must be generic over &mut, and BorrowMut.
// Inner product must be generic over &mut and RefCell as well.

#[pin_project]
pub struct CopyBufferFuture<
    S: BufferLike,
    T: BufferLike,
    SRef: Deref<Target = RenderRes<S>>,
    TRef: DerefMut<Target = RenderRes<T>>,
> {
    pub str: &'static str,
    pub src: SRef,
    pub dst: TRef,
}
impl<
        S: BufferLike,
        T: BufferLike,
        SRef: Deref<Target = RenderRes<S>>,
        TRef: DerefMut<Target = RenderRes<T>>,
    > GPUCommandFuture for CopyBufferFuture<S, T, SRef, TRef>
{
    type Output = ();
    type RetainedState = ();
    type RecycledState = ();
    #[inline]
    fn record(
        self: Pin<&mut Self>,
        ctx: &mut CommandBufferRecordContext,
        _recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        let src = this.src.deref().inner();
        let dst = this.dst.deref_mut().inner_mut();
        let region = vk::BufferCopy {
            src_offset: src.offset(),
            dst_offset: dst.offset(),
            size: src.size().min(dst.size()),
        };
        ctx.record(|ctx, command_buffer| unsafe {
            ctx.device().cmd_copy_buffer(
                command_buffer,
                src.raw_buffer(),
                dst.raw_buffer(),
                &[region],
            );
        });
        Poll::Ready(((), ()))
    }
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        let this = self.project();
        ctx.read(
            this.src,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
        );

        ctx.write(
            this.dst,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
    }
}

pub fn copy_buffer<
    S: BufferLike,
    T: BufferLike,
    SRef: Deref<Target = RenderRes<S>>,
    TRef: DerefMut<Target = RenderRes<T>>,
>(
    src: SRef,
    dst: TRef,
) -> CopyBufferFuture<S, T, SRef, TRef> {
    CopyBufferFuture {
        str: "aaa",
        src,
        dst,
    }
}

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

    pub fn contents_mut(&self) -> Option<&mut [u8]> {
        let info = self.allocator.inner().get_allocation_info(&self.allocation);
        if info.mapped_data.is_null() {
            None
        } else {
            unsafe {
                Some(std::slice::from_raw_parts_mut(
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
    /// Crate a small device-local buffer with uninitialized data, guaranteed local to the GPU.
    /// The data will be host visible on ResizableBar, Bar, and UMA memory models.
    /// TRANSFER_DST usage flag will be automatically added to the created buffer.
    /// Suitable for small amount of dynamic data.
    pub fn create_upload_buffer_uninit(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> VkResult<ResidentBuffer> {
        let mut create_info = vk::BufferCreateInfo {
            size,
            usage,
            ..Default::default()
        };

        let dst_buffer = match self.device().physical_device().memory_model() {
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
                buf
            }
            PhysicalDeviceMemoryModel::Discrete => {
                create_info.usage |= vk::BufferUsageFlags::TRANSFER_DST;
                let dst_buffer = self.create_resident_buffer(
                    &create_info,
                    &vk_mem::AllocationCreateInfo {
                        flags: vk_mem::AllocationCreateFlags::empty(),
                        usage: vk_mem::MemoryUsage::AutoPreferDevice,
                        ..Default::default()
                    },
                )?;
                dst_buffer
            }
        };
        Ok(dst_buffer)
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

    pub fn create_staging_buffer(&self, size: vk::DeviceSize) -> VkResult<ResidentBuffer> {
        self.create_resident_buffer(
            &vk::BufferCreateInfo {
                size,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                ..Default::default()
            },
            &vk_mem::AllocationCreateInfo {
                flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                usage: vk_mem::MemoryUsage::AutoPreferHost,
                ..Default::default()
            },
        )
    }

    /// Crate a small device-local buffer with a writer callback, only visible to the GPU.
    /// The data will be directly written to the buffer on ResizableBar, Bar, and UMA memory models.
    /// We will create a temporary staging buffer on Discrete GPUs with no host-accessible device-local memory.
    /// TRANSFER_DST usage flag will be automatically added to the created buffer.
    /// Suitable for small amount of dynamic data.
    pub fn create_device_buffer_with_writer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        writer: impl for<'a> FnOnce(&'a mut [u8]),
    ) -> VkResult<impl GPUCommandFuture<Output = RenderRes<ResidentBuffer>>> {
        let dst_buffer = self.create_upload_buffer_uninit(size, usage)?;
        let staging_buffer = if let Some(contents) = dst_buffer.contents_mut() {
            writer(contents);
            None
        } else {
            let staging_buffer = self.create_staging_buffer(size)?;
            writer(staging_buffer.contents_mut().unwrap());
            Some(staging_buffer)
        };

        Ok(commands! {
            let mut dst_buffer = RenderRes::new(dst_buffer);
            if let Some(staging_buffer) = staging_buffer {
                let staging_buffer = RenderRes::new(staging_buffer);
                copy_buffer(&staging_buffer, &mut dst_buffer).await;
                retain!(staging_buffer);
            }
            dst_buffer
        })
    }

    /// Crate a small device-local buffer with pre-populated data, only visible to the GPU.
    /// The data will be directly written to the buffer on ResizableBar, Bar, and UMA memory models.
    /// We will create a temporary staging buffer on Discrete GPUs with no host-accessible device-local memory.
    /// TRANSFER_DST usage flag will be automatically added to the created buffer.
    /// Suitable for small amount of dynamic data.
    pub fn create_device_buffer_with_data(
        &self,
        data: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> VkResult<impl GPUCommandFuture<Output = RenderRes<ResidentBuffer>>> {
        let dst_buffer = self.create_upload_buffer_uninit(data.len() as u64, usage)?;
        let staging_buffer = if let Some(contents) = dst_buffer.contents_mut() {
            contents[..data.len()].copy_from_slice(data);
            None
        } else {
            let staging_buffer = self.create_staging_buffer(data.len() as u64)?;
            staging_buffer.contents_mut().unwrap()[..data.len()].copy_from_slice(data);
            Some(staging_buffer)
        };

        Ok(commands! {
            let mut dst_buffer = RenderRes::new(dst_buffer);
            if let Some(staging_buffer) = staging_buffer {
                let staging_buffer = RenderRes::new(staging_buffer);
                copy_buffer(&staging_buffer, &mut dst_buffer).await;
                retain!(staging_buffer);
            }
            dst_buffer
        })
    }
}
