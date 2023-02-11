use std::{
    ops::{Deref, DerefMut},
    pin::Pin,
    task::Poll,
};

use ash::vk;
use pin_project::pin_project;

use crate::{
    future::{CommandBufferRecordContext, GPUCommandFuture, Res, StageContext},
    HasDevice,
};

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
    SRef: Deref<Target = Res<S>>,
    TRef: DerefMut<Target = Res<T>>,
> {
    pub str: &'static str,
    pub src: SRef,
    pub dst: TRef,
}
impl<
        S: BufferLike,
        T: BufferLike,
        SRef: Deref<Target = Res<S>>,
        TRef: DerefMut<Target = Res<T>>,
    > GPUCommandFuture for CopyBufferFuture<S, T, SRef, TRef>
{
    type Output = ();
    type RetainedState = ();
    type RecycledState = ();
    #[inline]
    fn record<'fa, 'fb: 'fa>(
        self: Pin<&mut Self>,
        ctx: &'fa mut CommandBufferRecordContext<'fb>,
        _recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        println!("{}, on stage {}", self.str, ctx.current_stage_index());
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
    SRef: Deref<Target = Res<S>>,
    TRef: DerefMut<Target = Res<T>>,
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
