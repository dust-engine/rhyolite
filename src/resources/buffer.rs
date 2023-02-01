use std::{
    ops::{Deref, DerefMut},
    pin::Pin,
    sync::Arc,
    task::Poll,
};

use ash::vk;
use pin_project::pin_project;

use crate::{
    future::{CommandBufferRecordContext, GPUCommandFuture, Res, StageContext},
    Device,
};

pub trait BufferLike {
    fn raw_buffer(&self) -> vk::Buffer;
    fn offset(&self) -> vk::DeviceSize;
    fn size(&self) -> vk::DeviceSize;
}

pub struct Buffer {
    device: Arc<Device>,
    raw: vk::Buffer,
}
impl Buffer {
    pub fn from_raw(device: Arc<Device>, raw: vk::Buffer) -> Self {
        Self { device, raw }
    }
}

// Everyone wants a mutable refence to outer.
// Some people wants a mutable reference to inner.
// In the case of Fork. Each fork gets a & of the container. Container must be generic over &mut, and BorrowMut.
// Inner product must be generic over &mut and RefCell as well.

#[pin_project]
pub struct CopyBufferFuture<
    'b,
    S: BufferLike + 'b,
    T: BufferLike + 'b,
    SRef: Deref<Target = Res<'b, S>>,
    TRef: DerefMut<Target = Res<'b, T>>,
> {
    pub str: &'static str,
    pub src: SRef,
    pub dst: TRef,
}
impl<
        'b,
        S: BufferLike,
        T: BufferLike,
        SRef: Deref<Target = Res<'b, S>>,
        TRef: DerefMut<Target = Res<'b, T>>,
    > GPUCommandFuture for CopyBufferFuture<'b, S, T, SRef, TRef>
{
    type Output = ();
    type RetainedState = ();
    type RecycledState = ();
    #[inline]
    fn record<'fa, 'fb: 'fa>(
        self: Pin<&mut Self>,
        ctx: &'fa mut CommandBufferRecordContext<'fb>,
        recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        println!("{}, on stage {}", self.str, ctx.current_stage_index());
        let this = self.project();
        let src = this.src.deref().inner();
        let dst = this.dst.deref_mut().inner_mut();
        let region = vk::BufferCopy2 {
            src_offset: src.offset(),
            dst_offset: dst.offset(),
            size: src.size().min(dst.size()),
            ..Default::default()
        };
        let copy = vk::CopyBufferInfo2 {
            src_buffer: src.raw_buffer(),
            dst_buffer: dst.raw_buffer(),
            region_count: 1,
            p_regions: &region,
            ..Default::default()
        };
        Poll::Ready(((), ()))
    }
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        /*
        let this = self.project();
        let src = this.src.deref();
        let dst = this.dst.deref_mut();
        ctx.read(
            src,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
        );
        ctx.write(
            dst,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
        */
    }
}
