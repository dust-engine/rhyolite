use std::{
    ops::{Deref, DerefMut},
    pin::Pin,
    task::Poll,
};

use ash::vk;
use pin_project::pin_project;

use crate::future::{GPUCommandFuture, GlobalContext, Res, StageContext};

pub trait BufferSlice {
    fn raw_buffer(&self) -> vk::Buffer;
    fn offset(&self) -> vk::DeviceSize;
    fn size(&self) -> vk::DeviceSize;
}

#[pin_project]
pub struct CopyBufferFuture<
    'b,
    S: BufferSlice + 'b,
    T: BufferSlice + 'b,
    SRef: Deref<Target = Res<'b, S>>,
    TRef: DerefMut<Target = Res<'b, T>>,
> {
    pub str: &'static str,
    pub src: SRef,
    pub dst: TRef,
}
impl<
        'b,
        S: BufferSlice,
        T: BufferSlice,
        SRef: Deref<Target = Res<'b, S>>,
        TRef: DerefMut<Target = Res<'b, T>>,
    > GPUCommandFuture for CopyBufferFuture<'b, S, T, SRef, TRef>
{
    type Output = ();
    type RetainedState = ();
    #[inline]
    fn record(
        self: Pin<&mut Self>,
        ctx: &mut GlobalContext,
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
    }
}
