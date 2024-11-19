use ash::vk;
use crate::{define_future, ImageLike};

use super::{res::TrackedResource, BarrierContext, GPUFuture, GPURetained, RecordContext};

pub struct Yield;
define_future!(Yield);
impl GPUFuture for Yield {
    type Output = ();

    fn barrier(&mut self, mut ctx: BarrierContext) {
    }

    fn record(self, ctx: RecordContext) -> (Self::Output, Self::Retained) {
        Default::default()
    }
}
pub fn yield_now() -> Yield {
    Yield
}




define_future!(BlitFuture<'a, S, T>, 'a, S: Unpin + TrackedResource + ImageLike, T: Unpin + TrackedResource + ImageLike);
pub struct BlitFuture<'a, S, T> {
    src_image: GPURetained<'a, S>,
    src_image_layout: vk::ImageLayout,
    dst_image: GPURetained<'a, T>,
    dst_image_layout: vk::ImageLayout,
    regions: &'a [vk::ImageBlit],
    filter: vk::Filter,
}
impl<S, T> GPUFuture for BlitFuture<'_, S, T>  where S: Unpin + TrackedResource + ImageLike, T: Unpin + TrackedResource + ImageLike {
    type Output = ();

    fn barrier(&mut self, mut ctx: BarrierContext) {
        ctx.use_image_resource(
            &mut *self.src_image,
            vk::PipelineStageFlags2::BLIT,
            vk::AccessFlags2::TRANSFER_READ,
            self.src_image_layout,
            false
        );

        ctx.use_image_resource(
            &mut *self.dst_image,
            vk::PipelineStageFlags2::BLIT,
            vk::AccessFlags2::TRANSFER_WRITE,
            self.dst_image_layout,
            true
        );
    }

    fn record(self, ctx: RecordContext) -> (Self::Output, Self::Retained) {
        unsafe {
            ctx.device.cmd_blit_image(
                ctx.command_buffer,
                self.src_image.raw_image(),
                self.src_image_layout,
                self.dst_image.raw_image(),
                self.dst_image_layout,
                self.regions,
                self.filter
            );
        }
        Default::default()
    }
}

