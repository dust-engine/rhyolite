use crate::future::{GPUFuture, GPUFutureBarrierContext, GPUResource, RecordContext};
use crate::{define_future, ImageLike};
use ash::vk;
use std::ops::Deref;

define_future!(BlitImageFuture<'a, S, T>, 'a, I1: ImageLike, I2: ImageLike, S: Unpin + GPUResource + Deref<Target = I1>, T: Unpin + GPUResource + Deref<Target = I2>);
pub struct BlitImageFuture<'a, S, T>
where
    S: GPUResource + Unpin,
    T: GPUResource + Unpin,
{
    src_image: &'a mut S,
    src_image_layout: vk::ImageLayout,
    dst_image: &'a mut T,
    dst_image_layout: vk::ImageLayout,
    regions: &'a [vk::ImageBlit],
    filter: vk::Filter,
}
impl<I1: ImageLike, I2: ImageLike, S, T> GPUFuture for BlitImageFuture<'_, S, T>
where
    S: Unpin + GPUResource + Deref<Target = I1>,
    T: Unpin + GPUResource + Deref<Target = I2>,
{
    type Output = ();

    fn barrier<Ctx: GPUFutureBarrierContext>(&mut self, mut ctx: Ctx) {
        ctx.use_image_resource(
            self.src_image,
            vk::PipelineStageFlags2::BLIT,
            vk::AccessFlags2::TRANSFER_READ,
            self.src_image_layout,
            false,
        );

        ctx.use_image_resource(
            self.dst_image,
            vk::PipelineStageFlags2::BLIT,
            vk::AccessFlags2::TRANSFER_WRITE,
            self.dst_image_layout,
            true,
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
                self.filter,
            );
        }
        Default::default()
    }
}
pub fn blit_image<'a, S, T>(
    src_image: &'a mut S,
    src_image_layout: vk::ImageLayout,
    dst_image: &'a mut T,
    dst_image_layout: vk::ImageLayout,
    regions: &'a [vk::ImageBlit],
    filter: vk::Filter,
) -> BlitImageFuture<'a, S, T>
where
    S: GPUResource + Unpin + ImageLike,
    T: GPUResource + Unpin + ImageLike,
{
    BlitImageFuture {
        src_image,
        src_image_layout,
        dst_image,
        dst_image_layout,
        regions,
        filter,
    }
}

define_future!(ClearColorImageFuture<'a, T>, 'a, I: ImageLike, T: Unpin + GPUResource + Deref<Target = I>);
pub struct ClearColorImageFuture<'a, T> {
    dst_image: T,
    layout: vk::ImageLayout,
    clear_color: vk::ClearColorValue,
    ranges: &'a [vk::ImageSubresourceRange],
}
impl<T> ClearColorImageFuture<'_, T> {
    pub fn with_layout(mut self, layout: vk::ImageLayout) -> Self {
        self.layout = layout;
        self
    }
}
impl<I: ImageLike, T> GPUFuture for ClearColorImageFuture<'_, T>
where
    T: Unpin + GPUResource + Deref<Target = I>,
{
    type Output = ();

    fn barrier<Ctx: GPUFutureBarrierContext>(&mut self, mut ctx: Ctx) {
        ctx.use_image_resource(
            &mut self.dst_image,
            vk::PipelineStageFlags2::CLEAR,
            vk::AccessFlags2::TRANSFER_WRITE,
            self.layout,
            true,
        );
    }

    fn record(self, ctx: RecordContext) -> (Self::Output, Self::Retained) {
        unsafe {
            ctx.device.cmd_clear_color_image(
                ctx.command_buffer,
                self.dst_image.raw_image(),
                self.layout,
                &self.clear_color,
                self.ranges,
            );
        }
        Default::default()
    }
}
pub fn clear_color_image<'a, T, I: ImageLike>(
    dst_image: T,
    clear_color: vk::ClearColorValue,
    ranges: &'a [vk::ImageSubresourceRange],
) -> ClearColorImageFuture<'a, T>
where
    T: GPUResource + Deref<Target = I> + Unpin,
{
    ClearColorImageFuture {
        dst_image,
        layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        clear_color,
        ranges,
    }
}
