use crate::buffer::BufferLike;
use crate::future::{BarrierContext, GPUFuture, GPUResource, RecordContext};
use crate::{define_future, ImageLike};
use ash::vk;
use std::ops::Deref;

//region Blit
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
impl<'a, I1: ImageLike, I2: ImageLike, S, T> BlitImageFuture<'a, S, T>
where
    S: Unpin + GPUResource + Deref<Target = I1>,
    T: Unpin + GPUResource + Deref<Target = I2>,
{
    pub fn with_filter(mut self, filter: vk::Filter) -> Self {
        self.filter = filter;
        self
    }
    pub fn with_src_layout(mut self, layout: vk::ImageLayout) -> Self {
        self.src_image_layout = layout;
        self
    }
    pub fn with_dst_layout(mut self, layout: vk::ImageLayout) -> Self {
        self.dst_image_layout = layout;
        self
    }
    pub fn with_regions(mut self, regions: &'a [vk::ImageBlit]) -> Self {
        assert!(self.regions.is_empty());
        self.regions = regions;
        self
    }
}
impl<I1: ImageLike, I2: ImageLike, S, T> GPUFuture for BlitImageFuture<'_, S, T>
where
    S: Unpin + GPUResource + Deref<Target = I1>,
    T: Unpin + GPUResource + Deref<Target = I2>,
{
    type Output = ();

    fn barrier(&mut self, mut ctx: BarrierContext) {
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
        let image_blit;
        unsafe {
            ctx.device.cmd_blit_image(
                ctx.command_buffer,
                self.src_image.raw_image(),
                self.src_image_layout,
                self.dst_image.raw_image(),
                self.dst_image_layout,
                if self.regions.is_empty() {
                    let src_subresource_range = self.src_image.subresource_range();
                    let src_offset_min = self.src_image.offset();
                    let src_offset_max = src_offset_min + self.src_image.extent().as_ivec3();
                    let dst_subresource_range = self.dst_image.subresource_range();
                    let dst_offset_min = self.dst_image.offset();
                    let dst_offset_max = dst_offset_min + self.dst_image.extent().as_ivec3();
                    image_blit = [vk::ImageBlit {
                        src_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: src_subresource_range.aspect_mask,
                            mip_level: 0,
                            base_array_layer: src_subresource_range.base_array_layer,
                            layer_count: src_subresource_range.layer_count,
                        },
                        src_offsets: [
                            vk::Offset3D {
                                x: src_offset_min.x as i32,
                                y: src_offset_min.y as i32,
                                z: src_offset_min.z as i32,
                            },
                            vk::Offset3D {
                                x: src_offset_max.x as i32,
                                y: src_offset_max.y as i32,
                                z: src_offset_max.z as i32,
                            },
                        ],
                        dst_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: dst_subresource_range.aspect_mask,
                            mip_level: 0,
                            base_array_layer: dst_subresource_range.base_array_layer,
                            layer_count: dst_subresource_range.layer_count,
                        },
                        dst_offsets: [
                            vk::Offset3D {
                                x: dst_offset_min.x as i32,
                                y: dst_offset_min.y as i32,
                                z: dst_offset_min.z as i32,
                            },
                            vk::Offset3D {
                                x: dst_offset_max.x as i32,
                                y: dst_offset_max.y as i32,
                                z: dst_offset_max.z as i32,
                            },
                        ],
                    }];
                    &image_blit
                } else {
                    self.regions
                },
                self.filter,
            );
        }
        Default::default()
    }
}
pub fn blit_image<'a, S, T, SI: ImageLike, TI: ImageLike>(
    src_image: &'a mut S,
    dst_image: &'a mut T,
) -> BlitImageFuture<'a, S, T>
where
    S: GPUResource + Unpin + Deref<Target = SI>,
    T: GPUResource + Unpin + Deref<Target = TI>,
{
    BlitImageFuture {
        src_image,
        src_image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        dst_image,
        dst_image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        regions: &[],
        filter: vk::Filter::NEAREST,
    }
}
//endregion

//region Clear
define_future!(ClearColorImageFuture<'a, T>, 'a, I: ImageLike, T: Unpin + GPUResource + Deref<Target = I>);
pub struct ClearColorImageFuture<'a, T> {
    dst_image: &'a mut T,
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

    fn barrier(&mut self, mut ctx: BarrierContext) {
        ctx.use_image_resource(
            self.dst_image,
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
    dst_image: &'a mut T,
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
//endregion

//region CopyBufferToImage
define_future!(CopyBufferToImageFuture<'a, T, B>, 'a, I: ImageLike + ?Sized, T: Unpin + GPUResource + Deref<Target = I>, J: BufferLike + ?Sized, B: Unpin + GPUResource + Deref<Target = J>);
pub struct CopyBufferToImageFuture<'a, T, B> {
    src_buffer: &'a mut B,
    dst_image: &'a mut T,
    layout: vk::ImageLayout,
    regions: &'a [vk::BufferImageCopy],
}
impl<T, B> CopyBufferToImageFuture<'_, T, B> {
    pub fn with_layout(mut self, layout: vk::ImageLayout) -> Self {
        self.layout = layout;
        self
    }
}
impl<T, B, I: ImageLike + ?Sized, J: BufferLike + ?Sized> GPUFuture
    for CopyBufferToImageFuture<'_, T, B>
where
    T: Unpin + GPUResource + Deref<Target = I>,
    B: Unpin + GPUResource + Deref<Target = J>,
{
    type Output = ();

    fn barrier(&mut self, mut ctx: BarrierContext) {
        ctx.use_image_resource(
            self.dst_image,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_WRITE,
            self.layout,
            true,
        );
        ctx.use_resource(
            self.src_buffer,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
        );
    }

    fn record(self, ctx: RecordContext) -> (Self::Output, Self::Retained) {
        let image_region;
        unsafe {
            ctx.device.cmd_copy_buffer_to_image(
                ctx.command_buffer,
                self.src_buffer.raw_buffer(),
                self.dst_image.raw_image(),
                self.layout,
                if self.regions.is_empty() {
                    let dst_subresource_range = self.dst_image.subresource_range();
                    let dst_offset = self.dst_image.offset();
                    let dst_extent = self.dst_image.extent();
                    image_region = [vk::BufferImageCopy {
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: dst_subresource_range.aspect_mask,
                            mip_level: 0,
                            base_array_layer: dst_subresource_range.base_array_layer,
                            layer_count: dst_subresource_range.layer_count,
                        },
                        image_offset: vk::Offset3D {
                            x: dst_offset.x as i32,
                            y: dst_offset.y as i32,
                            z: dst_offset.z as i32,
                        },
                        image_extent: vk::Extent3D {
                            width: dst_extent.x,
                            height: dst_extent.y,
                            depth: dst_extent.z,
                        },
                        buffer_image_height: 0,
                        buffer_row_length: 0,
                        buffer_offset: self.src_buffer.offset(),
                    }];
                    &image_region
                } else {
                    self.regions
                },
            );
        }
        Default::default()
    }
}
pub fn copy_buffer_to_image<'a, T, B, I: ImageLike + ?Sized, J: BufferLike + ?Sized>(
    src_buffer: &'a mut B,
    dst_image: &'a mut T,
) -> CopyBufferToImageFuture<'a, T, B>
where
    T: GPUResource + Deref<Target = I> + Unpin,
    B: GPUResource + Deref<Target = J> + Unpin,
{
    CopyBufferToImageFuture {
        src_buffer,
        dst_image,
        layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        regions: &[],
    }
}
//endregion
