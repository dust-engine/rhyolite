mod buffer;
mod copy;
use std::ops::Deref;

//mod image;
use ash::vk;
pub use buffer::*;
pub use copy::*;
//pub use image::*;
pub trait ImageLike {
    fn raw_image(&self) -> vk::Image;
    fn subresource_range(&self) -> vk::ImageSubresourceRange;
    fn extent(&self) -> vk::Extent3D;
    fn offset(&self) -> vk::Offset3D {
        Default::default()
    }
}
impl<I: ImageLike, T: Deref<Target = I>> ImageLike for T {
    fn raw_image(&self) -> vk::Image {
        self.deref().raw_image()
    }
    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        self.deref().subresource_range()
    }
    fn extent(&self) -> vk::Extent3D {
        self.deref().extent()
    }
    fn offset(&self) -> vk::Offset3D {
        self.deref().offset()
    }
}

pub trait ImageExt {
    fn crop(self, extent: vk::Extent3D, offset: vk::Offset3D) -> ImageSubregion<Self>
    where
        Self: ImageLike + Sized,
    {
        let sub_offset = self.offset();
        let sub_extent = self.extent();

        let offset = vk::Offset3D {
            x: sub_offset.x + offset.x,
            y: sub_offset.y + offset.y,
            z: sub_offset.z + offset.z,
        };
        assert!(
            extent.width <= sub_extent.width
                && extent.height <= sub_extent.height
                && extent.depth <= sub_extent.depth
        );
        ImageSubregion {
            inner: self,
            extent,
            offset,
        }
    }
}
impl<T> ImageExt for T where T: ImageLike {}

pub struct ImageSubregion<T: ImageLike> {
    inner: T,
    extent: vk::Extent3D,
    offset: vk::Offset3D,
}
impl<T: ImageLike> ImageSubregion<T> {
    pub fn into_inner(self) -> T {
        self.inner
    }
}
impl<T: ImageLike> ImageLike for ImageSubregion<T> {
    fn raw_image(&self) -> vk::Image {
        self.inner.raw_image()
    }

    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        self.inner.subresource_range()
    }

    fn extent(&self) -> vk::Extent3D {
        self.extent
    }

    fn offset(&self) -> vk::Offset3D {
        self.offset
    }
}

pub enum SharingMode<'a> {
    Exclusive,
    Concurrent { queue_family_indices: &'a [u32] },
}
impl<'a> Default for SharingMode<'a> {
    fn default() -> Self {
        Self::Exclusive
    }
}
