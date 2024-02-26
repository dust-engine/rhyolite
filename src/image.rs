use ash::{prelude::VkResult, vk};
use std::{ops::DerefMut, sync::Arc};

pub trait ImageLike {
    fn raw_image(&self) -> vk::Image;
    fn subresource_range(&self) -> vk::ImageSubresourceRange;
    fn extent(&self) -> vk::Extent3D;
    fn offset(&self) -> vk::Offset3D {
        Default::default()
    }
    fn format(&self) -> vk::Format;
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

    fn format(&self) -> vk::Format {
        self.inner.format()
    }
}
