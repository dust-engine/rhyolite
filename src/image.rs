use ash::{prelude::VkResult, vk};

use crate::Allocator;
use vk_mem::Alloc;

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
pub trait ImageViewLike: ImageLike {
    fn raw_image_view(&self) -> vk::ImageView;
}

pub struct Image {
    allocator: Allocator,
    image: vk::Image,
    allocation: vk_mem::Allocation,
    extent: vk::Extent3D,
}
impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .destroy_image(self.image, &mut self.allocation);
        }
    }
}
impl Image {
    pub fn new_device_image(allocator: Allocator, info: &vk::ImageCreateInfo) -> VkResult<Self> {
        unsafe {
            let (image, allocation) = allocator.create_image(
                info,
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferDevice,
                    ..Default::default()
                },
            )?;
            Ok(Self {
                extent: info.extent,
                allocator,
                image,
                allocation,
            })
        }
    }
    pub fn extent(&self) -> vk::Extent3D {
        self.extent
    }
    pub fn raw(&self) -> vk::Image {
        self.image
    }
}
