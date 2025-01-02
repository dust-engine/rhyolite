use ash::{prelude::VkResult, vk};
use bevy::math::{IVec3, UVec3};

use crate::{Allocator, HasDevice};
use vk_mem::Alloc;

pub trait ImageLike: Send + Sync + 'static {
    fn raw_image(&self) -> vk::Image;
    fn subresource_range(&self) -> vk::ImageSubresourceRange;
    fn extent(&self) -> UVec3;
    fn offset(&self) -> IVec3 {
        IVec3::ZERO
    }
    fn format(&self) -> vk::Format;
}

pub trait ImageExt {
    fn crop(self, extent: UVec3, offset: IVec3) -> ImageSubregion<Self>
    where
        Self: ImageLike + Sized,
    {
        let sub_offset = self.offset();
        let sub_extent = self.extent();

        let offset = sub_offset + offset;
        assert!(extent.x <= sub_extent.x && extent.y <= sub_extent.y && extent.z <= sub_extent.z);
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
    extent: UVec3,
    offset: IVec3,
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

    fn extent(&self) -> UVec3 {
        self.extent
    }

    fn offset(&self) -> IVec3 {
        self.offset
    }

    fn format(&self) -> vk::Format {
        self.inner.format()
    }
}
pub trait ImageViewLike: ImageLike {
    fn raw_image_view(&self) -> vk::ImageView;
}

/// A regular image with 1 mip level and 1 layer fully backed by memory
pub struct Image {
    allocator: Allocator,
    image: vk::Image,
    allocation: vk_mem::Allocation,
    extent: UVec3,
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
                extent: UVec3::new(info.extent.width, info.extent.height, info.extent.depth),
                allocator,
                image,
                allocation,
            })
        }
    }
    pub fn extent(&self) -> UVec3 {
        self.extent
    }
    pub fn raw(&self) -> vk::Image {
        self.image
    }
}
impl ImageLike for Image {
    fn raw_image(&self) -> vk::Image {
        self.image
    }
    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
    fn extent(&self) -> UVec3 {
        self.extent
    }
    fn format(&self) -> vk::Format {
        vk::Format::R8G8B8A8_UNORM
    }
}
