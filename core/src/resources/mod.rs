mod buffer;
use std::ops::Deref;

//mod image;
use ash::vk;
pub use buffer::*;
//pub use image::*;
pub trait ImageLike {
    fn raw_image(&self) -> vk::Image;
    fn subresource_range(&self) -> vk::ImageSubresourceRange;
}
impl<I: ImageLike, T: Deref<Target = I>> ImageLike for T {
    fn raw_image(&self) -> vk::Image {
        self.deref().raw_image()
    }
    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        self.deref().subresource_range()
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
