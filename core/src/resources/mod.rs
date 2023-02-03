mod buffer;
//mod image;
use ash::vk;
pub use buffer::*;
//pub use image::*;
pub trait ImageLike: Send + Sync {
    fn raw_image(&self) -> vk::Image;
    fn subresource_range(&self) -> vk::ImageSubresourceRange;
}
