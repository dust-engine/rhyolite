use std::ops::Deref;

use ash::{prelude::VkResult, vk};

use crate::{HasDevice, ImageLike};

pub trait ImageViewLike: ImageLike {
    fn raw_image_view(&self) -> vk::ImageView;
}

pub struct ImageView<T: ImageLike> {
    image: T,
    view: vk::ImageView,
}
impl<T: ImageLike> HasDevice for ImageView<T> {
    fn device(&self) -> &std::sync::Arc<crate::Device> {
        self.image.device()
    }
}
impl<T: ImageLike> ImageLike for ImageView<T> {
    fn raw_image(&self) -> vk::Image {
        self.image.raw_image()
    }

    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        self.image.subresource_range()
    }

    fn extent(&self) -> vk::Extent3D {
        self.image.extent()
    }

    fn format(&self) -> vk::Format {
        self.image.format()
    }
    fn offset(&self) -> vk::Offset3D {
        self.image.offset()
    }
}
impl<T: ImageLike> ImageViewLike for ImageView<T> {
    fn raw_image_view(&self) -> vk::ImageView {
        self.view
    }
}

impl<T: ImageLike> ImageView<T> {
    pub fn new(image: T, ty: vk::ImageViewType) -> VkResult<Self> {
        let view = unsafe {
            image.device().create_image_view(
                &vk::ImageViewCreateInfo {
                    image: image.raw_image(),
                    view_type: ty,
                    format: image.format(),
                    components: vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    },
                    subresource_range: image.subresource_range(),
                    ..Default::default()
                },
                None,
            )
        }?;
        Ok(Self { view, image })
    }
}

// This is in fact very questionable
impl<I: ImageViewLike, T: Deref<Target = I> + ImageLike> ImageViewLike for T {
    fn raw_image_view(&self) -> vk::ImageView {
        self.deref().raw_image_view()
    }
}
