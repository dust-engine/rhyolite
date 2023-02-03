use ash::{prelude::VkResult, vk};
use std::sync::Arc;

use crate::{debug::DebugObject, Device, HasDevice};

pub trait ImageLike: Send + Sync {
    fn raw_image(&self) -> vk::Image;
    fn subresource_range(&self) -> vk::ImageSubresourceRange;
}

pub trait ImageViewLike: ImageLike {
    fn raw_image_view(&self) -> vk::ImageView;
    fn subresource_range(&self) -> vk::ImageSubresourceRange;
}

impl ImageLike for vk::Image {
    fn raw_image(&self) -> vk::Image {
        *self
    }
    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        unimplemented!("Raw image cannot be used here")
    }
}

pub struct Image {
    device: Arc<Device>,
    pub(crate) image: vk::Image,
    format: vk::Format,
}

impl HasDevice for Image {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl DebugObject for Image {
    fn object_handle(&mut self) -> u64 {
        unsafe { std::mem::transmute(self.image) }
    }

    const OBJECT_TYPE: vk::ObjectType = vk::ObjectType::IMAGE;
}

impl Image {
    pub fn new(device: Arc<Device>, info: &vk::ImageCreateInfo) -> VkResult<Self> {
        let image = unsafe { device.create_image(info, None)? };
        Ok(Self {
            device,
            image,
            format: info.format,
        })
    }
}

impl ImageLike for Image {
    fn raw_image(&self) -> vk::Image {
        self.image
    }
    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        let aspect_mask = match self.format {
            vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
                vk::ImageAspectFlags::DEPTH
            }
            vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT => {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            }
            _ => vk::ImageAspectFlags::COLOR,
        };
        vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        }
    }
}

impl<T: ImageLike> ImageLike for Arc<T> {
    fn raw_image(&self) -> vk::Image {
        let r: &T = self.as_ref();
        r.raw_image()
    }
    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        let r: &T = self.as_ref();
        r.subresource_range()
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        tracing::debug!(image = ?self.image, "drop image");
        unsafe { self.device.destroy_image(self.image, None) }
    }
}

/// Image bound to allocator memory
pub struct MemImage {
    allocator: Arc<Allocator>,
    pub image: vk::Image,
    pub memory: Allocation,
    pub memory_flags: vk::MemoryPropertyFlags,
    format: vk::Format,
}

impl ImageLike for MemImage {
    fn raw_image(&self) -> vk::Image {
        self.image
    }
    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        let aspect_mask = match self.format {
            vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
                vk::ImageAspectFlags::DEPTH
            }
            vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT => {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            }
            _ => vk::ImageAspectFlags::COLOR,
        };
        vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        }
    }
}

impl Drop for MemImage {
    fn drop(&mut self) {
        tracing::debug!(image = ?self.image, "drop mem image");
        unsafe {
            let mut memory: Allocation = std::mem::zeroed();
            std::mem::swap(&mut memory, &mut self.memory);
            self.allocator.allocator.destroy_image(self.image, memory);
        }
    }
}

#[derive(Clone)]
pub struct ImageRequest<'a> {
    pub scenario: MemoryAllocScenario,
    pub allocation_flags: AllocationCreateFlags,

    pub image_type: vk::ImageType,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: vk::SampleCountFlags,
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub sharing_mode: vk::SharingMode,
    pub queue_families: &'a [u32],
    pub initial_layout: vk::ImageLayout,
}
impl<'a> Default for ImageRequest<'a> {
    fn default() -> Self {
        Self {
            scenario: MemoryAllocScenario::DeviceAccess,
            allocation_flags: AllocationCreateFlags::empty(),

            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: vk::Extent3D::default(),
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::empty(),
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_families: &[],
            initial_layout: vk::ImageLayout::UNDEFINED,
        }
    }
}

impl Allocator {
    pub fn allocate_image(self: &Arc<Self>, image_request: &ImageRequest) -> VkResult<MemImage> {
        use vk_mem::Alloc;
        let build_info = vk::ImageCreateInfo {
            flags: vk::ImageCreateFlags::empty(),
            image_type: image_request.image_type,
            format: image_request.format,
            extent: image_request.extent,
            mip_levels: image_request.mip_levels,
            array_layers: image_request.array_layers,
            samples: image_request.samples,
            tiling: image_request.tiling,
            usage: image_request.usage,
            sharing_mode: image_request.sharing_mode,
            queue_family_index_count: image_request.queue_families.len() as u32,
            p_queue_family_indices: image_request.queue_families.as_ptr(),
            initial_layout: image_request.initial_layout,
            ..Default::default()
        };
        let create_info =
            self.create_info_by_scenario(image_request.allocation_flags, &image_request.scenario);
        let (image, allocation) =
            unsafe { self.allocator.create_image(&build_info, &create_info) }?;
        let memory_flags = unsafe {
            let allocation_info = self.allocator.get_allocation_info(&allocation).unwrap();
            let memory_flags = self.types[allocation_info.memory_type as usize].property_flags;
            memory_flags
        };
        Ok(MemImage {
            allocator: self.clone(),
            image,
            memory: allocation,
            memory_flags,
            format: image_request.format,
        })
    }
}

pub struct ImageView<T: ImageLike> {
    device: Arc<Device>,
    image: T,
    view: vk::ImageView,
}
impl<T: ImageLike> ImageView<T> {
    pub fn new(
        device: Arc<Device>,
        image: T,
        view_type: vk::ImageViewType,
        format: vk::Format,
        components: vk::ComponentMapping,
        subresource_range: vk::ImageSubresourceRange,
    ) -> VkResult<Self> {
        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo {
                    image: image.raw_image(),
                    view_type,
                    format,
                    components,
                    subresource_range,
                    ..Default::default()
                },
                None,
            )?
        };
        Ok(Self {
            device,
            image,
            view,
        })
    }
    pub fn raw_image_view(&self) -> vk::ImageView {
        self.view
    }
}
impl<T: ImageLike> Drop for ImageView<T> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);
        }
    }
}
