use std::{sync::Arc, ops::Deref};

use ash::{vk, prelude::VkResult};
use bevy_app::{App, Plugin};
use bevy_ecs::prelude::*;
use bevy_window::{RawHandleWrapper, Window};

use crate::{plugin::RhyoliteApp, Device, utils::SharingMode, utils::ColorSpace, Surface, surface};

pub struct SwapchainPlugin {
    num_frame_in_flight: u32,
}

impl Default for SwapchainPlugin {
    fn default() -> Self {
        Self {
            num_frame_in_flight: 3,
        }
    }
}

impl Plugin for SwapchainPlugin {
    fn build(&self, app: &mut App) {
        app.add_device_extension(ash::extensions::khr::Swapchain::name())
            .unwrap();
    }
    fn finish(&self, app: &mut App) {
        let device: &Device = app.world.resource();
        let swapchain_loader = ash::extensions::khr::Swapchain::new(device.instance(), device);
        app.insert_resource(SwapchainLoader(Arc::new(SwapchainLoaderInner{
            device: device.clone(),
            loader: Arc::new(swapchain_loader),
        })));
    }
}

#[derive(Resource, Clone)]
pub struct SwapchainLoader(Arc<SwapchainLoaderInner>);
struct SwapchainLoaderInner {
    device: Device,
    loader: ash::extensions::khr::Swapchain,
}
impl Deref for SwapchainLoader {
    type Target = ash::extensions::khr::Swapchain;
    fn deref(&self) -> &Self::Target {
        &self.0.loader
    }
}

#[derive(Component)]
pub struct Swapchain {
    loader: SwapchainLoader,
    inner: vk::SwapchainKHR,

    images: Vec<(vk::Image, vk::ImageView)>,
    format: vk::Format,
    generation: u64,

    surface: Arc<Surface>,
    color_space: ColorSpace,
    extent: vk::Extent2D,
    layer_count: u32,
}
impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_swapchain(self.inner, None);
        }
    }
}

pub struct SwapchainCreateInfo<'a> {
    pub flags: vk::SwapchainCreateFlagsKHR,
    pub min_image_count: u32,
    pub image_format: vk::Format,
    pub image_color_space: vk::ColorSpaceKHR,
    pub image_extent: vk::Extent2D,
    pub image_array_layers: u32,
    pub image_usage: vk::ImageUsageFlags,
    pub image_sharing_mode: SharingMode<&'a [u32]>,
    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
    pub present_mode: vk::PresentModeKHR,
    pub clipped: bool,
}


/// Unsafe APIs for Swapchain
impl Swapchain {
    /// # Safety
    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCreateSwapchainKHR.html>
    pub fn create(
        surface: &Surface,
        device: &Device,
        info: SwapchainCreateInfo,
    ) -> VkResult<Self> {
        tracing::info!(
            width = %info.image_extent.width,
            height = %info.image_extent.height,
            format = ?info.image_format,
            color_space = ?info.image_color_space,
            "Creating swapchain"
        );
        unsafe {
            let mut create_info = vk::SwapchainCreateInfoKHR {
                flags: info.flags,
                surface: surface.raw(),
                min_image_count: info.min_image_count,
                image_format: info.image_format,
                image_color_space: info.image_color_space,
                image_extent: info.image_extent,
                image_array_layers: info.image_array_layers,
                image_usage: info.image_usage,
                image_sharing_mode: vk::SharingMode::EXCLUSIVE,
                pre_transform: info.pre_transform,
                composite_alpha: info.composite_alpha,
                present_mode: info.present_mode,
                clipped: info.clipped.into(),
                ..Default::default()
            };
            match &info.image_sharing_mode {
                SharingMode::Exclusive => (),
                SharingMode::Concurrent {
                    queue_family_indices,
                } => {
                    create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
                    create_info.p_queue_family_indices = queue_family_indices.as_ptr();
                }
            }
            let swapchain = device
                .swapchain_loader()
                .create_swapchain(&create_info, None)?;
            let images = get_swapchain_images(&device, swapchain, info.image_format)?;
            let inner = SwapchainInner {
                device,
                surface,
                swapchain,
                images,
                generation: 0,
                extent: info.image_extent,
                layer_count: info.image_array_layers,
                format: info.image_format,
                color_space: info.image_color_space.into(),
            };
            Ok(Self {
                inner: Arc::new(inner),
            })
        }
    }

    pub fn recreate(&mut self, info: SwapchainCreateInfo) -> VkResult<()> {
        tracing::info!(
            width = %info.image_extent.width,
            height = %info.image_extent.height,
            format = ?info.image_format,
            color_space = ?info.image_color_space,
            "Recreating swapchain"
        );
        unsafe {
            let mut create_info = vk::SwapchainCreateInfoKHR {
                flags: info.flags,
                surface: self.inner.surface.surface,
                min_image_count: info.min_image_count,
                image_format: info.image_format,
                image_color_space: info.image_color_space,
                image_extent: info.image_extent,
                image_array_layers: info.image_array_layers,
                image_usage: info.image_usage,
                image_sharing_mode: vk::SharingMode::EXCLUSIVE,
                pre_transform: info.pre_transform,
                composite_alpha: info.composite_alpha,
                present_mode: info.present_mode,
                clipped: info.clipped.into(),
                old_swapchain: self.inner.swapchain,
                ..Default::default()
            };
            match &info.image_sharing_mode {
                SharingMode::Exclusive => (),
                SharingMode::Concurrent {
                    queue_family_indices,
                } => {
                    create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
                    create_info.p_queue_family_indices = queue_family_indices.as_ptr();
                }
            }
            let swapchain = self
                .inner
                .device
                .swapchain_loader()
                .create_swapchain(&create_info, None)?;

            let images = get_swapchain_images(self.device(), swapchain, info.image_format)?;
            let inner = SwapchainInner {
                device: self.inner.device.clone(),
                surface: self.inner.surface.clone(),
                swapchain,
                images,
                generation: self.inner.generation.wrapping_add(1),
                extent: info.image_extent,
                layer_count: info.image_array_layers,
                format: info.image_format,
                color_space: info.image_color_space.into(),
            };
            self.inner = Arc::new(inner);
        }
        Ok(())
    }
    pub fn image_format(&self) -> vk::Format {
        self.inner.format
    }
    pub fn image_color_space(&self) -> ColorSpace {
        self.inner.color_space.clone()
    }
    pub fn image_extent(&self) -> vk::Extent2D {
        self.inner.extent
    }
}



#[derive(Component)]
pub struct SwapchainConfigExt {
    pub flags: vk::SwapchainCreateFlagsKHR,
    pub min_image_count: u32,
    /// If set to None, the implementation will select the best available color space.
    pub image_format: Option<vk::SurfaceFormatKHR>,
    pub image_array_layers: u32,
    pub image_usage: vk::ImageUsageFlags,
    pub required_feature_flags: vk::FormatFeatureFlags,
    pub sharing_mode: SharingMode,
    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    pub clipped: bool,

    /// If set to true and the `image_format` property weren't set,
    /// the implementation will select the best available HDR color space.
    /// On Windows 11, it is recommended to turn this off when the application was started in Windowed mode
    /// and the system HDR toggle was turned off. Otherwise, the screen may flash when the application is started.
    pub hdr: bool,

    /// If set to true, SDR swapchains will be created with a sRGB format.
    /// If set to false, SDR swapchains will be created with a UNORM format.
    ///
    /// Set this to false if the data will be written to the swapchain image as a storage image,
    /// and the tonemapper will apply gamma correction manually. This is the default.
    ///
    /// Set this to true if the swapchain will be directly used as a render target. In this case,
    /// the sRGB gamma correction will be applied automatically.
    pub srgb_format: bool,
}

pub(super) fn extract_swapchains(
    mut commands: Commands,
    device: Res<Device>,
    mut window_created_events: EventReader<bevy_window::WindowCreated>,
    mut window_resized_events: EventReader<bevy_window::WindowResized>,
    mut query: Query<(
        &Window,
        &RawHandleWrapper,
        Option<&SwapchainConfigExt>,
        Option<&mut Swapchain>,
        &Surface,
    )>,
) {
    for resize_event in window_resized_events.read() {
        let (window, _, config, swapchain, surface) = query.get_mut(resize_event.window).unwrap();
        if let Some(mut swapchain) = swapchain {
            let default_config = SwapchainConfigExt::default();
            let swapchain_config = config.unwrap_or(&default_config);
            swapchain.recreate(window, swapchain_config);
        }
    }
    for create_event in window_created_events.read() {
        let (window, raw_handle, config, swapchain, surface) = query.get(create_event.window).unwrap();
        let raw_handle = unsafe { raw_handle.get_handle() };
        assert!(swapchain.is_none());
        let default_config = SwapchainConfigExt::default();
        let swapchain_config = config.unwrap_or(&default_config);
        let new_swapchain = Swapchain::create(
            device.inner().clone(),
            new_surface,
            window,
            swapchain_config,
        );
        commands.entity(create_event.window).insert(new_swapchain);
    }
}

unsafe fn get_swapchain_images(
    device: &Device,
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
) -> VkResult<Vec<(vk::Image, vk::ImageView)>> {
    let images = device.swapchain_loader().get_swapchain_images(swapchain)?;
    let mut image_views: Vec<(vk::Image, vk::ImageView)> = Vec::with_capacity(images.len());
    for image in images.into_iter() {
        let view = device.create_image_view(
            &vk::ImageViewCreateInfo {
                image,
                view_type: vk::ImageViewType::TYPE_2D,
                format,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                },
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            },
            None,
        );
        match view {
            Ok(view) => image_views.push((image, view)),
            Err(err) => {
                // Destroy existing
                for (_image, view) in image_views.into_iter() {
                    device.destroy_image_view(view, None);
                }
                return Err(err);
            }
        }
    }
    Ok(image_views)
}

