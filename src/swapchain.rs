use std::{borrow::BorrowMut, ops::Deref, sync::Arc};

use ash::{prelude::VkResult, vk};
use bevy_app::{App, Plugin, Update};
use bevy_ecs::prelude::*;
use bevy_window::Window;

use crate::{
    plugin::RhyoliteApp, utils::ColorSpace, utils::SharingMode, Device, HasDevice, PhysicalDevice,
    QueueType, QueuesRouter, Surface, ecs::{RenderSystemPass, QueueContext},
};

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

        app.add_systems(
            Update,
            (
                extract_swapchains.after(crate::surface::extract_surfaces),
                acquire_swapchain_image.with_option::<RenderSystemPass>(|entry| {
                    let item = entry.or_default();
                    item.force_binary_semaphore = true;
                }).after(extract_swapchains),
                present.with_option::<RenderSystemPass>(|entry| {
                    let item = entry.or_default();
                    item.force_binary_semaphore = true;
                }).after(acquire_swapchain_image),
            ),
        );
    }
    fn finish(&self, app: &mut App) {
        let device: &Device = app.world.resource();
        let swapchain_loader = ash::extensions::khr::Swapchain::new(device.instance(), device);
        app.insert_resource(SwapchainLoader(Arc::new(SwapchainLoaderInner {
            device: device.clone(),
            loader: swapchain_loader,
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
impl HasDevice for SwapchainLoader {
    fn device(&self) -> &Device {
        &self.0.device
    }
}

#[derive(Component, Clone)]
pub struct Swapchain(Arc<SwapchainInner>);
struct SwapchainInner {
    loader: SwapchainLoader,
    surface: Surface,
    inner: vk::SwapchainKHR,
    images: Vec<(vk::Image, vk::ImageView)>,
    format: vk::Format,
    generation: u64,

    color_space: ColorSpace,
    extent: vk::Extent2D,
    layer_count: u32,
}

impl Drop for SwapchainInner {
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
        swapchain_loader: SwapchainLoader,
        surface: Surface,
        _device: &Device,
        info: &SwapchainCreateInfo,
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
            let swapchain = swapchain_loader.create_swapchain(&create_info, None)?;
            let images = get_swapchain_images(&swapchain_loader, swapchain, info.image_format)?;
            let swapchain = SwapchainInner {
                loader: swapchain_loader,
                surface,
                inner: swapchain,
                images,
                generation: 0,
                extent: info.image_extent,
                layer_count: info.image_array_layers,
                format: info.image_format,
                color_space: info.image_color_space.into(),
            };
            Ok(Swapchain(Arc::new(swapchain)))
        }
    }

    pub fn recreate(&mut self, info: &SwapchainCreateInfo) -> VkResult<()> {
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
                surface: self.0.surface.raw(),
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
                old_swapchain: self.0.inner,
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
            let new_swapchain = self.0.loader.create_swapchain(&create_info, None)?;

            let images = get_swapchain_images(&self.0.loader, new_swapchain, info.image_format)?;

            let inner = SwapchainInner {
                loader: self.0.loader.clone(),
                surface: self.0.surface.clone(),
                inner: new_swapchain,
                images,
                generation: self.0.generation.wrapping_add(1),
                extent: info.image_extent,
                layer_count: info.image_array_layers,
                format: info.image_format,
                color_space: info.image_color_space.into(),
            };
            self.0 = Arc::new(inner);
        }
        Ok(())
    }
}

#[derive(Component)]
pub struct SwapchainConfig {
    pub flags: vk::SwapchainCreateFlagsKHR,
    pub min_image_count: u32,
    /// If set to None, the implementation will select the best available color space.
    pub image_format: Option<vk::SurfaceFormatKHR>,
    pub image_array_layers: u32,
    pub image_usage: vk::ImageUsageFlags,
    pub required_feature_flags: vk::FormatFeatureFlags,
    pub sharing_mode: SharingMode<Vec<u32>>,
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
impl Default for SwapchainConfig {
    fn default() -> Self {
        Self {
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            min_image_count: 3,
            image_format: None,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            required_feature_flags: vk::FormatFeatureFlags::empty(),
            sharing_mode: SharingMode::Exclusive,
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            clipped: true,
            hdr: false,
            srgb_format: false,
        }
    }
}

pub(super) fn extract_swapchains(
    mut commands: Commands,
    device: Res<Device>,
    swapchain_loader: Res<SwapchainLoader>,
    mut window_created_events: EventReader<bevy_window::WindowCreated>,
    mut window_resized_events: EventReader<bevy_window::WindowResized>,
    mut query: Query<(
        &Window,
        Option<&SwapchainConfig>,
        Option<&mut Swapchain>,
        &Surface,
    )>,
) {
    for resize_event in window_resized_events.read() {
        let (window, config, swapchain, surface) = query.get_mut(resize_event.window).unwrap();
        if let Some(mut swapchain) = swapchain {
            let default_config = SwapchainConfig::default();
            let config = config.unwrap_or(&default_config);
            let create_info = get_create_info(surface, device.physical_device(), window, config);
            swapchain.recreate(&create_info).unwrap();
        }
    }
    for create_event in window_created_events.read() {
        let (window, config, swapchain, surface) = query.get(create_event.window).unwrap();
        assert!(swapchain.is_none());
        let default_config = SwapchainConfig::default();
        let swapchain_config = config.unwrap_or(&default_config);
        let create_info =
            get_create_info(surface, device.physical_device(), window, swapchain_config);
        let new_swapchain = Swapchain::create(
            swapchain_loader.clone(),
            surface.clone(),
            device.deref(),
            &create_info,
        )
        .unwrap();
        commands
            .entity(create_event.window)
            .insert(new_swapchain)
            .insert(SwapchainImage {
                image: vk::Image::null(),
                full_image_view: vk::ImageView::null(),
                indice: u32::MAX,
            });
    }
}

unsafe fn get_swapchain_images(
    loader: &SwapchainLoader,
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
) -> VkResult<Vec<(vk::Image, vk::ImageView)>> {
    let images = loader.get_swapchain_images(swapchain)?;
    let mut image_views: Vec<(vk::Image, vk::ImageView)> = Vec::with_capacity(images.len());
    for image in images.into_iter() {
        let view = loader.device().create_image_view(
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
                    loader.device().destroy_image_view(view, None);
                }
                return Err(err);
            }
        }
    }
    Ok(image_views)
}

fn get_create_info<'a>(
    surface: &'_ Surface,
    pdevice: &'_ PhysicalDevice,
    window: &'_ Window,
    config: &'a SwapchainConfig,
) -> SwapchainCreateInfo<'a> {
    let surface_capabilities = pdevice.get_surface_capabilities(surface).unwrap();
    let supported_present_modes = pdevice.get_surface_present_modes(surface).unwrap();
    let image_format = config.image_format.unwrap_or_else(|| {
        if config.hdr {
            get_surface_preferred_format(
                surface,
                pdevice,
                config.required_feature_flags,
                config.srgb_format,
            )
        } else {
            if config.srgb_format {
                vk::SurfaceFormatKHR {
                    format: vk::Format::B8G8R8A8_SRGB,
                    color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                }
            } else {
                vk::SurfaceFormatKHR {
                    format: vk::Format::B8G8R8A8_UNORM,
                    color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                }
            }
        }
    });
    SwapchainCreateInfo {
        flags: config.flags,
        min_image_count: config.min_image_count,
        image_format: image_format.format,
        image_color_space: image_format.color_space,
        image_extent: vk::Extent2D {
            width: window.resolution.physical_width(),
            height: window.resolution.physical_height(),
        },
        image_array_layers: config.image_array_layers,
        image_usage: config.image_usage,
        image_sharing_mode: match &config.sharing_mode {
            SharingMode::Exclusive => SharingMode::Exclusive,
            SharingMode::Concurrent {
                queue_family_indices,
            } => SharingMode::Concurrent {
                queue_family_indices: &queue_family_indices,
            },
        },
        pre_transform: config.pre_transform,
        composite_alpha: match window.composite_alpha_mode {
            bevy_window::CompositeAlphaMode::Auto => {
                if surface_capabilities
                    .supported_composite_alpha
                    .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
                {
                    vk::CompositeAlphaFlagsKHR::OPAQUE
                } else {
                    vk::CompositeAlphaFlagsKHR::INHERIT
                }
            }
            bevy_window::CompositeAlphaMode::Opaque => vk::CompositeAlphaFlagsKHR::OPAQUE,
            bevy_window::CompositeAlphaMode::PreMultiplied => {
                vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
            }
            bevy_window::CompositeAlphaMode::PostMultiplied => {
                vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED
            }
            bevy_window::CompositeAlphaMode::Inherit => vk::CompositeAlphaFlagsKHR::INHERIT,
        },
        present_mode: match window.present_mode {
            bevy_window::PresentMode::AutoVsync => {
                if supported_present_modes.contains(&vk::PresentModeKHR::FIFO_RELAXED) {
                    vk::PresentModeKHR::FIFO_RELAXED
                } else {
                    vk::PresentModeKHR::FIFO
                }
            }
            bevy_window::PresentMode::AutoNoVsync => {
                if supported_present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                    vk::PresentModeKHR::IMMEDIATE
                } else if supported_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                    vk::PresentModeKHR::MAILBOX
                } else {
                    vk::PresentModeKHR::FIFO
                }
            }
            bevy_window::PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
            bevy_window::PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
            bevy_window::PresentMode::Fifo => vk::PresentModeKHR::FIFO,
            bevy_window::PresentMode::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
        },
        clipped: config.clipped,
    }
}

fn get_surface_preferred_format(
    surface: &Surface,
    physical_device: &PhysicalDevice,
    required_feature_flags: vk::FormatFeatureFlags,
    use_srgb_format: bool,
) -> vk::SurfaceFormatKHR {
    use crate::utils::{ColorSpacePrimaries, ColorSpaceTransferFunction, Format, FormatType};
    let supported_formats = physical_device.get_surface_formats(surface).unwrap();

    supported_formats
        .iter()
        .filter(|&surface_format| {
            let format_properties = unsafe {
                physical_device
                    .instance()
                    .get_physical_device_format_properties(
                        physical_device.raw(),
                        surface_format.format,
                    )
            };
            format_properties
                .optimal_tiling_features
                .contains(required_feature_flags)
                | format_properties
                    .linear_tiling_features
                    .contains(required_feature_flags)
        })
        .max_by_key(|&surface_format| {
            // Select color spaces based on the following criteria:
            // Prefer larger color spaces. For extended srgb, consider it the same as Rec2020 but after all other Rec2020 color spaces.
            // Prefer formats with larger color depth.
            // If small swapchain format, prefer non-linear. Otherwise, prefer linear.
            let format: Format = surface_format.format.into();
            let format_priority = format.r + format.g + format.b;

            let color_space: ColorSpace = surface_format.color_space.into();

            let color_space_priority = if matches!(
                color_space.transfer_function,
                ColorSpaceTransferFunction::scRGB
            ) {
                // Special case for extended srgb
                (ColorSpacePrimaries::BT2020.area_size() * 4096.0) as u32 - 1
            } else {
                (color_space.primaries.area_size() * 4096.0) as u32
            };
            let linearity_priority: u8 = if format_priority <= 30 {
                // <= 10 bit color. Prefer non-linear color space
                if color_space.transfer_function.is_linear() {
                    0
                } else {
                    1
                }
            } else {
                // > 10 bit color, for example 10 bit color and above. Prefer linear color space
                if color_space.transfer_function.is_linear() {
                    1
                } else {
                    0
                }
            };

            let srgb_priority = if (format.ty == FormatType::sRGB) ^ use_srgb_format {
                0_u8
            } else {
                1_u8
            };
            (
                color_space_priority,
                format_priority,
                linearity_priority,
                srgb_priority,
            )
        })
        .cloned()
        .unwrap_or(vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        })
}

#[derive(Component)]
pub struct SwapchainImage {
    image: vk::Image,
    full_image_view: vk::ImageView,
    indice: u32,
}

pub fn acquire_swapchain_image(
    queue_ctx: QueueContext<'g'>,
    mut query: Query<(
        &mut Swapchain, // Need mutable reference to swapchain to call acquire_next_image
        &mut SwapchainImage,
    )>,
) {
    println!("acquire {:?}", queue_ctx.queue);

    for (mut swapchain, mut swapchain_image) in query.iter_mut() {
        let (indice, suboptimal) = unsafe {
            let swapchain = swapchain.borrow_mut();
            swapchain.0.loader.acquire_next_image(
                swapchain.0.inner,
                !0,
                vk::Semaphore::null(),
                vk::Fence::null(),
            )
        }
        .unwrap();
        if suboptimal {
            tracing::warn!("Suboptimal swapchain");
        }
        let (image, image_view) = swapchain.0.images[indice as usize];
        *swapchain_image = SwapchainImage {
            image,
            full_image_view: image_view,
            indice,
        };
    }
}

pub fn present(
    queue_ctx: QueueContext<'g'>,
    swapchain_loader: Res<SwapchainLoader>,
    device: Res<Device>,
    queues_router: Res<QueuesRouter>,
    mut query: Query<(&mut Swapchain, &SwapchainImage)>,
) {
    println!("present {:?}", queue_ctx.queue);
    let mut swapchains: Vec<vk::SwapchainKHR> = Vec::new();
    let mut swapchain_image_indices: Vec<u32> = Vec::new();
    for (swapchain, swapchain_image) in query.iter_mut() {
        swapchains.push(swapchain.0.inner);
        swapchain_image_indices.push(swapchain_image.indice);
    }
    if swapchains.is_empty() {
        tracing::warn!("Nothing to present!");
        return;
    }

    // TODO: this isn't exactly the best. Ideally we check surface-pdevice-queuefamily compatibility, then
    // select the best one.
    let present_queue = queues_router.of_type(QueueType::Graphics);
    let queue = device.get_raw_queue(present_queue);

    unsafe {
        swapchain_loader
            .queue_present(
                queue,
                &vk::PresentInfoKHR {
                    swapchain_count: swapchains.len() as u32,
                    p_swapchains: swapchains.as_ptr(),
                    p_image_indices: swapchain_image_indices.as_ptr(),
                    ..Default::default()
                },
            )
            .unwrap();
    }
}
