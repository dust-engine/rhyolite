use std::{borrow::BorrowMut, ops::Deref, sync::Arc};

use ash::{extensions::khr, prelude::VkResult, vk};
use bevy_app::{App, Plugin, Update};
use bevy_ecs::{
    prelude::*,
    query::{QueryFilter, QuerySingleError},
};
use bevy_window::{PrimaryWindow, Window};

use crate::{
    ecs::{QueueContext, RenderImage, RenderSystemPass, RenderSystemsBinarySemaphoreTracker},
    plugin::RhyoliteApp,
    utils::{ColorSpace, SharingMode},
    Device, HasDevice, ImageLike, PhysicalDevice, QueueType, QueuesRouter, ResourceState, Surface,
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
        app.add_device_extension::<ash::extensions::khr::Swapchain>()
            .unwrap();

        app.add_systems(
            Update,
            (
                extract_swapchains.after(crate::surface::extract_surfaces),
                present.with_option::<RenderSystemPass>(|entry| {
                    let item = entry.or_default();
                    item.force_binary_semaphore = true;
                }),
                acquire_swapchain_image::<With<PrimaryWindow>>
                    .with_option::<RenderSystemPass>(|entry| {
                        let item = entry.or_default();
                        item.force_binary_semaphore = true;
                    })
                    .after(extract_swapchains)
                    .before(present),
            ),
        );
    }
}

#[derive(Component, Clone)]
pub struct Swapchain(Arc<SwapchainInner>);
impl PartialEq for Swapchain {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Swapchain {}

struct SwapchainInner {
    device: Device,
    surface: Surface,
    inner: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    generation: u64,

    color_space: ColorSpace,
    extent: vk::Extent2D,
    layer_count: u32,
}

impl Drop for SwapchainInner {
    fn drop(&mut self) {
        unsafe {
            self.device
                .extension::<khr::Swapchain>()
                .destroy_swapchain(self.inner, None);
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
    pub fn create(device: Device, surface: Surface, info: &SwapchainCreateInfo) -> VkResult<Self> {
        tracing::info!(
            width = %info.image_extent.width,
            height = %info.image_extent.height,
            format = ?info.image_format,
            color_space = ?info.image_color_space,
            "Creating swapchain"
        );
        let swapchain_loader = device.extension::<khr::Swapchain>();
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
            let images = swapchain_loader.get_swapchain_images(swapchain)?;
            let swapchain = SwapchainInner {
                device,
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
            let new_swapchain = self
                .0
                .device
                .extension::<khr::Swapchain>()
                .create_swapchain(&create_info, None)?;

            let images = self
                .0
                .device
                .extension::<khr::Swapchain>()
                .get_swapchain_images(new_swapchain)?;

            let inner = SwapchainInner {
                device: self.0.device.clone(),
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
        let new_swapchain =
            Swapchain::create(device.clone(), surface.clone(), &create_info).unwrap();
        commands
            .entity(create_event.window)
            .insert(new_swapchain)
            .insert(RenderImage::new(SwapchainImage {
                image: vk::Image::null(),
                indice: u32::MAX,
                swapchain: None,
            }));
    }
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
    pub image: vk::Image,
    indice: u32,
    swapchain: Option<Swapchain>,
}

impl ImageLike for SwapchainImage {
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

    fn extent(&self) -> vk::Extent3D {
        let swapchain = self
            .swapchain
            .as_ref()
            .expect("Swapchain image used before it was first acquired");
        vk::Extent3D {
            width: swapchain.0.extent.width,
            height: swapchain.0.extent.width,
            depth: 1,
        }
    }

    fn format(&self) -> vk::Format {
        let swapchain = self
            .swapchain
            .as_ref()
            .expect("Swapchain image used before it was first acquired");
        swapchain.0.format
    }
}

/// Acquires the next image from the swapchain by calling [`ash::extensions::khr::Swapchain::acquire_next_image`].
/// Generic parameter `Filter` is used to uniquely specify the swapchain to acquire from.
/// For example, `With<PrimaryWindow>` will only acquire the next image from the swapchain
/// associated with the primary window.
pub fn acquire_swapchain_image<Filter: QueryFilter>(
    mut queue_ctx: QueueContext<'g'>,
    mut query: Query<
        (
            &mut Swapchain, // Need mutable reference to swapchain to call acquire_next_image
            &mut RenderImage<SwapchainImage>,
        ),
        Filter,
    >,
    binary_semaphore_tracker: Res<RenderSystemsBinarySemaphoreTracker>,
) {
    assert!(queue_ctx.binary_waits.is_empty());
    assert!(queue_ctx.timeline_signals.is_empty());
    assert!(queue_ctx.timeline_waits.is_empty());
    assert!(queue_ctx.binary_signals.len() == 1, "Due to Vulkan constraints, you may not have more than two tasks dependent on the same swapchain acquire operation simultaneously.");

    let (mut swapchain, mut swapchain_image) = match query.get_single_mut() {
        Ok(item) => item,
        Err(QuerySingleError::NoEntities(_str)) => {
            return;
        }
        Err(QuerySingleError::MultipleEntities(str)) => {
            panic!("{}", str)
        }
    };
    let semaphore = queue_ctx.binary_signals[0].index;
    let semaphore = binary_semaphore_tracker.signal(semaphore);

    let (indice, suboptimal) = unsafe {
        // Technically, we don't have to do this here. With Swapchain_Maintenance1,
        // we could do this in the present workflow which should be more correct.
        // TODO: verify correctness.
        let fence = queue_ctx.fence_to_wait();
        let swapchain = swapchain.borrow_mut();
        swapchain
            .0
            .device
            .extension::<khr::Swapchain>()
            .acquire_next_image(swapchain.0.inner, !0, semaphore, fence)
    }
    .unwrap();
    if suboptimal {
        tracing::warn!("Suboptimal swapchain");
    }
    let image = swapchain.0.images[indice as usize];
    unsafe {
        swapchain_image.res.state = ResourceState::default();
        swapchain_image.res.state.write.stage = vk::PipelineStageFlags2::CLEAR;
        swapchain_image.layout = vk::ImageLayout::UNDEFINED;
        let swapchain_image = swapchain_image.get_mut();
        swapchain_image.image = image;
        swapchain_image.indice = indice;
        if swapchain_image.swapchain.as_ref() != Some(&*swapchain) {
            swapchain_image.swapchain = Some(swapchain.clone());
        }
    }
}

pub fn present(
    mut queue_ctx: QueueContext<'g'>,
    device: Res<Device>,
    queues_router: Res<QueuesRouter>,
    mut query: Query<(&mut Swapchain, &RenderImage<SwapchainImage>)>,
    binary_semaphore_tracker: Res<RenderSystemsBinarySemaphoreTracker>,
) {
    assert!(queue_ctx.timeline_signals.is_empty());
    assert!(queue_ctx.timeline_waits.is_empty());
    assert!(queue_ctx.binary_signals.is_empty());

    // TODO: this isn't exactly the best. Ideally we check surface-pdevice-queuefamily compatibility, then
    // select the best one.
    let present_queue = queues_router.of_type(QueueType::Graphics);
    let queue = device.get_raw_queue(present_queue);

    let mut swapchains: Vec<vk::SwapchainKHR> = Vec::new();
    let mut swapchain_image_indices: Vec<u32> = Vec::new();
    for (swapchain, swapchain_image) in query.iter_mut() {
        swapchains.push(swapchain.0.inner);
        // Safety: we're only getting the indice of the image and we're not actually reading / writing to it.
        swapchain_image_indices.push(swapchain_image.indice);
    }
    if swapchains.is_empty() {
        // A bit of a special case: we can't call `queue_present` with an empty swapchain list.
        // So, we call an empty `vkQueueSubmit` instead to reset all these semaphores and give
        // us an opportunity to wait on its completion.
        let mut semaphores: Vec<vk::SemaphoreSubmitInfo> = Vec::new();
        for op in queue_ctx.binary_waits.iter() {
            if let Some(semaphore) = binary_semaphore_tracker.wait(op.index) {
                semaphores.push(vk::SemaphoreSubmitInfo {
                    semaphore: semaphore.raw(),
                    stage_mask: op.access.stage,
                    ..Default::default()
                });
                queue_ctx.retained_objects.push(Box::new(semaphore));
            }
        }
        if !semaphores.is_empty() {
            unsafe {
                let fence = queue_ctx.fence_to_wait();
                device
                    .queue_submit2(
                        queue,
                        &[vk::SubmitInfo2 {
                            wait_semaphore_info_count: semaphores.len() as u32,
                            p_wait_semaphore_infos: semaphores.as_ptr(),
                            ..Default::default()
                        }],
                        fence,
                    )
                    .unwrap();
            }
        }
        return;
    }

    let semaphore_to_wait: Vec<vk::Semaphore> = queue_ctx
        .binary_waits
        .iter()
        .filter_map(|wait| {
            let semaphore = binary_semaphore_tracker.wait(wait.index)?;
            let raw = semaphore.raw();
            queue_ctx.retained_objects.push(Box::new(semaphore));
            Some(raw)
        })
        .collect();
    unsafe {
        device
            .extension::<khr::Swapchain>()
            .queue_present(
                queue,
                &vk::PresentInfoKHR {
                    swapchain_count: swapchains.len() as u32,
                    p_swapchains: swapchains.as_ptr(),
                    p_image_indices: swapchain_image_indices.as_ptr(),
                    p_wait_semaphores: semaphore_to_wait.as_ptr(),
                    wait_semaphore_count: semaphore_to_wait.len() as u32,
                    ..Default::default()
                },
            )
            .unwrap();
    }
}
