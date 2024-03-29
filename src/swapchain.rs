use std::{
    borrow::BorrowMut,
    collections::BTreeSet,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use ash::{extensions::khr, prelude::VkResult, vk};
use bevy::{app::{App, Plugin, PostUpdate}, utils::smallvec::SmallVec};
use bevy::ecs::{
    prelude::*,
    query::{QueryFilter, QuerySingleError},
};
use bevy::window::{PrimaryWindow, Window};

use crate::{
    commands::{ResourceTransitionCommands, TrackedResource},
    ecs::{
        Barriers, IntoRenderSystemConfigs, QueueContext, RenderImage, RenderSystemPass,
    },
    plugin::RhyoliteApp,
    utils::{ColorSpace, SharingMode},
    Access, Device, HasDevice, ImageLike, ImageViewLike, PhysicalDevice, QueueType, QueuesRouter,
    Surface,
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
            PostUpdate,
            (
                extract_swapchains.after(crate::surface::extract_surfaces),
                present
                    .with_option::<RenderSystemPass>(|entry| {
                        let item = entry.or_default();
                        item.force_binary_semaphore = true;
                    })
                    .with_barriers(present_barriers),
                acquire_swapchain_image::<With<PrimaryWindow>>
                    .with_option::<RenderSystemPass>(|entry| {
                        let item = entry.or_default();
                        item.force_binary_semaphore = true;
                    })
                    .after(extract_swapchains)
                    .before(present),
            ),
        );

        app.add_event::<SuboptimalEvent>();
    }
}

#[derive(Component)]
pub struct Swapchain {
    inner: Arc<SwapchainInner>,
    acquire_semaphore: vk::Semaphore,
    images: SmallVec<[Option<RenderImage<SwapchainImageInner>>; 3]>,
}

struct SwapchainInner {
    device: Device,
    surface: Surface,
    inner: vk::SwapchainKHR,
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
                device: device.clone(),
                surface,
                inner: swapchain,
                generation: 0,
                extent: info.image_extent,
                layer_count: info.image_array_layers,
                format: info.image_format,
                color_space: info.image_color_space.into(),
            };
            let inner = Arc::new(swapchain);
            Ok(Swapchain {
                images: images
                    .into_iter()
                    .enumerate()
                    .map(|(i, image)| {
                        let view = device.create_image_view(
                            &vk::ImageViewCreateInfo {
                                image,
                                view_type: vk::ImageViewType::TYPE_2D,
                                format: info.image_format,
                                components: vk::ComponentMapping::default(),
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
                        )?;
                        let present_semaphore = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap();
                        let acquire_semaphore = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap();
                        let mut img = RenderImage::new(SwapchainImageInner {
                            image,
                            indice: i as u32,
                            swapchain: inner.clone(),
                            view,
                            present_semaphore,
                            acquire_semaphore,
                        });
                        img.res.state.read.stage = vk::PipelineStageFlags2::BOTTOM_OF_PIPE;
                        img.res.state.write.stage = vk::PipelineStageFlags2::BOTTOM_OF_PIPE;
                        Ok(Some(img))
                    })
                    .collect::<VkResult<Vec<Option<RenderImage<SwapchainImageInner>>>>>()?.into(),
                inner,
                // Create one extra semaphore for the first acquire
                acquire_semaphore: device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap(),
            })
        }
    }

    pub fn recreate(&mut self, info: &SwapchainCreateInfo) -> VkResult<()> {
        panic!();
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
                surface: self.inner.surface.raw(),
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
                old_swapchain: self.inner.inner,
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
                .inner
                .device
                .extension::<khr::Swapchain>()
                .create_swapchain(&create_info, None)?;

            let images = self
                .inner
                .device
                .extension::<khr::Swapchain>()
                .get_swapchain_images(new_swapchain)?;

            let inner = SwapchainInner {
                device: self.inner.device.clone(),
                surface: self.inner.surface.clone(),
                inner: new_swapchain,
                generation: self.inner.generation.wrapping_add(1),
                extent: info.image_extent,
                layer_count: info.image_array_layers,
                format: info.image_format,
                color_space: info.image_color_space.into(),
            };
            self.inner = Arc::new(inner);
            self.images = images
                .into_iter()
                .enumerate()
                .map(|(i, image)| {
                    let view = self.inner.device.create_image_view(
                        &vk::ImageViewCreateInfo {
                            image,
                            view_type: vk::ImageViewType::TYPE_2D,
                            format: self.inner.format,
                            components: vk::ComponentMapping::default(),
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
                    )?;
                    let mut img = RenderImage::new(SwapchainImageInner {
                        image,
                        indice: i as u32,
                        swapchain: self.inner.clone(),
                        view,
                        present_semaphore: vk::Semaphore::null(),
                        acquire_semaphore: vk::Semaphore::null(),
                    });
                    img.res.state.read.stage = vk::PipelineStageFlags2::BOTTOM_OF_PIPE;
                    img.res.state.write.stage = vk::PipelineStageFlags2::BOTTOM_OF_PIPE;
                    Ok(Some(img))
                })
                .collect::<VkResult<Vec<Option<RenderImage<SwapchainImageInner>>>>>()?.into();
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
    /// and the tonemapper will apply gamma correction manually.
    ///
    /// Set this to true if the swapchain will be directly used as a render target. In this case,
    /// the sRGB gamma correction will be applied automatically. This is the default.
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
            srgb_format: true,
        }
    }
}

#[derive(Event)]
pub struct SuboptimalEvent {
    window: Entity,
}

pub(super) fn extract_swapchains(
    mut commands: Commands,
    device: Res<Device>,
    mut window_created_events: EventReader<bevy::window::WindowCreated>,
    mut window_resized_events: EventReader<bevy::window::WindowResized>,
    mut suboptimal_events: EventReader<SuboptimalEvent>,
    mut query: Query<(
        &Window,
        Option<&SwapchainConfig>,
        Option<&mut Swapchain>,
        &Surface,
    )>,
    #[cfg(any(target_os = "macos", target_os = "ios"))] _marker: Option<
        NonSend<bevy::core::NonSendMarker>,
    >,
) {
    let mut windows_to_rebuild: BTreeSet<Entity> = BTreeSet::new();
    windows_to_rebuild.extend(window_resized_events.read().map(|a| a.window));
    windows_to_rebuild.extend(suboptimal_events.read().map(|a| a.window));
    for resized_window in windows_to_rebuild.into_iter() {
        let (window, config, swapchain, surface) = query.get_mut(resized_window).unwrap();
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
            .insert(SwapchainImage {
                inner: None,
                acquire_semaphore: vk::Semaphore::null(),
                present_semaphore: vk::Semaphore::null(),
            });
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
        min_image_count: config
            .min_image_count
            .max(surface_capabilities.min_image_count)
            .min({
                if surface_capabilities.max_image_count == 0 {
                    u32::MAX
                } else {
                    surface_capabilities.max_image_count
                }
            }),
        image_format: image_format.format,
        image_color_space: image_format.color_space,
        image_extent: surface_capabilities.current_extent,
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
            bevy::window::CompositeAlphaMode::Auto => {
                if surface_capabilities
                    .supported_composite_alpha
                    .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
                {
                    vk::CompositeAlphaFlagsKHR::OPAQUE
                } else {
                    vk::CompositeAlphaFlagsKHR::INHERIT
                }
            }
            bevy::window::CompositeAlphaMode::Opaque => vk::CompositeAlphaFlagsKHR::OPAQUE,
            bevy::window::CompositeAlphaMode::PreMultiplied => {
                vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
            }
            bevy::window::CompositeAlphaMode::PostMultiplied => {
                vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED
            }
            bevy::window::CompositeAlphaMode::Inherit => vk::CompositeAlphaFlagsKHR::INHERIT,
        },
        present_mode: match window.present_mode {
            bevy::window::PresentMode::AutoVsync => {
                if supported_present_modes.contains(&vk::PresentModeKHR::FIFO_RELAXED) {
                    vk::PresentModeKHR::FIFO_RELAXED
                } else {
                    vk::PresentModeKHR::FIFO
                }
            }
            bevy::window::PresentMode::AutoNoVsync => {
                if supported_present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                    vk::PresentModeKHR::IMMEDIATE
                } else if supported_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                    vk::PresentModeKHR::MAILBOX
                } else {
                    vk::PresentModeKHR::FIFO
                }
            }
            bevy::window::PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
            bevy::window::PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
            bevy::window::PresentMode::Fifo => vk::PresentModeKHR::FIFO,
            bevy::window::PresentMode::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
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
    inner: Option<RenderImage<SwapchainImageInner>>,
    
    acquire_semaphore: vk::Semaphore,
    present_semaphore: vk::Semaphore,
}
impl TrackedResource for SwapchainImage {
    type State = vk::ImageLayout;


    fn transition(&mut self, access: Access, retain_data: bool, next_state: Self::State, commands: &mut impl ResourceTransitionCommands) {
        let image = self.inner.as_mut().expect("SwapchainAcquire must have been called");

        if image.res.state.write_semaphore.is_none() || image.res.state.read_semaphores.is_empty() {
            assert!(self.acquire_semaphore != vk::Semaphore::null());
            commands.wait_binary_semaphore(std::mem::take(&mut self.acquire_semaphore), access.stage);
        }
        <RenderImage::<SwapchainImageInner> as TrackedResource>::transition(image, access, retain_data, next_state, commands);
    }
}
impl Deref for SwapchainImage {
    type Target = SwapchainImageInner;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref().expect("SwapchainAcquire must have been called")
    }
}

pub struct SwapchainImageInner {
    pub image: vk::Image,
    view: vk::ImageView,
    indice: u32,
    swapchain: Arc<SwapchainInner>,

    acquire_semaphore: vk::Semaphore,
    present_semaphore: vk::Semaphore,
}
impl Drop for SwapchainImageInner {
    fn drop(&mut self) {
        unsafe {
            self.swapchain.device.destroy_image_view(self.view, None);
        }
    }
}

impl ImageLike for SwapchainImageInner {
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
        vk::Extent3D {
            width: self.swapchain.extent.width,
            height: self.swapchain.extent.height,
            depth: 1,
        }
    }

    fn format(&self) -> vk::Format {
        self.swapchain.format
    }
}
impl ImageViewLike for SwapchainImageInner {
    fn raw_image_view(&self) -> vk::ImageView {
        self.view
    }
}

/// Acquires the next image from the swapchain by calling [`ash::extensions::khr::Swapchain::acquire_next_image`].
/// Generic parameter `Filter` is used to uniquely specify the swapchain to acquire from.
/// For example, `With<PrimaryWindow>` will only acquire the next image from the swapchain
/// associated with the primary window.
pub fn acquire_swapchain_image<Filter: QueryFilter>(
    mut queue_ctx: QueueContext,
    mut query: Query<
        (
            Entity,
            &mut Swapchain, // Need mutable reference to swapchain to call acquire_next_image
            &mut SwapchainImage,
        ),
        Filter,
    >,
    mut suboptimal_events: EventWriter<SuboptimalEvent>,
) {
    let (entity, mut swapchain, mut swapchain_image) = match query.get_single_mut() {
        Ok(item) => item,
        Err(QuerySingleError::NoEntities(_str)) => {
            return;
        }
        Err(QuerySingleError::MultipleEntities(str)) => {
            panic!("{}", str)
        }
    };

    let (indice, suboptimal) = unsafe {
        // Technically, we don't have to do this here. With Swapchain_Maintenance1,
        // we could do this in the present workflow which should be more correct.
        // TODO: verify correctness.
        let fence = queue_ctx.fence_to_wait();
        swapchain
            .inner
            .device
            .extension::<khr::Swapchain>()
            .acquire_next_image(swapchain.inner.inner, !0, swapchain.acquire_semaphore, fence)
    }
    .unwrap();
    if suboptimal {
        tracing::warn!("Suboptimal swapchain");
        suboptimal_events.send(SuboptimalEvent { window: entity });
    }
    let mut image = swapchain.images[indice as usize]
        .take()
        .expect("Acquiring image that hasn't been presented");
    assert!(
        swapchain_image.inner.is_none(),
        "Must present the current image before acquiring a new one"
    );
    std::mem::swap(&mut swapchain.acquire_semaphore, &mut unsafe{image.get_mut()}.acquire_semaphore);
    swapchain_image.acquire_semaphore = image.acquire_semaphore;
    swapchain_image.present_semaphore = image.present_semaphore;
    swapchain_image.inner = Some(image);
}

fn present_barriers(
    In(mut barriers): In<Barriers>,
    mut query: Query<&mut SwapchainImage>,
    queues_router: Res<QueuesRouter>
) {
    for i in query.iter_mut() {
        let i = i.into_inner();
        let image = i.inner.as_mut().unwrap();
        let barrier = image.res.state.transition(
            Access {
                stage: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                access: vk::AccessFlags2::empty(),
            },
            true,
        );
        if let Some(prev_queue) = image.res.state.queue_family {
            barriers.add_image_barrier_prev_stage(
                vk::ImageMemoryBarrier2 {
                    src_stage_mask: barrier.src_stage_mask,
                    src_access_mask: barrier.src_access_mask,
                    dst_stage_mask: barrier.dst_stage_mask,
                    dst_access_mask: barrier.dst_access_mask,
                    old_layout: image.layout,
                    new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    image: image.raw_image(),
                    subresource_range: image.subresource_range(),
                    ..Default::default()
                },
                prev_queue,
            );
            barriers.signal_binary_semaphore_prev_stage(
                i.present_semaphore,
                barrier.dst_stage_mask,
                prev_queue,
            );
        } else {
            // image was never accessed before! that means nothing within the current frame wrote to the swapchain image.
            // the present operation still expects the image to be in PRESENT layout tho. So we need to transition its layout.
            // Here we simply put the transition on the graphics queue. The graphics queue submission will now wait for the
            // acquire semaphore of the swapchain image.
            barriers.add_image_barrier_prev_stage(
                vk::ImageMemoryBarrier2 {
                    src_stage_mask: barrier.src_stage_mask,
                    src_access_mask: barrier.src_access_mask,
                    dst_stage_mask: barrier.dst_stage_mask,
                    dst_access_mask: barrier.dst_access_mask,
                    old_layout: image.layout,
                    new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    image: image.raw_image(),
                    subresource_range: image.subresource_range(),
                    ..Default::default()
                },
                queues_router.of_type(QueueType::Graphics),
            );
            barriers.signal_binary_semaphore_prev_stage(
                i.present_semaphore,
                barrier.dst_stage_mask,
                queues_router.of_type(QueueType::Graphics),
            );
            // This previous queue now needs to wait for the swapchain acquire semaphore, since nobody else is waiting on it.
            barriers.wait_binary_semaphore_prev_stage(
                i.acquire_semaphore,
                barrier.src_stage_mask,
                queues_router.of_type(QueueType::Graphics),
            );
        }
        image.layout = vk::ImageLayout::PRESENT_SRC_KHR;
        image.res.state.queue_family = None;
    }
}
pub fn present(
    mut queue_ctx: QueueContext,
    device: Res<Device>,
    queues_router: Res<QueuesRouter>,
    mut query: Query<(&mut Swapchain, &mut SwapchainImage)>,
) {
    assert!(queue_ctx.binary_signals.is_empty());

    // TODO: this isn't exactly the best. Ideally we check surface-pdevice-queuefamily compatibility, then
    // select the best one.
    let present_queue = queues_router.of_type(QueueType::Graphics);
    let queue = device.get_raw_queue(present_queue);

    let mut swapchains: Vec<vk::SwapchainKHR> = Vec::new();
    let mut semaphores: Vec<vk::Semaphore> = Vec::new();
    let mut swapchain_image_indices: Vec<u32> = Vec::new();
    for (mut swapchain, mut swapchain_image) in query.iter_mut() {
        let Some(swapchain_image) = swapchain_image.inner.take() else {
            continue;
        };
        swapchains.push(swapchain.inner.inner);
        semaphores.push(swapchain_image.present_semaphore);
        // Safety: we're only getting the indice of the image and we're not actually reading / writing to it.
        let indice = swapchain_image.indice;
        swapchain_image_indices.push(indice);
        swapchain.images[indice as usize] = Some(swapchain_image);
    }
    if swapchains.is_empty() {
        return;
    }

    unsafe {
        device
            .extension::<khr::Swapchain>()
            .queue_present(
                queue,
                &vk::PresentInfoKHR {
                    swapchain_count: swapchains.len() as u32,
                    p_swapchains: swapchains.as_ptr(),
                    p_image_indices: swapchain_image_indices.as_ptr(),
                    p_wait_semaphores: semaphores.as_ptr(),
                    wait_semaphore_count: semaphores.len() as u32,
                    ..Default::default()
                },
            )
            .unwrap();
    }
}
