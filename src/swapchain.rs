use ash::vk::Handle;
use bevy::ecs::component::ComponentId;
use cstr::cstr;
use std::ops::DerefMut;
use std::usize;
use std::{collections::BTreeSet, ops::Deref, sync::Arc};

use ash::{prelude::VkResult, vk};
use bevy::app::{App, Plugin, PostUpdate};
use bevy::window::{PrimaryWindow, Window};
use bevy::{
    ecs::{
        prelude::*,
        query::{QueryFilter, QuerySingleError},
    },
    math::{UVec2, UVec3},
};
use smallvec::SmallVec;

use crate::command::QueueDependency;
use crate::ecs2::{IntoRenderSystem, QueueSystemCtx};
use crate::future::{GPUResource, ResourceState};
use crate::selectors::Graphics;
use crate::{
    plugin::RhyoliteApp,
    utils::{ColorSpace, SharingMode},
    Device, ImageLike, ImageViewLike, PhysicalDevice, Surface,
};
use crate::{HasDevice, Queue};
use ash::ext::swapchain_maintenance1::Meta as ExtSwapchainMaintenance1;
use ash::khr::swapchain::Meta as KhrSwapchain;
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

/// A special system set for render systems requiring access to the swapchain image.
/// Render systems added to this system set are guaranteed to run before `present` and after `acquire_next_image`.
/// These systems will also be properly syncronized with the acquire and present queue systems with semaphores.
#[derive(SystemSet, Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct SwapchainSystemSet;
impl Plugin for SwapchainPlugin {
    fn build(&self, app: &mut App) {
        app.add_device_extension::<KhrSwapchain>().unwrap();
        app.add_device_extension::<ExtSwapchainMaintenance1>()
            .unwrap();
        app.enable_feature::<vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT>(|x| {
            &mut x.swapchain_maintenance1
        })
        .unwrap();

        app.add_systems(
            PostUpdate,
            (
                extract_swapchains.after(crate::surface::extract_surfaces),
                // Always present on the Graphics queue.
                // TODO: this isn't exactly the best. Ideally we check surface-pdevice-queuefamily compatibility
                // using vkGetPhysicalDeviceSurfaceSupportKHR select the best one.
                present.into_render_system::<Graphics>(),
                acquire_swapchain_image::<With<PrimaryWindow>>
                    // acquire_swapchain_image isn't exactly a queue operation.
                    .into_render_system::<Graphics>()
                    .after(extract_swapchains)
                    .before(present),
            ),
        );
        app.configure_sets(
            PostUpdate,
            SwapchainSystemSet
                .before(present)
                .after(acquire_swapchain_image::<With<PrimaryWindow>>),
        );

        app.add_event::<SuboptimalEvent>();
    }
}

#[derive(Component)]
pub struct Swapchain {
    inner: Arc<SwapchainInner>,

    /// An available semaphore for the next acquire.
    /// The semaphore becomes available after `acquire_fence` becomes signaled,
    /// indicating that the previous acquire has been completed.
    acquire_semaphore: vk::Semaphore,
    /// An unsignaled fence for the next acquire.
    acquire_fence: vk::Fence,
    images: SmallVec<[Option<SwapchainImageInner>; 3]>,
}

impl HasDevice for Swapchain {
    fn device(&self) -> &Device {
        &self.inner.device
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            // This semaphore can be immediately deleted because we know it's not in use.
            // This semaphore is only used for the `next` acquire. It only gets swapped back
            // after we've waited on it.
            self.inner
                .device
                .destroy_semaphore(self.acquire_semaphore, None);
            // Same goes for this fence. It should be unsignaled.
            self.inner.device.destroy_fence(self.acquire_fence, None);
        }
    }
}

struct SwapchainInner {
    device: Device,
    surface: Surface,
    inner: vk::SwapchainKHR,
    format: vk::Format,
    generation: u64,

    color_space: ColorSpace,
    extent: UVec2,
    layer_count: u32,
}

impl Drop for SwapchainInner {
    fn drop(&mut self) {
        unsafe {
            self.device
                .extension::<KhrSwapchain>()
                .destroy_swapchain(self.inner, None);
        }
    }
}

pub struct SwapchainCreateInfo<'a> {
    pub flags: vk::SwapchainCreateFlagsKHR,
    pub min_image_count: u32,
    pub image_format: vk::Format,
    pub image_color_space: vk::ColorSpaceKHR,
    pub image_extent: UVec2,
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
            width = %info.image_extent.x,
            height = %info.image_extent.y,
            format = ?info.image_format,
            color_space = ?info.image_color_space,
            "Creating swapchain"
        );
        let swapchain_loader = device.extension::<KhrSwapchain>();
        unsafe {
            let mut create_info = vk::SwapchainCreateInfoKHR {
                flags: info.flags,
                surface: surface.raw(),
                min_image_count: info.min_image_count,
                image_format: info.image_format,
                image_color_space: info.image_color_space,
                image_extent: vk::Extent2D {
                    width: info.image_extent.x,
                    height: info.image_extent.y,
                },
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
                        let present_semaphore = device
                            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                            .unwrap();
                        let acquire_semaphore = device
                            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                            .unwrap();
                        let acquire_fence = device
                            .create_fence(
                                &vk::FenceCreateInfo {
                                    flags: vk::FenceCreateFlags::SIGNALED,
                                    ..Default::default()
                                },
                                None,
                            )
                            .unwrap();
                        let present_fence = device
                            .create_fence(
                                &vk::FenceCreateInfo {
                                    flags: vk::FenceCreateFlags::SIGNALED,
                                    ..Default::default()
                                },
                                None,
                            )
                            .unwrap();
                        let img = SwapchainImageInner {
                            image,
                            indice: i as u32,
                            swapchain: inner.clone(),
                            view,
                            present_semaphore,
                            acquire_semaphore,
                            acquire_fence,
                            present_fence,
                        };
                        Ok(Some(img))
                    })
                    .collect::<VkResult<Vec<Option<SwapchainImageInner>>>>()?
                    .into(),
                inner,
                acquire_semaphore: {
                    // Create one extra semaphore for the first acquire
                    let semaphore = device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .unwrap();
                    device
                        .set_debug_name(semaphore, cstr!("Extra Acquire Semaphore"))
                        .ok();
                    semaphore
                },
                acquire_fence: {
                    // Create one extra semaphore for the first acquire
                    let semaphore = device
                        .create_fence(
                            &vk::FenceCreateInfo {
                                flags: vk::FenceCreateFlags::empty(), // Important: we expect this fence to be unsignaled!
                                ..Default::default()
                            },
                            None,
                        )
                        .unwrap();
                    device
                        .set_debug_name(semaphore, cstr!("Extra Acquire Fence"))
                        .ok();
                    semaphore
                },
            })
        }
    }

    pub fn recreate(&mut self, info: &SwapchainCreateInfo) -> VkResult<()> {
        tracing::info!(
            width = %info.image_extent.x,
            height = %info.image_extent.y,
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
                image_extent: vk::Extent2D {
                    width: info.image_extent.x,
                    height: info.image_extent.y,
                },
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
                .extension::<KhrSwapchain>()
                .create_swapchain(&create_info, None)?;

            let images = self
                .inner
                .device
                .extension::<KhrSwapchain>()
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

                    let present_semaphore = self
                        .inner
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .unwrap();
                    let acquire_semaphore = self
                        .inner
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .unwrap();
                    let acquire_fence = self
                        .inner
                        .device
                        .create_fence(
                            &vk::FenceCreateInfo {
                                flags: vk::FenceCreateFlags::SIGNALED,
                                ..Default::default()
                            },
                            None,
                        )
                        .unwrap();
                    let present_fence = self
                        .inner
                        .device
                        .create_fence(
                            &vk::FenceCreateInfo {
                                flags: vk::FenceCreateFlags::SIGNALED,
                                ..Default::default()
                            },
                            None,
                        )
                        .unwrap();
                    let img = SwapchainImageInner {
                        image,
                        indice: i as u32,
                        swapchain: self.inner.clone(),
                        view,
                        present_semaphore,
                        acquire_semaphore,
                        acquire_fence,
                        present_fence,
                    };
                    Ok(Some(img))
                })
                .collect::<VkResult<Vec<Option<SwapchainImageInner>>>>()?
                .into();
        }
        Ok(())
    }

    pub fn extent(&self) -> UVec2 {
        self.inner.extent
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
    ///
    /// Ignored when `image_format` is set.
    pub hdr: bool,

    /// If set to true, SDR swapchains will be created with a sRGB format.
    /// If set to false, SDR swapchains will be created with a UNORM format.
    ///
    /// Set this to false if the data will be written to the swapchain image as a storage image,
    /// and the tonemapper will apply gamma correction manually.
    ///
    /// Set this to true if the swapchain will be directly used as a render target. In this case,
    /// the sRGB gamma correction will be applied automatically. This is the default.
    ///
    /// Ignored when `image_format` is set.
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

fn recreate_swapchain(
    swapchain: &mut Swapchain,
    surface: &Surface,
    window: &Window,
    config: Option<&SwapchainConfig>,
) {
    let default_config = SwapchainConfig::default();
    let config = config.unwrap_or(&default_config);
    let create_info = get_create_info(
        surface,
        swapchain.device().physical_device(),
        window,
        config,
    );
    swapchain.recreate(&create_info).unwrap();
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
    windows_to_rebuild.extend(window_resized_events.read().filter_map(|event| {
        let (window, _, swapchain, _) = query.get(event.window).ok()?;
        let swapchain = swapchain?;
        if window.physical_height() != swapchain.extent().y
            || window.physical_width() != swapchain.extent().x
        {
            Some(event.window)
        } else {
            None
        }
    }));
    windows_to_rebuild.extend(suboptimal_events.read().map(|a| a.window));
    for resized_window in windows_to_rebuild.into_iter() {
        let (window, config, swapchain, surface) = query.get_mut(resized_window).unwrap();
        if let Some(mut swapchain) = swapchain {
            recreate_swapchain(&mut swapchain, surface, window, config);
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
                state: ResourceState::default(),
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
    let mut image_extent = UVec2::new(
        surface_capabilities.current_extent.width,
        surface_capabilities.current_extent.height,
    );
    if image_extent == UVec2::splat(u32::MAX) {
        // currentExtent is the current width and height of the surface, or the special value (0xFFFFFFFF, 0xFFFFFFFF)
        // indicating that the surface size will be determined by the extent of a swapchain targeting the surface.
        image_extent = UVec2::new(
            window.resolution.physical_width(),
            window.resolution.physical_height(),
        );
    }
    image_extent = image_extent.min(UVec2::new(
        surface_capabilities.max_image_extent.width,
        surface_capabilities.max_image_extent.height,
    ));
    image_extent = image_extent.max(UVec2::new(
        surface_capabilities.min_image_extent.width,
        surface_capabilities.min_image_extent.height,
    ));
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
        image_extent,
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
    state: ResourceState,
    pub(crate) inner: Option<SwapchainImageInner>,
}
impl SwapchainImage {
    pub fn blocking_stages(&self, stages: vk::PipelineStageFlags2) -> QueueDependency {
        debug_assert!(!self.acquire_semaphore.is_null());
        QueueDependency(vk::SemaphoreSubmitInfo {
            stage_mask: stages,
            semaphore: self.acquire_semaphore,
            ..Default::default()
        })
    }
}
unsafe impl<'t> GPUResource for &'t mut SwapchainImage {
    fn get_resource_state(&self, state_table: &crate::future::ResourceStateTable) -> ResourceState {
        self.state.clone()
    }

    fn set_resource_state(
        &mut self,
        state_table: &mut crate::future::ResourceStateTable,
        state: ResourceState,
    ) {
        self.state = state;
    }
}
impl Deref for SwapchainImage {
    type Target = SwapchainImageInner;

    fn deref(&self) -> &Self::Target {
        self.inner
            .as_ref()
            .expect("SwapchainAcquire must have been called")
    }
}

pub struct SwapchainImageInner {
    pub image: vk::Image,
    pub view: vk::ImageView,
    indice: u32,
    swapchain: Arc<SwapchainInner>,

    /// The swapchain image was acquired using this semaphore.
    acquire_semaphore: vk::Semaphore,
    acquire_fence: vk::Fence,
    pub(crate) present_semaphore: vk::Semaphore,
    present_fence: vk::Fence,
}
impl Drop for SwapchainImageInner {
    fn drop(&mut self) {
        unsafe {
            self.swapchain
                .device
                .wait_for_fences(&[self.acquire_fence, self.present_fence], true, !0)
                .unwrap();
            self.swapchain
                .device
                .destroy_fence(self.acquire_fence, None);
            self.swapchain
                .device
                .destroy_fence(self.present_fence, None);
            self.swapchain.device.destroy_image_view(self.view, None);
            self.swapchain
                .device
                .destroy_semaphore(self.acquire_semaphore, None);
            self.swapchain
                .device
                .destroy_semaphore(self.present_semaphore, None);
            // It is now ok to destroy the reference to the swapchain. We've waited on the acquire and present fences.
        }
    }
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

    fn extent(&self) -> UVec3 {
        UVec3::new(self.swapchain.extent.x, self.swapchain.extent.y, 1)
    }

    fn format(&self) -> vk::Format {
        self.swapchain.format
    }
}
impl ImageViewLike for SwapchainImage {
    fn raw_image_view(&self) -> vk::ImageView {
        self.view
    }
}

/// Acquires the next image from the swapchain by calling [`ash::khr::Swapchain::acquire_next_image`].
/// Generic parameter `Filter` is used to uniquely specify the swapchain to acquire from.
/// For example, `With<PrimaryWindow>` will only acquire the next image from the swapchain
/// associated with the primary window.
pub fn acquire_swapchain_image<Filter: QueryFilter>(
    In(queue): In<QueueSystemCtx>,
    mut query: Query<
        (
            Entity,
            &mut Swapchain, // Need mutable reference to swapchain to call acquire_next_image
            &mut SwapchainImage,
            Option<&SwapchainConfig>,
            &Window,
            &Surface,
        ),
        Filter,
    >,
    mut suboptimal_events: EventWriter<SuboptimalEvent>,
    device: Res<Device>,
) {
    println!("acquire");
    assert!(queue.dependencies().dependencies.is_empty());
    let (entity, mut swapchain, mut swapchain_image, swapchain_config, window, surface) =
        match query.get_single_mut() {
            Ok(item) => item,
            Err(QuerySingleError::NoEntities(_str)) => {
                return;
            }
            Err(QuerySingleError::MultipleEntities(str)) => {
                panic!("{}", str)
            }
        };

    let mut swapchain_acquire_second_try = |swapchain: &mut Swapchain| {
        recreate_swapchain(swapchain, surface, window, swapchain_config);
        match unsafe {
            swapchain
                .inner
                .device
                .extension::<KhrSwapchain>()
                .acquire_next_image(
                    swapchain.inner.inner,
                    !0,
                    swapchain.acquire_semaphore,
                    swapchain.acquire_fence,
                )
        } {
            Ok((indice, suboptimal)) => {
                if suboptimal {
                    tracing::warn!("Suboptimal swapchain");
                    suboptimal_events.send(SuboptimalEvent { window: entity });
                }
                indice
            }
            Err(err) => {
                panic!(
                    "Failed to acquire next image after swapchain recreate: {:?}",
                    err
                );
            }
        }
    };

    let indice = match unsafe {
        swapchain
            .inner
            .device
            .extension::<KhrSwapchain>()
            .acquire_next_image(
                swapchain.inner.inner,
                !0,
                swapchain.acquire_semaphore,
                swapchain.acquire_fence,
            )
    } {
        Ok((indice, suboptimal)) => {
            if suboptimal {
                tracing::warn!("vkAcquireNextImageKHR: Suboptimal");
                suboptimal_events.send(SuboptimalEvent { window: entity });
            }
            indice
        }
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
            tracing::warn!("vkAcquireNextImageKHR: OUT_OF_DATE");
            swapchain_acquire_second_try(&mut swapchain)
        }
        Err(err) => {
            panic!("Failed to acquire next image: {:?}", err);
        }
    };
    unsafe {
        let wait_value = queue.dependencies().this.wait_value();
        queue.dependencies().this.increment();
        device.queue_submit2(queue.raw_queue(), &[
            vk::SubmitInfo2::default().wait_semaphore_infos(&[
                vk::SemaphoreSubmitInfo {
                    semaphore: swapchain.acquire_semaphore,
                    ..Default::default()
                }, vk::SemaphoreSubmitInfo {
                    semaphore: queue.dependencies().this.semaphore.raw(),
                    value: wait_value,
                    ..Default::default()
                },
            ])
            .signal_semaphore_infos(&[
                vk::SemaphoreSubmitInfo {
                    semaphore: queue.dependencies().this.semaphore.raw(),
                    value: wait_value + 1,
                    ..Default::default()
                }
            ])
        ], vk::Fence::null()).unwrap();
    }
    let mut image = swapchain.images[indice as usize]
        .take()
        .expect("Acquiring image that hasn't been presented");
    assert!(
        swapchain_image.inner.is_none(),
        "Must present the current image before acquiring a new one"
    );

    // wait on the previous acquire
    unsafe {
        device
            .wait_for_fences(&[image.acquire_fence], true, !0)
            .unwrap();
        device.reset_fences(&[image.acquire_fence]).unwrap();
    }
    // At this point, we just acquired an image with the extra semaphore on swapchain.
    // We can also assume that 2 is now available since we've waited on its fence.
    // swap them into the image itself so that the extra semaphore on the swapchain remains unsignaled and available.
    std::mem::swap(
        &mut swapchain.acquire_semaphore, // 1
        &mut image.acquire_semaphore,     // 2
    );
    std::mem::swap(&mut swapchain.acquire_fence, &mut image.acquire_fence);
    swapchain_image.inner = Some(image);
}

pub fn present(
    In(queue): In<QueueSystemCtx>,
    device: Res<Device>,
    mut query: Query<(
        &mut Swapchain,
        &mut SwapchainImage,
        &Surface,
        &Window,
        Option<&SwapchainConfig>,
    )>,

    mut reused_states: Local<(
        Vec<vk::SwapchainKHR>,
        Vec<vk::Semaphore>,
        Vec<vk::SemaphoreSubmitInfo>,
        Vec<vk::SemaphoreSubmitInfo>,
        Vec<vk::Fence>,
        Vec<u32>,
    )>,
) {
    println!("present");
    let (ref mut swapchains, ref mut semaphores, ref mut semaphore_submit_infos, ref mut semaphore_wait_infos, ref mut fences, ref mut swapchain_image_indices) =
        reused_states.deref_mut();
    swapchains.clear();
    semaphores.clear();
    semaphore_submit_infos.clear();
    semaphore_wait_infos.clear();
    fences.clear();
    swapchain_image_indices.clear();


    for (timeline, stages) in queue.dependencies().dependencies.iter() {
        semaphore_wait_infos.push(vk::SemaphoreSubmitInfo {
            semaphore: timeline.semaphore.raw(),
            value: timeline.wait_value(),
            stage_mask: *stages,
            ..Default::default()
        });
    }

    for (mut swapchain, mut swapchain_image, _, _, _) in query.iter_mut() {
        let Some(swapchain_image) = swapchain_image.inner.take() else {
            continue;
        };
        semaphores.push(swapchain_image.present_semaphore);
        fences.push(swapchain_image.present_fence);

        swapchains.push(swapchain.inner.inner);
        semaphore_submit_infos.push(vk::SemaphoreSubmitInfo {
            semaphore: swapchain_image.present_semaphore,
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS, // TODO: We're using ALL_COMMANDS for now.
            ..Default::default()
        });
        // Safety: we're only getting the indice of the image and we're not actually reading / writing to it.
        let indice = swapchain_image.indice;
        swapchain_image_indices.push(indice);
        swapchain.images[indice as usize] = Some(swapchain_image);
    }
    if swapchains.is_empty() {
        return;
    }

    unsafe {
        // Wait for the previous presentations to finish
        device.wait_for_fences(&fences, true, !0).unwrap();
        device.reset_fences(&fences).unwrap();
        device.queue_submit2(queue.raw_queue(), &[
            vk::SubmitInfo2::default()
            .wait_semaphore_infos(&semaphore_wait_infos)
            .signal_semaphore_infos(&semaphore_submit_infos)
        ], vk::Fence::null()).unwrap();
        match device.extension::<KhrSwapchain>().queue_present(
            queue.raw_queue(),
            &vk::PresentInfoKHR::default()
                .swapchains(&swapchains)
                .image_indices(&swapchain_image_indices)
                .wait_semaphores(&semaphores)
                .push_next(&mut vk::SwapchainPresentFenceInfoEXT::default().fences(&fences)),
        ) {
            Ok(suboptimal) => {
                if suboptimal {
                    tracing::warn!("vkQueuePresent: Suboptimal");
                }
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                tracing::warn!("vkQueuePresent: ERROR_OUT_OF_DATE");
                // Proactively recreate surface if presenting one swapchain only.
                if let Ok((mut swapchain, _, surface, window, config)) = query.get_single_mut() {
                    tracing::warn!(
                        "Immediately recreating swapchain after vkQueuePresent: ERROR_OUT_OF_DATE"
                    );
                    recreate_swapchain(&mut swapchain, surface, window, config);
                };
                // If presenting multiple swapchains together, leave this for `acquire_next_image` which is done
                // individually for each surface.
            }
            Err(err) => panic!("Failed to present: {:?}", err),
        }
    }
    drop(queue);
}
