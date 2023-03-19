use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_window::{RawHandleWrapper, Window};
use rhyolite::{ash::vk, AcquireFuture, HasDevice, PhysicalDevice};

use crate::{queue::Queues, Device, Frame, SharingMode};

#[derive(Component)]
pub struct Swapchain(rhyolite::Swapchain);

impl Swapchain {
    pub fn acquire_next_image(&mut self, current_frame: &mut Frame) -> AcquireFuture {
        self.0
            .acquire_next_image(current_frame.shared_semaphore_pool.get_binary_semaphore())
    }
    fn get_create_info<'a>(
        surface: &'_ rhyolite::Surface,
        pdevice: &'_ PhysicalDevice,
        window: &'_ Window,
        config: &'a SwapchainConfigExt,
    ) -> rhyolite::SwapchainCreateInfo<'a> {
        let surface_capabilities = surface.get_capabilities(pdevice).unwrap();
        let supported_present_modes = surface.get_present_modes(pdevice).unwrap();
        rhyolite::SwapchainCreateInfo {
            flags: config.flags,
            min_image_count: config.min_image_count,
            image_format: config.image_format,
            image_color_space: config.image_color_space,
            image_extent: vk::Extent2D {
                width: window.resolution.physical_width(),
                height: window.resolution.physical_height(),
            },
            image_array_layers: config.image_array_layers,
            image_usage: config.image_usage,
            image_sharing_mode: (&config.sharing_mode).into(),
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
            },
            clipped: config.clipped,
        }
    }
    pub fn create(
        device: Arc<rhyolite::Device>,
        surface: Arc<rhyolite::Surface>,
        window: &Window,
        config: &SwapchainConfigExt,
    ) -> Self {
        let create_info = Self::get_create_info(&surface, device.physical_device(), window, config);
        Swapchain(rhyolite::Swapchain::create(device, surface, create_info).unwrap())
    }
    pub fn recreate(&mut self, window: &Window, config: &SwapchainConfigExt) {
        let create_info = Self::get_create_info(
            self.0.surface(),
            self.0.device().physical_device(),
            window,
            config,
        );
        self.0.recreate(create_info).unwrap()
    }
}

/// Runs in RenderSystems::SetUp
pub(super) fn extract_windows(
    mut commands: Commands,
    device: Res<Device>,
    mut queues: ResMut<Queues>,
    mut window_created_events: EventReader<bevy_window::WindowCreated>,
    mut window_resized_events: EventReader<bevy_window::WindowResized>,

    // By accessing a NonSend resource, we tell the scheduler to put this system on the main thread,
    // which is necessary for some OS s
    _marker: NonSend<NonSendResource>,
    mut query: Query<(
        &Window,
        &RawHandleWrapper,
        Option<&SwapchainConfigExt>,
        Option<&mut Swapchain>,
    )>,
) {
    queues.next_frame();
    for resize_event in window_resized_events.iter() {
        let (window, _, config, swapchain) = query.get_mut(resize_event.window).unwrap();
        if let Some(mut swapchain) = swapchain {
            let default_config = SwapchainConfigExt::default();
            let swapchain_config = config.unwrap_or(&default_config);
            swapchain.recreate(window, swapchain_config);
        }
    }
    for create_event in window_created_events.iter() {
        let (window, raw_handle, config, swapchain) = query.get(create_event.window).unwrap();
        let raw_handle = unsafe { raw_handle.get_handle() };
        assert!(swapchain.is_none());
        let new_surface = Arc::new(
            rhyolite::Surface::create(device.instance().clone(), &raw_handle, &raw_handle).unwrap(),
        );

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

#[derive(Component)]
pub struct SwapchainConfigExt {
    pub flags: vk::SwapchainCreateFlagsKHR,
    pub min_image_count: u32,
    pub image_format: vk::Format,
    pub image_color_space: vk::ColorSpaceKHR,
    pub image_array_layers: u32,
    pub image_usage: vk::ImageUsageFlags,
    pub sharing_mode: SharingMode,
    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    pub clipped: bool,
}

impl Default for SwapchainConfigExt {
    fn default() -> Self {
        Self {
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            min_image_count: 3,
            image_format: vk::Format::B8G8R8A8_SRGB,
            image_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            sharing_mode: SharingMode::Exclusive,
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            clipped: false,
        }
    }
}

#[derive(Default)]
pub(super) struct NonSendResource;
