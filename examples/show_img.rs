#![feature(generators, generator_trait)]
use std::{
    io::{BufReader, Cursor},
    sync::Arc,
};

use async_ash::{
    commands::SharedCommandPool, copy_buffer_to_image, cstr, debug::command_debug, Device,
    ImageExt, Queues, Surface, Swapchain, SwapchainCreateInfo,
};
use async_ash_alloc::{buffer::ResidentBuffer, Allocator};
use async_ash_core::{
    ash,
    ash::vk,
    copy_buffer,
    future::*,
    macros::{commands, gpu},
    DeviceCreateInfo, FencePool, Instance, InstanceCreateInfo, PhysicalDevice,
    PhysicalDeviceFeatures, QueueFuture, QueueType, QueuesRouter, TimelineSemaphorePool,
};
use image::GenericImageView;
struct WindowedApplication {
    device: Arc<Device>,
    queues: Queues,
    allocator: Allocator,
    shared_command_pools: Vec<Option<SharedCommandPool>>,
    shared_semaphore_pool: TimelineSemaphorePool,
    shared_fence_pool: FencePool,
    surface: Arc<Surface>,
}
impl WindowedApplication {
    pub fn new(window: &winit::window::Window) -> Self {
        let entry = unsafe { ash::Entry::load().unwrap() };
        let instance = Arc::new(
            Instance::create(
                Arc::new(entry),
                &InstanceCreateInfo {
                    enabled_extension_names: &[
                        ash::extensions::khr::Surface::name().as_ptr(),
                        ash::extensions::khr::Win32Surface::name().as_ptr(),
                    ],
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let physical_device = PhysicalDevice::enumerate(&instance)
            .unwrap()
            .into_iter()
            .skip(1)
            .next()
            .unwrap();
        println!(
            "Using {:?}, api version {:?}, driver version {:?}",
            physical_device.properties().device_name(),
            physical_device.properties().api_version(),
            physical_device.properties().driver_version()
        );
        let queues_router = QueuesRouter::new(&physical_device);

        let (device, queues) = physical_device
            .create_device(DeviceCreateInfo {
                enabled_features: PhysicalDeviceFeatures {
                    v13: vk::PhysicalDeviceVulkan13Features {
                        synchronization2: vk::TRUE,
                        ..Default::default()
                    },
                    v12: vk::PhysicalDeviceVulkan12Features {
                        timeline_semaphore: vk::TRUE,
                        buffer_device_address: vk::TRUE,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                enabled_extension_names: &[
                    ash::extensions::khr::Swapchain::name().as_ptr(),
                    //ash::vk::ExtSwapchainMaintenance1Fn::name().as_ptr(),
                ],
                ..DeviceCreateInfo::with_queue_create_callback(|queue_family_index| {
                    queues_router.priorities(queue_family_index)
                })
            })
            .unwrap();
        let allocator = Allocator::new(device.clone());

        let mut shared_command_pools = queues.make_shared_command_pools();
        let mut shared_semaphore_pool = TimelineSemaphorePool::new(device.clone());
        let mut shared_fence_pool = FencePool::new(device.clone());

        let surface = Surface::create(instance, window, window).unwrap();
        Self {
            device,
            queues,
            allocator,
            shared_command_pools,
            shared_semaphore_pool,
            shared_fence_pool,
            surface: Arc::new(surface),
        }
    }

    pub fn submit<'a, F: QueueFuture + 'a>(
        &'a mut self,
        future: F,
        recycled_state: &'a mut F::RecycledState,
    ) -> F::Output {
        let fut = self.queues.submit(
            future,
            &mut self.shared_command_pools,
            &mut self.shared_semaphore_pool,
            &mut self.shared_fence_pool,
            recycled_state,
        );
        futures::executor::block_on(fut)
    }
    pub fn create_swapchain(
        &mut self,
        size: vk::Extent2D,
        usage: vk::ImageUsageFlags,
    ) -> Swapchain {
        Swapchain::create(
            self.device.clone(),
            self.surface.clone(),
            SwapchainCreateInfo {
                image_extent: size,
                image_format: vk::Format::B8G8R8A8_UNORM,
                image_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                ..SwapchainCreateInfo::pick(&self.surface, self.device.physical_device(), usage)
                    .unwrap()
            },
        )
        .unwrap()
    }
}

fn main() {
    let res = reqwest::blocking::get(
        "https://github.githubassets.com/images/modules/signup/gc_banner_dark.png",
    )
    .unwrap()
    .bytes()
    .unwrap();
    let res = Cursor::new(res);
    let image = image::load(res, image::ImageFormat::Png).unwrap();
    let image_size = image.dimensions();
    let image = image.into_rgba8();

    use winit::{
        event::{Event, WindowEvent},
        event_loop::EventLoop,
        window::WindowBuilder,
    };

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(1300, 600))
        .build(&event_loop)
        .unwrap();
    let mut application = WindowedApplication::new(&window);

    let size = window.inner_size();
    let size = vk::Extent2D {
        width: size.width,
        height: size.height,
    };
    let mut swapchain = application.create_swapchain(size, vk::ImageUsageFlags::TRANSFER_DST);

    let mut recycled_state = Default::default();
    event_loop.run(move |event, _, control_flow| {
        control_flow.set_wait();

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                control_flow.set_exit();
            }
            Event::MainEventsCleared => {
                // Application update code.

                // Queue a RedrawRequested event.
                //
                // You only need to call this if you've determined that you need to redraw, in
                // applications which do not always need to. Applications that redraw continuously
                // can just render here instead.
                //window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                let image_buffer = application.allocator.create_device_buffer_with_data(image.as_raw(), vk::BufferUsageFlags::TRANSFER_SRC).unwrap();

                let swapchain_image = swapchain.acquire_next_image(application.shared_semaphore_pool.get_binary_semaphore());
                let future = gpu! {
                    let swapchain_image = swapchain_image.await;
                    let mut swapchain_image_region = swapchain_image.map(|image| {
                        image.crop(vk::Extent3D {
                            width: image_size.0,
                            height: image_size.1,
                            depth: 1
                        }, Default::default())
                    });
                    commands! {
                        let image_buffer = image_buffer.await;
                        copy_buffer_to_image(&image_buffer, &mut swapchain_image_region, vk::ImageLayout::TRANSFER_DST_OPTIMAL).await;
                        retain!(image_buffer);
                    }.schedule().await;
                    let swapchain_image = swapchain_image_region.map(|image| image.into_inner());
                    swapchain_image.present().await;
                };

                application.submit(future, &mut recycled_state);
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in MainEventsCleared, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.
                println!("Redraw");
            },
            _ => (),
        }
    });
}
