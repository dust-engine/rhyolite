#![feature(get_mut_unchecked)]

mod queue;
mod swapchain;

use std::{sync::Arc, ops::Deref, ffi::{CStr, c_char, CString}};

use async_ash::{Instance, ash::{vk, self}, Version, cstr, PhysicalDevice, HasDevice};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_window::{Window, RawHandleWrapper};


pub use swapchain::{Swapchain, SwapchainConfigExt};
pub use queue::{Queues, Frame, QueuesRouter};


pub struct RenderPlugin {
    pub enabled_instance_extensions: Vec<&'static CStr>,
    pub enabled_instance_layers: Vec<&'static CStr>,
    
    pub application_name: CString,
    pub application_version: Version,
    pub engine_name: CString,
    pub engine_version: Version,
    pub api_version: Version,

    pub physical_device_index: usize,

    pub max_frame_in_flight: usize,
}
impl Default for RenderPlugin {
    fn default() -> Self {
        Self {
            application_name: cstr!(b"Unnamed Application").to_owned(),
            application_version: Default::default(),
            engine_name: cstr!(b"Unnamed Engine").to_owned(),
            engine_version: Default::default(),
            api_version: Version::new(0, 1, 3, 0),
            enabled_instance_layers: vec![],
            enabled_instance_extensions: vec![
                ash::extensions::khr::Surface::name(),
                ash::extensions::khr::Win32Surface::name(),
            ],
            physical_device_index: 1,
            max_frame_in_flight: 3,
        }
    }
}


#[derive(Clone, Hash, Debug, PartialEq, Eq, PartialOrd, Ord, SystemSet)]
pub enum RenderSystems {
    SetUp,
    Render,
    CleanUp,
}

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut bevy_app::App) {

        let entry = unsafe { ash::Entry::load().unwrap() };
        let instance = {
            let enabled_instance_extensions: Vec<*const c_char> = self.enabled_instance_extensions.iter().map(|a| a.as_ptr()).collect();
            let enabled_instance_layers: Vec<*const c_char> = self.enabled_instance_layers.iter().map(|a| a.as_ptr()).collect();
            Arc::new(
                Instance::create(
                    Arc::new(entry),
                    &async_ash::InstanceCreateInfo {
                        enabled_extension_names: &enabled_instance_extensions,
                        enabled_layer_names: &enabled_instance_layers,
                        api_version: self.api_version.clone(),
                        engine_name: self.engine_name.as_c_str(),
                        engine_version: self.engine_version,
                        application_name: self.application_name.as_c_str(),
                        application_version: self.application_version
                    },
                )
                .unwrap(),
            )
        };
        let physical_device = async_ash::PhysicalDevice::enumerate(&instance).unwrap().into_iter().skip(self.physical_device_index).next().unwrap();
        let queues_router = async_ash::QueuesRouter::new(&physical_device);

        let (device, queues) = physical_device
            .create_device(async_ash::DeviceCreateInfo {
                enabled_features: async_ash::PhysicalDeviceFeatures {
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
                ..async_ash::DeviceCreateInfo::with_queue_create_callback(|queue_family_index| {
                    queues_router.priorities(queue_family_index)
                })
            })
            .unwrap();
        let allocator = async_ash_alloc::Allocator::new(device.clone());

        app
            .insert_resource(Device(device))
            .insert_resource(Queues::new(queues, self.max_frame_in_flight))
            .insert_resource(QueuesRouter::new(queues_router))
            .insert_resource(Allocator(allocator))
            .insert_non_send_resource(swapchain::NonSendResource::default())
            .add_system(swapchain::extract_windows.in_set(RenderSystems::SetUp));
    }
}




#[derive(Resource)]
pub struct Allocator(async_ash_alloc::Allocator);
impl Deref for Allocator {
    type Target = async_ash_alloc::Allocator;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}




#[derive(Resource, Clone)]
pub struct Device(Arc<async_ash::Device>);

impl Device {
    pub fn inner(&self) -> &Arc<async_ash::Device> {
        &self.0
    }
}

impl Deref for Device {
    type Target = Arc<async_ash::Device>;

    fn deref(&self) -> &Self::Target {
        self.inner()
    }
}


pub enum SharingMode {
    Exclusive,
    Concurrent { queue_family_indices: Vec<u32> },
}

impl Default for SharingMode {
    fn default() -> Self {
        Self::Exclusive
    }
}


impl<'a> From<&'a SharingMode> for async_ash::SharingMode<'a> {
    fn from(value: &'a SharingMode) -> Self {
        match value {
            SharingMode::Exclusive => async_ash::SharingMode::Exclusive,
            SharingMode::Concurrent { queue_family_indices } => async_ash::SharingMode::Concurrent { queue_family_indices: &queue_family_indices }
        }
    }
}
