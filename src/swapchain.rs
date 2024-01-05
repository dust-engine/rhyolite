use std::sync::Arc;

use bevy_app::{App, Plugin};
use bevy_ecs::system::Resource;

use crate::Device;

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
        //app.add_device_extension(ash::extensions::khr::Swapchain::name());
    }
    fn finish(&self, app: &mut App) {
        let device: &Device = app.world.resource();
        let swapchain_loader = ash::extensions::khr::Swapchain::new(device.instance(), device);
        app.insert_resource(SwapchainLoader {
            device: device.clone(),
            loader: Arc::new(swapchain_loader),
        });
    }
}

#[derive(Resource, Clone)]
pub struct SwapchainLoader {
    device: Device,
    loader: Arc<ash::extensions::khr::Swapchain>,
}
