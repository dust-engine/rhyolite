use ash::prelude::VkResult;
use ash::vk;
use bevy_ecs::system::Resource;

use crate::Instance;
use crate::PhysicalDevice;

use std::ffi::c_char;
use std::ffi::CStr;
use std::ops::Deref;
use std::sync::Arc;

pub trait HasDevice {
    fn device(&self) -> &Device;
    fn physical_device(&self) -> &PhysicalDevice {
        &self.device().physical_device()
    }
    fn instance(&self) -> &Instance {
        self.device().physical_device().instance()
    }
}

#[derive(Clone, Resource)]
pub struct Device(Arc<DeviceInner>);

pub struct DeviceInner {
    physical_device: PhysicalDevice,
    device: ash::Device,
}

impl Device {
    pub fn create(
        physical_device: PhysicalDevice,
        queues: &[vk::DeviceQueueCreateInfo],
        extensions: &[*const c_char],
    ) -> VkResult<Self> {
        let create_info = vk::DeviceCreateInfo {
            queue_create_info_count: queues.len() as u32,
            p_queue_create_infos: queues.as_ptr(),
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: extensions.as_ptr(),
            p_enabled_features: todo!(),
            ..Default::default()
        };
        let device = unsafe {
            physical_device
                .instance()
                .create_device(physical_device.raw(), &create_info, None)
        }?;
        Ok(Self(Arc::new(DeviceInner {
            physical_device,
            device,
        })))
    }
    pub fn instance(&self) -> &Instance {
        self.0.physical_device.instance()
    }
    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.0.physical_device
    }
}

impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.0.device
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        tracing::info!(device = ?self.handle(), "drop device");
        // Safety: Host Syncronization rule for vkDestroyDevice:
        // - Host access to device must be externally synchronized.
        // - Host access to all VkQueue objects created from device must be externally synchronized
        // We have &mut self and therefore exclusive control on device.
        // VkQueue objects may not exist at this point, because Queue retains an Arc to Device.
        // If there still exist a Queue, the Device wouldn't be dropped.
        unsafe {
            self.destroy_device(None);
        }
    }
}
