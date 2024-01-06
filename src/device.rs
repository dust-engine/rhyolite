use ash::prelude::VkResult;
use ash::vk;
use bevy_ecs::system::Resource;

use crate::Instance;
use crate::PhysicalDevice;
use crate::QueueRef;

use std::ffi::c_char;
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
    queues: Vec<vk::Queue>,
}

impl Device {
    pub(crate) fn get_raw_queue(&self, queue: QueueRef) -> vk::Queue {
        self.0.queues[queue.0 as usize]
    }
    pub fn create(
        physical_device: PhysicalDevice,
        queues: &[vk::DeviceQueueCreateInfo],
        extensions: &[*const c_char],
        features: &vk::PhysicalDeviceFeatures2,
    ) -> VkResult<Self> {
        let create_info = vk::DeviceCreateInfo {
            p_next: features as *const vk::PhysicalDeviceFeatures2 as *const _,
            queue_create_info_count: queues.len() as u32,
            p_queue_create_infos: queues.as_ptr(),
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: extensions.as_ptr(),
            ..Default::default()
        };
        let device = unsafe {
            physical_device
                .instance()
                .create_device(physical_device.raw(), &create_info, None)
        }?;
        let queues_created = queues
            .iter()
            .flat_map(|queue_info| unsafe {
                (0..queue_info.queue_count)
                    .map(|i| device.get_device_queue(queue_info.queue_family_index, i))
            })
            .collect::<Vec<_>>();
        Ok(Self(Arc::new(DeviceInner {
            physical_device,
            queues: queues_created,
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

impl Drop for DeviceInner {
    fn drop(&mut self) {
        tracing::info!(device = ?self.device.handle(), "drop device");
        // Safety: Host Syncronization rule for vkDestroyDevice:
        // - Host access to device must be externally synchronized.
        // - Host access to all VkQueue objects created from device must be externally synchronized
        // We have &mut self and therefore exclusive control on device.
        // VkQueue objects may not exist at this point, because Queue retains an Arc to Device.
        // If there still exist a Queue, the Device wouldn't be dropped.
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
