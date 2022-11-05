
use crate::Instance;
use crate::PhysicalDevice;
use std::ops::Deref;
use std::sync::Arc;

pub trait HasDevice {
    fn device(&self) -> &Arc<Device>;
    fn physical_device(&self) -> &PhysicalDevice {
        &self.device().physical_device
    }
    fn instance(&self) -> &Arc<Instance> {
        &self.device().physical_device.instance()
    }
}

pub struct Device {
    physical_device: PhysicalDevice,
    device: ash::Device,
}

impl Device {
    pub(crate) fn new(physical_device: PhysicalDevice, device: ash::Device) -> Self {
        Self {
            physical_device,
            device,
        }
    }
    pub fn instance(&self) -> &Arc<Instance> {
        self.physical_device.instance()
    }
    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.physical_device
    }
}

impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        tracing::info!(device = ?self.device.handle(), "drop deice");
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
