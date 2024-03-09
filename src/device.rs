use crate::extensions::DeviceExtension;
use crate::extensions::ExtensionNotFoundError;
use crate::plugin::DeviceMetaBuilder;
use crate::Instance;
use crate::PhysicalDevice;
use crate::QueueRef;
use ash::prelude::VkResult;
use ash::vk;
use bevy::ecs::system::Resource;
use bevy::utils::hashbrown::HashMap;

use std::any::Any;
use std::any::TypeId;
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
    extensions: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Device {
    pub(crate) fn get_raw_queue(&self, queue: QueueRef) -> vk::Queue {
        self.0.queues[queue.0 as usize]
    }
    pub(crate) fn create(
        physical_device: PhysicalDevice,
        queues: &[vk::DeviceQueueCreateInfo],
        extensions: &[*const c_char],
        features: &vk::PhysicalDeviceFeatures2,
        meta: Vec<DeviceMetaBuilder>,
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
        let extensions: HashMap<TypeId, Box<dyn Any + Send + Sync>> = meta
            .into_iter()
            .map(|builder| {
                let item = builder(&physical_device.instance(), &device);
                (item.as_ref().type_id(), item)
            })
            .collect();
        Ok(Self(Arc::new(DeviceInner {
            physical_device,
            queues: queues_created,
            device,
            extensions,
        })))
    }
    pub fn instance(&self) -> &Instance {
        self.0.physical_device.instance()
    }
    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.0.physical_device
    }

    pub fn get_extension<T: DeviceExtension>(&self) -> Result<&T, ExtensionNotFoundError> {
        self.0
            .extensions
            .get(&TypeId::of::<T>())
            .map(|item| item.downcast_ref::<T>().unwrap())
            .ok_or(ExtensionNotFoundError)
    }
    pub fn extension<T: DeviceExtension>(&self) -> &T {
        self.get_extension::<T>().unwrap()
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
        self.extensions.clear();
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
