use crate::extensions::DeviceExtension;
use crate::extensions::ExtensionNotFoundError;
use crate::plugin::DeviceMetaBuilder;
use crate::Feature;
use crate::FeatureMap;
use crate::Instance;
use crate::PhysicalDevice;
use crate::QueueRef;
use crate::Queues;
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
    extensions: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
    features: FeatureMap,
}

impl Device {
    pub(crate) fn create(
        physical_device: PhysicalDevice,
        extensions: &[*const c_char],
        mut features: FeatureMap,
        meta: Vec<DeviceMetaBuilder>,
    ) -> VkResult<(Self, Queues)> {
        let mut available_queue_family = physical_device.get_queue_family_properties();
        available_queue_family.iter_mut().for_each(|props| {
            if props.queue_flags.contains(vk::QueueFlags::COMPUTE)
                || props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            {
                props.queue_flags |= vk::QueueFlags::TRANSFER;
            }
        });
        let queue_create_infos = Queues::find_with_queue_family_properties(&available_queue_family);
        let pdevice_features2 = features.as_physical_device_features();
        let create_info = vk::DeviceCreateInfo {
            p_next: &pdevice_features2 as *const vk::PhysicalDeviceFeatures2 as *const _,
            queue_create_info_count: queue_create_infos.len() as u32,
            p_queue_create_infos: queue_create_infos.as_ptr(),
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: extensions.as_ptr(),
            ..Default::default()
        };
        let mut device = unsafe {
            physical_device
                .instance()
                .create_device(physical_device.raw(), &create_info, None)
        }?;
        let extensions: HashMap<TypeId, Box<dyn Any + Send + Sync>> = meta
            .into_iter()
            .filter_map(|builder| {
                let item = builder(&physical_device.instance(), &mut device)?;
                Some((item.as_ref().type_id(), item))
            })
            .collect();
        let queues = Queues::create(&device, &queue_create_infos, &available_queue_family);
        Ok((
            Self(Arc::new(DeviceInner {
                physical_device,
                device,
                extensions,
                features,
            })),
            queues,
        ))
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
    #[track_caller]
    pub fn extension<T: DeviceExtension>(&self) -> &T {
        self.get_extension::<T>().unwrap()
    }

    pub fn feature<T: Feature + Default + 'static>(&self) -> Option<&T> {
        self.0.features.get::<T>()
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
