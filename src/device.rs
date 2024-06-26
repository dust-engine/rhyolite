use crate::extensions::ExtensionNotFoundError;
use crate::plugin::DeviceMetaBuilder;
use crate::Feature;
use crate::FeatureMap;
use crate::Instance;
use crate::PhysicalDevice;
use crate::Queues;
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::ExtensionMeta;
use bevy::ecs::system::Resource;
use bevy::utils::hashbrown::HashMap;
use bevy::utils::HashSet;

use std::any::Any;
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
    extensions: HashMap<&'static CStr, Option<Box<dyn Any + Send + Sync>>>,
    features: FeatureMap,
}

impl Device {
    pub(crate) fn create(
        physical_device: PhysicalDevice,
        mut features: FeatureMap,
        extension_names: HashSet<&'static CStr>,
        extensions: HashMap<&'static CStr, Option<DeviceMetaBuilder>>,
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
        let mut pdevice_features2 = features.as_physical_device_features();
        let extension_names = extension_names
            .into_iter()
            .map(|k| k.as_ptr())
            .collect::<Vec<_>>();
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extension_names)
            .push_next(&mut pdevice_features2);
        let mut device = unsafe {
            physical_device
                .instance()
                .create_device(physical_device.raw(), &create_info, None)
        }?;
        let extensions: HashMap<&'static CStr, Option<Box<dyn Any + Send + Sync>>> = extensions
            .into_iter()
            .map(|(name, builder)| {
                return (
                    name,
                    builder.map(|builder| builder(&physical_device.instance(), &mut device)),
                );
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

    /// Only applicable to extensions not promoted to Vulkan core.
    /// For extensions promoted to Vulkan core, you may directly call the corresponding
    /// function on [`Device`].
    pub fn get_extension<T: ExtensionMeta>(&self) -> Result<&T::Device, ExtensionNotFoundError>
    where
        T::Device: 'static,
    {
        self.0
            .extensions
            .get(T::NAME)
            .map(|item| item.as_ref().expect("Extension did not add any additional commands; use `has_extension_named` to test if the extension was enabled.").downcast_ref::<T::Device>().unwrap())
            .ok_or(ExtensionNotFoundError)
    }
    pub fn has_extension_named(&self, name: &CStr) -> bool {
        self.0.extensions.contains_key(name)
    }

    /// Only applicable to extensions not promoted to Vulkan core.
    /// For extensions promoted to Vulkan core, you may directly call the corresponding
    /// function on [`Device`].
    #[track_caller]
    pub fn extension<T: ExtensionMeta>(&self) -> &T::Device
    where
        T::Device: 'static,
    {
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
