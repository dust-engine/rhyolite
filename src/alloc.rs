use std::{ops::Deref, sync::Arc};

use ash::{prelude::VkResult, vk};
use bevy::ecs::{system::Resource, world::FromWorld};

use crate::{Device, HasDevice};

#[derive(Resource, Clone)]
pub struct Allocator(Arc<AllocatorInner>);
struct AllocatorInner {
    device: Device,
    inner: vk_mem::Allocator,
}

impl FromWorld for Allocator {
    fn from_world(world: &mut bevy::ecs::world::World) -> Self {
        let device = world.resource::<Device>();
        Self::new(device.clone()).unwrap()
    }
}

impl HasDevice for Allocator {
    fn device(&self) -> &Device {
        &self.0.device
    }
}

impl Allocator {
    pub fn new(device: Device) -> VkResult<Self> {
        let info = vk_mem::AllocatorCreateInfo::new(
            device.instance(),
            &device,
            device.physical_device().raw(),
        );
        let mut flags = vk_mem::AllocatorCreateFlags::NONE;

        let buffer_device_address_enabled = device
            .feature::<vk::PhysicalDeviceBufferDeviceAddressFeatures>()
            .map(|f| f.buffer_device_address)
            .map(|b| b == vk::TRUE)
            .unwrap_or(false);
        if buffer_device_address_enabled {
            flags |= vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
        }
        let info = info.flags(flags);
        let alloc = vk_mem::Allocator::new(info)?;
        Ok(Self(Arc::new(AllocatorInner {
            device,
            inner: alloc,
        })))
    }
}

impl Deref for Allocator {
    type Target = vk_mem::Allocator;

    fn deref(&self) -> &Self::Target {
        &self.0.inner
    }
}
