use std::{ops::Deref, sync::Arc};

use ash::prelude::VkResult;
use bevy_ecs::{system::Resource, world::FromWorld};

use crate::{Device, HasDevice};

#[derive(Resource, Clone)]
pub struct Allocator(Arc<AllocatorInner>);
struct AllocatorInner {
    device: Device,
    inner: vk_mem::Allocator,
}

impl FromWorld for Allocator {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
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
        let info = vk_mem::AllocatorCreateInfo::new(device.instance(), &device, device.physical_device().raw());
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
