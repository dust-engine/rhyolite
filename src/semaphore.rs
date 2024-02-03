use std::fmt::Debug;

use ash::vk;
use bevy_ecs::system::Resource;

use crate::{
    utils::resource_pool::{ResourcePool, ResourcePoolItem},
    Device,
};

#[derive(Resource)]
pub(crate) struct BinarySemaphorePool {
    device: Device,
    pool: ResourcePool<vk::Semaphore>,
}

pub struct BinarySemaphore(ResourcePoolItem<vk::Semaphore>);
impl BinarySemaphore {
    pub fn raw(&self) -> vk::Semaphore {
        *self.0
    }
}
impl Debug for BinarySemaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("BinarySemaphore").field(&self.raw()).finish()
    }
}

impl BinarySemaphorePool {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            pool: ResourcePool::new(),
        }
    }
    pub fn create(&self) -> BinarySemaphore {
        let item = self.pool.create(|| unsafe {
            let info = vk::SemaphoreCreateInfo::default();
            self.device.create_semaphore(&info, None).unwrap()
        });
        BinarySemaphore(item)
    }
}
