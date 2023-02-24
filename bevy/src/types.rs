use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use bevy_ecs::system::Resource;

#[derive(Resource)]
pub struct Allocator(rhyolite::Allocator);
impl Allocator {
    pub fn new(inner: rhyolite::Allocator) -> Self {
        Allocator(inner)
    }
}
impl Deref for Allocator {
    type Target = rhyolite::Allocator;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Resource)]
pub struct DescriptorSetLayoutCache(rhyolite::descriptor::DescriptorSetLayoutCache);
impl DescriptorSetLayoutCache {
    pub fn new(device: Device) -> Self {
        let inner = rhyolite::descriptor::DescriptorSetLayoutCache::new(device.0);
        Self(inner)
    }
}
impl Deref for DescriptorSetLayoutCache {
    type Target = rhyolite::descriptor::DescriptorSetLayoutCache;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for DescriptorSetLayoutCache {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Resource, Clone)]
pub struct Device(Arc<rhyolite::Device>);

impl Device {
    pub fn new(inner: Arc<rhyolite::Device>) -> Self {
        Device(inner)
    }
    pub fn inner(&self) -> &Arc<rhyolite::Device> {
        &self.0
    }
}

impl Deref for Device {
    type Target = Arc<rhyolite::Device>;

    fn deref(&self) -> &Self::Target {
        self.inner()
    }
}

pub enum SharingMode {
    Exclusive,
    Concurrent { queue_family_indices: Vec<u32> },
}

impl Default for SharingMode {
    fn default() -> Self {
        Self::Exclusive
    }
}

impl<'a> From<&'a SharingMode> for rhyolite::SharingMode<'a> {
    fn from(value: &'a SharingMode) -> Self {
        match value {
            SharingMode::Exclusive => rhyolite::SharingMode::Exclusive,
            SharingMode::Concurrent {
                queue_family_indices,
            } => rhyolite::SharingMode::Concurrent {
                queue_family_indices: &queue_family_indices,
            },
        }
    }
}
