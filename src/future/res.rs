use std::{
    borrow::Borrow,
    collections::BTreeMap,
    num::{NonZero, NonZeroU64},
    pin::Pin,
    ptr::NonNull,
    sync::atomic::AtomicU64,
    u32, u64,
};

use ash::vk;
use bevy::prelude::Resource;

use crate::semaphore::TimelineSemaphore;

#[derive(Clone)]
pub struct ResourceState {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
    pub queue_family: u32,
    pub layout: vk::ImageLayout,
}
impl Default for ResourceState {
    fn default() -> Self {
        Self {
            stage: vk::PipelineStageFlags2::default(),
            access: vk::AccessFlags2::default(),
            queue_family: u32::MAX,
            layout: vk::ImageLayout::default(),
        }
    }
}
pub type ResourceStateTable = ();

pub unsafe trait GPUResource {
    fn get_resource_state(&self, state_table: &ResourceStateTable) -> ResourceState;

    fn set_resource_state(&mut self, state_table: &mut ResourceStateTable, state: ResourceState);
}

// This is returned by the retain! macro.
pub struct GPUOwned<'a, T> {
    state: ResourceState,
    inner: &'a T,
}
