use std::{borrow::Borrow, collections::BTreeMap, num::{NonZero, NonZeroU64}, pin::Pin, ptr::NonNull, sync::atomic::AtomicU64, u32, u64};

use ash::vk;
use bevy::prelude::Resource;

use crate::semaphore::TimelineSemaphore;


pub struct ResourceState {
    stage: vk::PipelineStageFlags2,
    mask: vk::AccessFlags2,
    queue_family: u32,
    layout: vk::ImageLayout,
}
impl Default for ResourceState {
    fn default() -> Self {
        Self {
            stage: vk::PipelineStageFlags2::default(),
            mask: vk::AccessFlags2::default(),
            queue_family: u32::MAX,
            layout: vk::ImageLayout::default()
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct ResourceId {
    id: NonZeroU64,
    parent: u64,
}
impl ResourceId {
    pub fn new() -> Self {
        static RESOURCE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);
        let id = RESOURCE_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ResourceId {
            id: NonZeroU64::new(id).expect("ResourdeId overflow!"),
            parent: 0
        }
    }
}


pub type ResourceStateTable = BTreeMap<ResourceId, ResourceState>;

pub unsafe trait GPUResource {
    fn resource_state<'a>(self, state_table: &'a mut ResourceStateTable) -> &'a mut ResourceState where Self: 'a;
}


pub struct GPUBorrowedResource<'a, T> {
    id: ResourceId,
    inner: &'a T,
}

pub struct GPUOwnedResource<'a, T> {
    state: ResourceState,
    inner: &'a mut T,
}

unsafe impl<'t, T> GPUResource for &'t GPUBorrowedResource<'t, T> {
    fn resource_state<'a>(self, state_table: &'a mut ResourceStateTable) -> &'a mut ResourceState where Self: 'a {
        state_table.get_mut(&self.id).unwrap()
    }
}

unsafe impl<'t, T> GPUResource for &'t mut GPUOwnedResource<'t, T> {
    fn resource_state<'a>(self, _state_table: &'a mut ResourceStateTable) -> &'a mut ResourceState where Self: 'a {
        &mut self.state
    }
}
