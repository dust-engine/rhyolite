use std::{mem::ManuallyDrop, ops::Deref};

use crate::{
    dispose::{dispose, DisposeObject},
    semaphore::TimelineSemaphore,
    ResourceState,
};
use ash::vk;
use bevy::ecs::{component::Component, system::Resource};

pub enum State {
    None,
    Write,
    Read,
}

#[derive(Resource, Component)]
pub struct RenderRes<T: Send + Sync + 'static> {
    inner: ManuallyDrop<T>,
    pub(crate) state: ManuallyDrop<ResourceState>,
}

impl<T: Send + Sync + 'static> RenderRes<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner: ManuallyDrop::new(inner),
            state: ManuallyDrop::new(ResourceState::default()),
        }
    }
    pub unsafe fn get_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}
impl<T: Send + Sync + 'static> Drop for RenderRes<T> {
    fn drop(&mut self) {
        if self.state.read_semaphores.is_empty() && self.state.write_semaphore.is_none() {
            unsafe {
                ManuallyDrop::drop(&mut self.inner);
                ManuallyDrop::drop(&mut self.state);
            }
            return;
        }
        unsafe {
            let inner = ManuallyDrop::take(&mut self.inner);
            let state = ManuallyDrop::take(&mut self.state);
            dispose(Box::new((inner, state)));
        }
    }
}
impl<T: Send + Sync + 'static> DisposeObject for (T, ResourceState) {
    fn wait_blocked(&mut self) {
        TimelineSemaphore::wait_all_blocked(
            self.1
                .read_semaphores
                .iter()
                .chain(self.1.write_semaphore.iter())
                .map(|(sem, val)| (sem.as_ref(), *val)),
            !0,
        )
        .unwrap();
    }
}

impl<T: Send + Sync + 'static> Deref for RenderRes<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Resource, Component)]
pub struct RenderImage<T: Send + Sync + 'static> {
    pub(crate) res: RenderRes<T>,
    pub(crate) layout: vk::ImageLayout,
}

impl<T: Send + Sync + 'static> RenderImage<T> {
    pub fn new(inner: T) -> Self {
        Self {
            res: RenderRes::new(inner),
            layout: vk::ImageLayout::UNDEFINED,
        }
    }
    pub fn preinitialized(inner: T) -> Self {
        Self {
            res: RenderRes::new(inner),
            layout: vk::ImageLayout::PREINITIALIZED,
        }
    }
    pub unsafe fn get_mut(&mut self) -> &mut T {
        &mut self.res.inner
    }
}

impl<T: Send + Sync + 'static> Deref for RenderImage<T> {
    type Target = RenderRes<T>;

    fn deref(&self) -> &Self::Target {
        &self.res
    }
}
