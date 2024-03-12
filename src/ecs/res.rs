use std::ops::Deref;

use ash::vk;
use bevy::ecs::{component::Component, system::Resource};

use crate::{utils::Dispose, Access, QueueType, ResourceState};

pub enum State {
    None,
    Write,
    Read,
}

struct DropMarker;
impl Drop for DropMarker {
    fn drop(&mut self) {
        println!("RenderRes needs to be properly disposed by calling 'retain'.");
    }
}

#[derive(Resource, Component)]
pub struct RenderRes<T> {
    inner: T,
    pub(crate) state: ResourceState,
    drop_marker: DropMarker,
}

impl<T> RenderRes<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            state: ResourceState::default(),
            drop_marker: DropMarker,
        }
    }
    pub unsafe fn get_mut(&mut self) -> &mut T {
        &mut self.inner
    }
    pub fn replace(&mut self, replacer: impl FnOnce(Dispose<T>) -> RenderRes<T>) {
        unsafe {
            let old = std::ptr::read(&mut self.inner);
            let new = replacer(Dispose::new(old));
            std::ptr::write(&mut self.inner, new.inner);
            self.state = new.state;
            std::mem::forget(new.drop_marker);
        }
    }
    pub fn into_dispose(self) -> Dispose<T> {
        std::mem::forget(self.drop_marker);
        Dispose::new(self.inner)
    }
}

impl<T> Deref for RenderRes<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Resource, Component)]
pub struct RenderImage<T> {
    pub(crate) res: RenderRes<T>,
    pub(crate) layout: vk::ImageLayout,
}

impl<T> RenderImage<T> {
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
    pub fn replace(&mut self, replacer: impl FnOnce(Dispose<T>) -> RenderImage<T>) {
        unsafe {
            let old = std::ptr::read(&mut self.res.inner);
            let new = replacer(Dispose::new(old));
            std::ptr::write(&mut self.res.inner, new.res.inner);
            self.res.state = new.res.state;
            self.layout = new.layout;
            std::mem::forget(new.res.drop_marker);
        }
    }
    pub fn into_dispose(self) -> Dispose<T> {
        std::mem::forget(self.res.drop_marker);
        Dispose::new(self.res.inner)
    }
}

impl<T> Deref for RenderImage<T> {
    type Target = RenderRes<T>;

    fn deref(&self) -> &Self::Target {
        &self.res
    }
}
