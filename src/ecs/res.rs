use std::ops::Deref;

use ash::vk;
use bevy::ecs::{component::Component, system::Resource};

use crate::{utils::Dispose, ResourceState};

pub enum State {
    None,
    Write,
    Read,
}
#[derive(Resource, Component)]
pub struct RenderRes<T> {
    inner: T,
    pub(crate) state: ResourceState,
}

impl<T> RenderRes<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            state: ResourceState::default(),
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
        }
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
    pub unsafe fn get_mut(&mut self) -> &mut T {
        &mut self.res.inner
    }
}

impl<T> Deref for RenderImage<T> {
    type Target = RenderRes<T>;

    fn deref(&self) -> &Self::Target {
        &self.res
    }
}
