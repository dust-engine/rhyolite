use std::{collections::{BTreeMap, BTreeSet}, ops::Deref};

use ash::vk;
use bevy_ecs::{
    archetype::{ArchetypeComponentId, ArchetypeGeneration},
    component::{Component, ComponentId},
    query::{QueryData, ReadOnlyQueryData, WorldQuery},
    system::{Res, ResMut, Resource, SystemParam},
    world::{unsafe_world_cell::UnsafeWorldCell, Mut},
};
use bevy_utils::ConfigMap;

use crate::{Access, ResourceState};

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
            layout: vk::ImageLayout::UNDEFINED
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
