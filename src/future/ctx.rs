use std::{collections::BTreeMap, ops::Deref};

use ash::vk;

use crate::{Device, ImageLike};

use super::{
    res::{ResourceState, ResourceStateTable},
    GPUResource,
};

struct GlobalResourceContext {
    // manages resource id allocator
}

pub struct BarrierContext<'a> {
    memory_barrier: &'a mut vk::MemoryBarrier2<'static>,
    image_barrier: &'a mut Vec<vk::ImageMemoryBarrier2<'static>>,
    // The local resource state table
    expected_resource_states: &'a mut ResourceStateTable,
    resource_states: &'a mut ResourceStateTable,
}
impl<'a> BarrierContext<'a> {
    pub fn use_resource(
        &mut self,
        resource: &mut impl GPUResource,
        stages: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) {
    }

    pub fn use_image_resource<I: ImageLike, T: GPUResource + Deref<Target = I>>(
        &mut self,
        resource: &mut T,
        stages: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
        layout: vk::ImageLayout,
        discard_contents: bool,
    ) {
    }
}

pub struct RecordContext<'a> {
    pub device: &'a Device,
    pub command_buffer: vk::CommandBuffer,
}

pub struct GPUFutureContext {
    device: Device,
    command_buffer: vk::CommandBuffer,

    memory_barrier: vk::MemoryBarrier2<'static>,
    image_barrier: Vec<vk::ImageMemoryBarrier2<'static>>,
    expected_resource_states: ResourceStateTable,
    resource_states: ResourceStateTable,
}

impl GPUFutureContext {
    pub(crate) fn new(device: Device, command_buffer: vk::CommandBuffer) -> Self {
        Self {
            device,
            command_buffer,
            memory_barrier: vk::MemoryBarrier2::default(),
            image_barrier: Vec::new(),
            expected_resource_states: BTreeMap::new(),
            resource_states: BTreeMap::new(),
        }
    }
    pub(crate) fn clear(&mut self) {
        self.memory_barrier = vk::MemoryBarrier2::default();
        self.image_barrier.clear();
        self.expected_resource_states.clear();
        self.resource_states.clear();
    }
    pub(crate) fn record_ctx(&mut self) -> RecordContext {
        RecordContext {
            device: &self.device,
            command_buffer: self.command_buffer,
        }
    }
    pub(crate) fn barrier_ctx(&mut self) -> BarrierContext {
        BarrierContext {
            memory_barrier: &mut self.memory_barrier,
            image_barrier: &mut self.image_barrier,
            expected_resource_states: &mut self.expected_resource_states,
            resource_states: &mut self.resource_states,
        }
    }
}
