use std::{collections::BTreeMap, ops::Deref};

use ash::vk::{self, AccessFlags2};

use crate::{Device, ImageLike};

use super::{
    res::{self, ResourceState, ResourceStateTable},
    Access, GPUResource,
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
        resource: &impl GPUResource,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) {
        let old_state = resource.get_resource_state(&self.resource_states);
        *self.memory_barrier = old_state.get_barrier(Access { stage, access }, false);
    }

    pub fn use_image_resource<I: ImageLike, T: GPUResource + Deref<Target = I>>(
        &mut self,
        resource: &mut T,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
        layout: vk::ImageLayout,
        discard_contents: bool,
    ) {
        let old_state = resource.get_resource_state(&self.resource_states);
        let had_image_layout_transfer = layout != old_state.layout;
        if had_image_layout_transfer {
            let memory_barrier = old_state.get_barrier(Access { stage, access }, true);
            self.image_barrier.push(vk::ImageMemoryBarrier2 {
                dst_access_mask: memory_barrier.dst_access_mask,
                src_access_mask: memory_barrier.src_access_mask,
                dst_stage_mask: memory_barrier.dst_stage_mask,
                src_stage_mask: memory_barrier.src_stage_mask,
                old_layout: if discard_contents {
                    vk::ImageLayout::UNDEFINED
                } else {
                    old_state.layout
                },
                new_layout: layout,
                image: resource.raw_image(),
                subresource_range: resource.subresource_range(),
                ..Default::default()
            });
        } else {
            *self.memory_barrier = old_state.get_barrier(Access { stage, access }, false);
        }
    }
}

pub struct RecordContext<'a> {
    pub device: &'a Device,
    pub command_buffer: vk::CommandBuffer,
    pub resource_states: &'a mut ResourceStateTable,
}
impl<'a> RecordContext<'a> {
    pub fn set_resource_state(
        &mut self,
        resource: &mut impl GPUResource,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) {
        let mut old_state = resource.get_resource_state(&self.resource_states);
        old_state.transition(Access { stage, access });
        resource.set_resource_state(&mut self.resource_states, old_state);
    }
    pub fn set_image_resource_state<I: ImageLike, T: GPUResource + Deref<Target = I>>(
        &mut self,
        resource: &mut T,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
        layout: vk::ImageLayout,
    ) {
        let mut old_state = resource.get_resource_state(&self.resource_states);
        old_state.transition(Access { stage, access });
        old_state.layout = layout;
        resource.set_resource_state(&mut self.resource_states, old_state);
    }
}

#[derive(Debug)]
pub struct GPUFutureContext {
    device: Device,
    command_buffer: vk::CommandBuffer,

    pub(crate) memory_barrier: vk::MemoryBarrier2<'static>,
    pub(crate) image_barrier: Vec<vk::ImageMemoryBarrier2<'static>>,
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
            expected_resource_states: Default::default(),
            resource_states: Default::default(),
        }
    }
    pub(crate) fn has_barriers(&mut self) -> bool {
        return !self.image_barrier.is_empty()
            || !self.memory_barrier.dst_access_mask.is_empty()
            || !self.memory_barrier.dst_stage_mask.is_empty()
            || !self.memory_barrier.src_access_mask.is_empty()
            || !self.memory_barrier.src_stage_mask.is_empty();
    }
    pub(crate) fn clear_barriers(&mut self) {
        self.memory_barrier = vk::MemoryBarrier2::default();
        self.image_barrier.clear();
    }
    pub(crate) fn record_ctx(&mut self) -> RecordContext {
        RecordContext {
            device: &self.device,
            command_buffer: self.command_buffer,
            resource_states: &mut self.resource_states,
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
