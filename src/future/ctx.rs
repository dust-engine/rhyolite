use std::ops::Deref;

use ash::vk::{self};

use crate::{Device, ImageLike};

use super::{res::ResourceStateTable, Access, GPUResource};

struct GlobalResourceContext {
    // manages resource id allocator
}

pub enum BarrierContext<'a> {
    Barrier {
        queue_family_index: u32,
        memory_barrier: &'a mut vk::MemoryBarrier2<'static>,
        image_barrier: &'a mut Vec<vk::ImageMemoryBarrier2<'static>>,
        // The local resource state table
        expected_resource_states: &'a mut ResourceStateTable,
        resource_states: &'a mut ResourceStateTable,
    },
    Record {
        queue_family_index: u32,
        resource_states: &'a mut ResourceStateTable,
    },
}

impl<'a> BarrierContext<'a> {
    pub fn use_resource(
        &mut self,
        resource: &mut impl GPUResource,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) {
        match self {
            Self::Barrier {
                memory_barrier,
                resource_states,
                ..
            } => {
                let old_state = resource.get_resource_state(resource_states);
                let new_barrier = old_state.get_barrier(Access { stage, access }, false);
                memory_barrier.src_access_mask |= new_barrier.src_access_mask;
                memory_barrier.dst_access_mask |= new_barrier.dst_access_mask;
                memory_barrier.src_stage_mask |= new_barrier.src_stage_mask;
                memory_barrier.dst_stage_mask |= new_barrier.dst_stage_mask;
            }
            Self::Record {
                resource_states, ..
            } => {
                let mut old_state = resource.get_resource_state(resource_states);
                old_state.transition(Access { stage, access });
                resource.set_resource_state(resource_states, old_state);
            }
        }
    }

    pub fn use_image_resource<I: ImageLike + ?Sized, T: GPUResource + Deref<Target = I>>(
        &mut self,
        resource: &mut T,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
        layout: vk::ImageLayout,
        discard_contents: bool,
    ) {
        match self {
            Self::Barrier {
                queue_family_index,
                memory_barrier,
                image_barrier,
                expected_resource_states: _,
                resource_states,
            } => {
                let old_state = resource.get_resource_state(&resource_states);
                let had_image_layout_transfer = layout != old_state.layout;
                let had_queue_family_transfer = *queue_family_index != old_state.queue_family;
                if had_image_layout_transfer || had_queue_family_transfer {
                    let memory_barrier = old_state.get_barrier(Access { stage, access }, true);
                    image_barrier.push(vk::ImageMemoryBarrier2 {
                        dst_access_mask: memory_barrier.dst_access_mask,
                        src_access_mask: memory_barrier.src_access_mask,
                        dst_stage_mask: memory_barrier.dst_stage_mask,
                        src_stage_mask: memory_barrier.src_stage_mask,
                        old_layout: if discard_contents {
                            vk::ImageLayout::UNDEFINED
                        } else {
                            old_state.layout
                        },
                        //src_queue_family_index: old_state.queue_family,
                        //dst_queue_family_index: self.queue_family_index,
                        new_layout: layout,
                        image: resource.raw_image(),
                        subresource_range: resource.subresource_range(),
                        ..Default::default()
                    });
                } else {
                    let new_barrier = old_state.get_barrier(Access { stage, access }, false);
                    memory_barrier.src_access_mask |= new_barrier.src_access_mask;
                    memory_barrier.dst_access_mask |= new_barrier.dst_access_mask;
                    memory_barrier.src_stage_mask |= new_barrier.src_stage_mask;
                    memory_barrier.dst_stage_mask |= new_barrier.dst_stage_mask;
                }
            }
            Self::Record {
                resource_states,
                queue_family_index,
            } => {
                let mut old_state = resource.get_resource_state(resource_states);
                old_state.transition(Access { stage, access });
                old_state.layout = layout;
                old_state.queue_family = *queue_family_index;
                resource.set_resource_state(resource_states, old_state);
            }
        }
    }
}

pub struct RecordContext<'a> {
    pub device: &'a Device,
    pub command_buffer: vk::CommandBuffer,
    queue_family_index: u32,
    pub resource_states: &'a mut ResourceStateTable,
}

pub struct GPUFutureContext {
    device: Device,
    queue_family_index: u32,
    pub(crate) command_buffer: vk::CommandBuffer,

    pub(crate) memory_barrier: vk::MemoryBarrier2<'static>,
    pub(crate) image_barrier: Vec<vk::ImageMemoryBarrier2<'static>>,
    expected_resource_states: ResourceStateTable,
    resource_states: ResourceStateTable,
}

impl GPUFutureContext {
    pub(crate) fn new(
        device: Device,
        command_buffer: vk::CommandBuffer,
        queue_family_index: u32,
    ) -> Self {
        Self {
            device,
            command_buffer,
            queue_family_index,
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
            queue_family_index: self.queue_family_index,
        }
    }
    pub(crate) fn barrier_ctx_barrier(&mut self) -> BarrierContext {
        BarrierContext::Barrier {
            queue_family_index: self.queue_family_index,
            memory_barrier: &mut self.memory_barrier,
            image_barrier: &mut self.image_barrier,
            expected_resource_states: &mut self.expected_resource_states,
            resource_states: &mut self.resource_states,
        }
    }
    pub(crate) fn barrier_ctx_record(&mut self) -> BarrierContext {
        BarrierContext::Record {
            queue_family_index: self.queue_family_index,
            resource_states: &mut self.resource_states,
        }
    }
}
