use std::{collections::BTreeMap, ops::Deref};

use ash::vk;

use crate::{Device, ImageLike};

use super::{
    res::{self, ResourceState, ResourceStateTable},
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
        resource: &impl GPUResource,
        stages: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) {
        let old_state = resource.get_resource_state(&self.resource_states);
        compute_memory_access(
            self.memory_barrier,
            old_state.access,
            old_state.stage,
            access,
            stages,
            false,
        );
    }

    pub fn use_image_resource<I: ImageLike, T: GPUResource + Deref<Target = I>>(
        &mut self,
        resource: &mut T,
        stages: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
        layout: vk::ImageLayout,
        discard_contents: bool,
    ) {
        let old_state = resource.get_resource_state(&self.resource_states);
        let had_image_layout_transfer = layout != old_state.layout;
        if had_image_layout_transfer {
            let mut memory_barrier = vk::MemoryBarrier2::default();
            compute_memory_access(
                &mut memory_barrier,
                old_state.access,
                old_state.stage,
                access,
                stages,
                true,
            );
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
            compute_memory_access(
                self.memory_barrier,
                old_state.access,
                old_state.stage,
                access,
                stages,
                false,
            );
        }
    }
}

pub struct RecordContext<'a> {
    pub device: &'a Device,
    pub command_buffer: vk::CommandBuffer,
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

fn compute_memory_access(
    memory_barrier: &mut vk::MemoryBarrier2,
    before_access: vk::AccessFlags2,
    before_stage: vk::PipelineStageFlags2,
    after_access: vk::AccessFlags2,
    after_stage: vk::PipelineStageFlags2,
    had_image_layout_transfer: bool,
) {
    if access_flag_is_read_only(before_access)
        && access_flag_is_read_only(after_access)
        && !had_image_layout_transfer
    {
        // read after read
        return;
    }
    if access_flag_is_read_only(before_access) && !had_image_layout_transfer {
        // Write after read
        memory_barrier.src_stage_mask |= before_stage;
        memory_barrier.dst_stage_mask |= after_stage;
        return;
        // No need for memory barrier. Execution barrier only
    }
    memory_barrier.src_stage_mask |= before_stage;
    memory_barrier.src_access_mask |= before_access;
    memory_barrier.dst_stage_mask |= after_stage;
    memory_barrier.dst_access_mask |= after_access;
}

const ALL_READ_BITS: vk::AccessFlags2 = vk::AccessFlags2::from_raw(
    vk::AccessFlags2::INDIRECT_COMMAND_READ.as_raw()
        | vk::AccessFlags2::INDEX_READ.as_raw()
        | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ.as_raw()
        | vk::AccessFlags2::UNIFORM_READ.as_raw()
        | vk::AccessFlags2::INPUT_ATTACHMENT_READ.as_raw()
        | vk::AccessFlags2::SHADER_READ.as_raw()
        | vk::AccessFlags2::COLOR_ATTACHMENT_READ.as_raw()
        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ.as_raw()
        | vk::AccessFlags2::TRANSFER_READ.as_raw()
        | vk::AccessFlags2::HOST_READ.as_raw()
        | vk::AccessFlags2::MEMORY_READ.as_raw()
        | vk::AccessFlags2::SHADER_SAMPLED_READ.as_raw()
        | vk::AccessFlags2::SHADER_STORAGE_READ.as_raw()
        | vk::AccessFlags2::VIDEO_DECODE_READ_KHR.as_raw()
        | vk::AccessFlags2::VIDEO_ENCODE_READ_KHR.as_raw()
        | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_READ_EXT.as_raw()
        | vk::AccessFlags2::CONDITIONAL_RENDERING_READ_EXT.as_raw()
        | vk::AccessFlags2::COMMAND_PREPROCESS_READ_NV.as_raw()
        | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR.as_raw()
        | vk::AccessFlags2::FRAGMENT_DENSITY_MAP_READ_EXT.as_raw()
        | vk::AccessFlags2::COLOR_ATTACHMENT_READ_NONCOHERENT_EXT.as_raw()
        | vk::AccessFlags2::DESCRIPTOR_BUFFER_READ_EXT.as_raw()
        | vk::AccessFlags2::INVOCATION_MASK_READ_HUAWEI.as_raw()
        | vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR.as_raw()
        | vk::AccessFlags2::MICROMAP_READ_EXT.as_raw()
        | vk::AccessFlags2::OPTICAL_FLOW_READ_NV.as_raw(),
);

const ALL_WRITE_BITS: vk::AccessFlags2 = vk::AccessFlags2::from_raw(
    vk::AccessFlags2::SHADER_WRITE.as_raw()
        | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw()
        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
        | vk::AccessFlags2::TRANSFER_WRITE.as_raw()
        | vk::AccessFlags2::HOST_WRITE.as_raw()
        | vk::AccessFlags2::MEMORY_WRITE.as_raw()
        | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw()
        | vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR.as_raw()
        | vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR.as_raw()
        | vk::AccessFlags2::TRANSFORM_FEEDBACK_WRITE_EXT.as_raw()
        | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT.as_raw()
        | vk::AccessFlags2::COMMAND_PREPROCESS_WRITE_NV.as_raw()
        | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR.as_raw()
        | vk::AccessFlags2::MICROMAP_WRITE_EXT.as_raw()
        | vk::AccessFlags2::OPTICAL_FLOW_WRITE_NV.as_raw(),
);

fn access_flag_is_read_only(flags: vk::AccessFlags2) -> bool {
    // Clear all the read bits. If nothing is left, that means there's no write bits.
    flags & !ALL_READ_BITS == vk::AccessFlags2::NONE
}
