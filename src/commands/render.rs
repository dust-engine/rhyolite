use crate::{
    future::RecordContext,
    pipeline::{GraphicsPipeline, Pipeline},
};
use ash::vk;

pub struct DynamicRenderPass<'a, 's> {
    ctx: &'s mut RecordContext<'a>,
}

impl<'a> RecordContext<'a> {
    pub fn begin_rendering<'s>(
        &'s mut self,
        rendering_info: &vk::RenderingInfoKHR,
    ) -> DynamicRenderPass<'a, 's> {
        unsafe {
            self.device
                .extension::<ash::khr::dynamic_rendering::Meta>()
                .cmd_begin_rendering(self.command_buffer, rendering_info);
        }
        DynamicRenderPass { ctx: self }
    }
}

impl Drop for DynamicRenderPass<'_, '_> {
    fn drop(&mut self) {
        unsafe {
            self.ctx
                .device
                .extension::<ash::khr::dynamic_rendering::Meta>()
                .cmd_end_rendering(self.ctx.command_buffer);
        }
    }
}

impl DynamicRenderPass<'_, '_> {
    pub fn bind_pipeline(&mut self, pipeline: &GraphicsPipeline) {
        unsafe {
            self.ctx.device.cmd_bind_pipeline(
                self.ctx.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.as_raw(),
            );
        }
    }
    pub fn bind_index_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        unsafe {
            self.ctx.device.cmd_bind_index_buffer(
                self.ctx.command_buffer,
                buffer,
                offset,
                index_type,
            );
        }
    }
    pub fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        unsafe {
            self.ctx.device.cmd_bind_vertex_buffers(
                self.ctx.command_buffer,
                first_binding,
                buffers,
                offsets,
            );
        }
    }

    pub fn set_viewport(&mut self, first_viewport: u32, viewports: &[vk::Viewport]) {
        unsafe {
            self.ctx
                .device
                .cmd_set_viewport(self.ctx.command_buffer, first_viewport, viewports);
        }
    }
    pub fn set_scissor(&mut self, first_scissor: u32, scissors: &[vk::Rect2D]) {
        unsafe {
            self.ctx
                .device
                .cmd_set_scissor(self.ctx.command_buffer, first_scissor, scissors);
        }
    }

    pub fn push_constants(
        &mut self,
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        constants: &[u8],
    ) {
        unsafe {
            self.ctx.device.cmd_push_constants(
                self.ctx.command_buffer,
                layout,
                stage_flags,
                offset,
                constants,
            );
        }
    }

    pub fn push_descriptor_set(
        &mut self,
        layout: vk::PipelineLayout,
        set: u32,
        descriptor_writes: &[vk::WriteDescriptorSet<'_>],
    ) {
        unsafe {
            self.ctx
                .device
                .extension::<ash::khr::push_descriptor::Meta>()
                .cmd_push_descriptor_set(
                    self.ctx.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    set,
                    descriptor_writes,
                );
        }
    }

    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unsafe {
            self.ctx.device.cmd_draw_indexed(
                self.ctx.command_buffer,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }
}
