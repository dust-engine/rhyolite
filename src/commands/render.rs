use ash::vk::{self};

use crate::{ecs::queue_cap::IsGraphicsQueueCap, Device, HasDevice, QueueRef, QueueType};

use super::CommandRecorder;

pub trait GraphicsCommands: Sized + CommandRecorder {
    fn begin_rendering(&mut self, info: &vk::RenderingInfo) -> DynamicRenderPass<Self> {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device().cmd_begin_rendering(cmd_buf, info);
            DynamicRenderPass { recorder: self }
        }
    }
    fn begin_render_pass(
        &mut self,
        info: &vk::RenderPassBeginInfo,
        contents: vk::SubpassContents,
    ) -> DynamicRenderPass<Self> {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device().cmd_begin_render_pass(cmd_buf, info, contents);
            DynamicRenderPass { recorder: self }
        }
    }
}

impl<T> GraphicsCommands for T
where
    T: CommandRecorder,
    (): IsGraphicsQueueCap<{ T::QUEUE_CAP }>,
{
}

pub struct DynamicRenderPass<'w, T: GraphicsCommands> {
    recorder: &'w mut T,
}

impl<T> Drop for DynamicRenderPass<'_, T>
where
    T: GraphicsCommands,
{
    fn drop(&mut self) {
        let cmd_buf = self.recorder.cmd_buf();
        unsafe {
            self.recorder.device().cmd_end_rendering(cmd_buf);
        }
    }
}

pub trait RenderPassCommands: CommandRecorder {
    fn bind_pipeline(&mut self, pipeline: vk::Pipeline) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::GRAPHICS, pipeline);
        }
    }
    fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .cmd_bind_vertex_buffers(cmd_buf, first_binding, buffers, offsets);
        }
    }
    fn bind_index_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .cmd_bind_index_buffer(cmd_buf, buffer, offset, index_type);
        }
    }
    fn bind_descriptor_sets(
        &mut self,
        layout: vk::PipelineLayout,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device().cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            );
        }
    }
    fn push_constants(
        &mut self,
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        constants: &[u8],
    ) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .cmd_push_constants(cmd_buf, layout, stage_flags, offset, constants);
        }
    }
    fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device().cmd_draw(
                cmd_buf,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }
    fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device().cmd_draw_indexed(
                cmd_buf,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }
    fn draw_indirect(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .cmd_draw_indirect(cmd_buf, buffer, offset, draw_count, stride);
        }
    }
    fn draw_indexed_indirect(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .cmd_draw_indexed_indirect(cmd_buf, buffer, offset, draw_count, stride);
        }
    }
    fn set_viewport(&mut self, first_viewport: u32, viewports: &[vk::Viewport]) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .cmd_set_viewport(cmd_buf, first_viewport, viewports);
        }
    }
    fn set_scissor(&mut self, first_scissor: u32, scissors: &[vk::Rect2D]) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .cmd_set_scissor(cmd_buf, first_scissor, scissors);
        }
    }

    fn push_descriptor_set(
        &mut self,
        pipeline_layout: vk::PipelineLayout,
        set: u32,
        descriptor_writes: &[vk::WriteDescriptorSet],
    ) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .extension::<ash::extensions::khr::PushDescriptor>()
                .cmd_push_descriptor_set(
                    cmd_buf,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    set,
                    descriptor_writes,
                );
        }
    }
}

impl<T> HasDevice for DynamicRenderPass<'_, T>
where
    T: GraphicsCommands,
{
    fn device(&self) -> &Device {
        self.recorder.device()
    }
}

impl<T> CommandRecorder for DynamicRenderPass<'_, T>
where
    T: GraphicsCommands,
{
    const QUEUE_CAP: char = T::QUEUE_CAP;
    fn cmd_buf(&mut self) -> vk::CommandBuffer {
        self.recorder.cmd_buf()
    }
    fn current_queue_family(&self) -> (QueueRef, u32) {
        self.recorder.current_queue_family()
    }
}
impl<T> RenderPassCommands for DynamicRenderPass<'_, T> where T: GraphicsCommands {}

pub struct RenderPass<'w, T: GraphicsCommands> {
    recorder: &'w mut T,
}
impl<T> Drop for RenderPass<'_, T>
where
    T: GraphicsCommands,
{
    fn drop(&mut self) {
        let cmd_buf = self.recorder.cmd_buf();
        unsafe {
            self.recorder.device().cmd_end_render_pass(cmd_buf);
        }
    }
}
impl<T> HasDevice for RenderPass<'_, T>
where
    T: GraphicsCommands,
{
    fn device(&self) -> &Device {
        self.recorder.device()
    }
}

impl<T> CommandRecorder for RenderPass<'_, T>
where
    T: GraphicsCommands,
{
    const QUEUE_CAP: char = T::QUEUE_CAP;
    fn cmd_buf(&mut self) -> vk::CommandBuffer {
        self.recorder.cmd_buf()
    }
    fn current_queue_family(&self) -> (QueueRef, u32) {
        self.recorder.current_queue_family()
    }
}
impl<T> RenderPassCommands for RenderPass<'_, T> where T: GraphicsCommands {}

pub trait SubpassCommands: CommandRecorder {
    fn next_subpass(&mut self, contents: vk::SubpassContents) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device().cmd_next_subpass(cmd_buf, contents);
        }
    }
}

impl<T> SubpassCommands for RenderPass<'_, T> where T: GraphicsCommands {}
