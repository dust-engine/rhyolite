use ash::vk;

use crate::ecs::queue_cap::IsGraphicsQueueCap;

use super::CommandRecorder;

impl<'w, const Q: char> CommandRecorder<'w, Q>
where
    (): IsGraphicsQueueCap<Q>,
{
    pub fn begin_rendering(self, info: &vk::RenderingInfo) -> RenderPass<'w, Q>{
        unsafe {
            self.device.cmd_begin_rendering(
                self.cmd_buf,
                &info,
            );
            RenderPass {
                recorder: self,
                is_dynamic_rendering: true,
            }
        }
    }
}

pub struct RenderPass<'w, const Q: char> where
(): IsGraphicsQueueCap<Q> {
    recorder: CommandRecorder<'w, Q>,
    is_dynamic_rendering: bool,
}

impl<'w, const Q: char> Drop for RenderPass<'w, Q>
where
    (): IsGraphicsQueueCap<Q>,
{
    fn drop(&mut self) {
        unsafe {
            if self.is_dynamic_rendering {
                self.recorder.device.cmd_end_rendering(self.recorder.cmd_buf);
            } else {
                self.recorder.device.cmd_end_render_pass(self.recorder.cmd_buf);
            }
        }
    }
}


impl<'w, const Q: char> RenderPass<'w, Q>
where
    (): IsGraphicsQueueCap<Q>,
{
    pub fn bind_pipeline(&mut self, pipeline: vk::Pipeline) {
        unsafe {
            self.recorder.device.cmd_bind_pipeline(
                self.recorder.cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
        }
    }
    pub fn bind_vertex_buffers(&mut self, first_binding: u32, buffers: &[vk::Buffer], offsets: &[vk::DeviceSize]) {
        unsafe {
            self.recorder.device.cmd_bind_vertex_buffers(
                self.recorder.cmd_buf,
                first_binding,
                buffers,
                offsets,
            );
        }
    }
    pub fn bind_index_buffer(&mut self, buffer: vk::Buffer, offset: vk::DeviceSize, index_type: vk::IndexType) {
        unsafe {
            self.recorder.device.cmd_bind_index_buffer(
                self.recorder.cmd_buf,
                buffer,
                offset,
                index_type,
            );
        }
    }
    pub fn bind_descriptor_sets(&mut self, layout: vk::PipelineLayout, first_set: u32, descriptor_sets: &[vk::DescriptorSet], dynamic_offsets: &[u32]) {
        unsafe {
            self.recorder.device.cmd_bind_descriptor_sets(
                self.recorder.cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            );
        }
    }
    pub fn push_constants(&mut self, layout: vk::PipelineLayout, stage_flags: vk::ShaderStageFlags, offset: u32, constants: &[u8]) {
        unsafe {
            self.recorder.device.cmd_push_constants(
                self.recorder.cmd_buf,
                layout,
                stage_flags,
                offset,
                constants,
            );
        }
    }
    pub fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
        unsafe {
            self.recorder.device.cmd_draw(
                self.recorder.cmd_buf,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }
    pub fn draw_indexed(&mut self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32) {
        unsafe {
            self.recorder.device.cmd_draw_indexed(
                self.recorder.cmd_buf,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }
    pub fn draw_indirect(&mut self, buffer: vk::Buffer, offset: vk::DeviceSize, draw_count: u32, stride: u32) {
        unsafe {
            self.recorder.device.cmd_draw_indirect(
                self.recorder.cmd_buf,
                buffer,
                offset,
                draw_count,
                stride,
            );
        }
    }
    pub fn draw_indexed_indirect(&mut self, buffer: vk::Buffer, offset: vk::DeviceSize, draw_count: u32, stride: u32) {
        unsafe {
            self.recorder.device.cmd_draw_indexed_indirect(
                self.recorder.cmd_buf,
                buffer,
                offset,
                draw_count,
                stride,
            );
        }
    }
}
