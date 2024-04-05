use ash::vk::{self};
use bevy::math::IVec3;

use crate::{
    dispose::RenderObject, ecs::{queue_cap::IsGraphicsQueueCap, RenderCommands, RenderImage}, pipeline::GraphicsPipeline, Device, HasDevice, ImageLike, QueueRef
};

use super::{CommandRecorder, SemaphoreSignalCommands, TrackedResource};

pub trait GraphicsCommands: Sized + CommandRecorder {
    
    fn blit_image<S: ImageLike, D: ImageLike>(
        &mut self,
        src: &impl TrackedResource<State = vk::ImageLayout, Target = S>,
        dst: &impl TrackedResource<State = vk::ImageLayout, Target = D>,
        filter: vk::Filter,
    ) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            let src_subresource = src.subresource_range();
            assert_eq!(src_subresource.level_count, 1);
            let src_offsets = [
                src.offset(),
                src.offset() + IVec3::try_from(src.extent()).unwrap(),
            ]
            .map(|v| vk::Offset3D {
                x: v.x,
                y: v.y,
                z: v.z,
            });
            let dst_subresource = dst.subresource_range();
            assert_eq!(dst_subresource.level_count, 1);

            let dst_offsets = [
                dst.offset(),
                dst.offset() + IVec3::try_from(dst.extent()).unwrap(),
            ]
            .map(|v| vk::Offset3D {
                x: v.x,
                y: v.y,
                z: v.z,
            });
            self.device().cmd_blit_image(
                cmd_buf,
                src.raw_image(),
                src.current_state(),
                dst.raw_image(),
                dst.current_state(),
                &[vk::ImageBlit {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: src_subresource.aspect_mask,
                        mip_level: src_subresource.base_mip_level,
                        base_array_layer: src_subresource.base_array_layer,
                        layer_count: src_subresource.layer_count,
                    },
                    src_offsets,
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: dst_subresource.aspect_mask,
                        mip_level: dst_subresource.base_mip_level,
                        base_array_layer: dst_subresource.base_array_layer,
                        layer_count: dst_subresource.layer_count,
                    },
                    dst_offsets,
                }],
                filter,
            )
        }
    }
    
    fn begin_rendering(&mut self, info: &vk::RenderingInfo) -> DynamicRenderPass<Self> {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device().cmd_begin_rendering(cmd_buf, info);
            DynamicRenderPass { recorder: self }
        }
    }
}

impl<const Q: char> GraphicsCommands for RenderCommands<'_, '_, Q>
where
    (): IsGraphicsQueueCap<Q>,
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
    T: GraphicsCommands + SemaphoreSignalCommands,
{
    const QUEUE_CAP: char = T::QUEUE_CAP;
    fn cmd_buf(&mut self) -> vk::CommandBuffer {
        self.recorder.cmd_buf()
    }
    fn current_queue(&self) -> QueueRef {
        self.recorder.current_queue()
    }
    fn semaphore_signal(&mut self) -> &mut impl SemaphoreSignalCommands {
        self.recorder
    }
}
impl<T> RenderPassCommands for DynamicRenderPass<'_, T> where
    T: GraphicsCommands + SemaphoreSignalCommands
{
}
