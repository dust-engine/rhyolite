use ash::{vk, Device};

use crate::{
    ecs::{queue_cap::IsQueueCap, RenderImage, RenderRes},
    Access, BufferLike, ImageLike,
};

pub struct CommandRecorder<'w, const Q: char>
where
    (): IsQueueCap<Q>,
{
    pub(crate) device: &'w Device,
    pub(crate) cmd_buf: vk::CommandBuffer,
}

impl<'w, const Q: char> CommandRecorder<'w, Q>
where
    (): IsQueueCap<Q>,
{
    pub fn clear_color_image(
        &mut self,
        image: vk::Image,
        image_layout: vk::ImageLayout,
        clear_color_value: &vk::ClearColorValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        unsafe {
            self.device.cmd_clear_color_image(
                self.cmd_buf,
                image,
                image_layout,
                clear_color_value,
                ranges,
            )
        }
    }
}

pub struct ResourceTransitionCommandRecorder<'w> {
    device: &'w Device,
    cmd_buf: vk::CommandBuffer,
    global_barriers: vk::MemoryBarrier2,
    image_barriers: Vec<vk::ImageMemoryBarrier2>,
    buffer_barriers: Vec<vk::BufferMemoryBarrier2>,
    dependency_flags: vk::DependencyFlags,
}
impl<'w> ResourceTransitionCommandRecorder<'w> {
    #[must_use]
    pub fn dependency_flags(mut self, flags: vk::DependencyFlags) -> Self {
        self.dependency_flags = flags;
        self
    }
    #[must_use]
    pub fn global<T>(mut self, res: &'w mut RenderRes<T>, access: Access) -> Self {
        let barrier = res.state.transition(access);
        self.global_barriers.src_stage_mask |= barrier.src_stage_mask;
        self.global_barriers.dst_stage_mask |= barrier.dst_stage_mask;
        self.global_barriers.src_access_mask |= barrier.src_access_mask;
        self.global_barriers.dst_access_mask |= barrier.dst_access_mask;
        self
    }

    #[must_use]
    pub fn image<T: ImageLike>(
        mut self,
        image: &'w mut RenderImage<T>,
        access: Access,
        layout: vk::ImageLayout,
    ) -> Self {
        let barrier = image.res.state.transition(access);
        if image.layout == layout {
            self.global_barriers.src_stage_mask |= barrier.src_stage_mask;
            self.global_barriers.dst_stage_mask |= barrier.dst_stage_mask;
            self.global_barriers.src_access_mask |= barrier.src_access_mask;
            self.global_barriers.dst_access_mask |= barrier.dst_access_mask;
        } else {
            image.layout = layout;
            self.image_barriers.push(vk::ImageMemoryBarrier2 {
                src_stage_mask: barrier.src_stage_mask,
                dst_stage_mask: barrier.dst_stage_mask,
                src_access_mask: barrier.src_access_mask,
                dst_access_mask: barrier.dst_access_mask,
                old_layout: image.layout,
                new_layout: layout,
                image: image.raw_image(),
                subresource_range: image.subresource_range(),
                ..Default::default()
            });
        }
        self
    }
    #[must_use]
    pub fn transition_resource<T: BufferLike>(
        &mut self,
        buffer: &mut RenderRes<T>,
        access: Access,
    ) {
        todo!()
    }
    pub fn end(self) {
        unsafe {
            self.device.cmd_pipeline_barrier2(
                self.cmd_buf,
                &vk::DependencyInfo {
                    dependency_flags: self.dependency_flags,
                    memory_barrier_count: if self.global_barriers.dst_access_mask
                        == vk::AccessFlags2::empty()
                        && self.global_barriers.src_access_mask == vk::AccessFlags2::empty()
                        && self.global_barriers.src_stage_mask == vk::PipelineStageFlags2::empty()
                        && self.global_barriers.dst_stage_mask == vk::PipelineStageFlags2::empty()
                    {
                        0
                    } else {
                        1
                    },
                    p_memory_barriers: &self.global_barriers,
                    buffer_memory_barrier_count: self.buffer_barriers.len() as u32,
                    p_buffer_memory_barriers: self.buffer_barriers.as_ptr(),
                    image_memory_barrier_count: self.image_barriers.len() as u32,
                    p_image_memory_barriers: self.image_barriers.as_ptr(),
                    ..Default::default()
                },
            )
        }
    }
}
