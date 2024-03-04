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
    pub fn transition_resources(&mut self) -> ResourceTransitionCommandRecorder<'w> {
        ResourceTransitionCommandRecorder {
            device: self.device,
            cmd_buf: self.cmd_buf,
            global_barriers: vk::MemoryBarrier2::default(),
            image_barriers: Vec::new(),
            buffer_barriers: Vec::new(),
            dependency_flags: vk::DependencyFlags::empty(),
        }
    }
    pub fn copy_buffer(&mut self, src: vk::Buffer, dst: vk::Buffer, regions: &[vk::BufferCopy]) {
        unsafe {
            self.device.cmd_copy_buffer(
                self.cmd_buf,
                src.raw_buffer(),
                dst.raw_buffer(),
                regions,
            );
        }
    }
}

pub struct ResourceTransitionCommandRecorder<'w> {
    device: &'w Device,
    cmd_buf: vk::CommandBuffer,
    pub(crate) global_barriers: vk::MemoryBarrier2,
    pub(crate) image_barriers: Vec<vk::ImageMemoryBarrier2>,
    pub(crate) buffer_barriers: Vec<vk::BufferMemoryBarrier2>,
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
        let barrier = res.state.transition(access, false);
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
        retain_data: bool,
    ) -> Self {
        if access.is_readonly() && !retain_data {
            tracing::warn!("Transitioning an image to readonly access without retaining image data. This is likely an error.");
        }
        if access.is_writeonly() && retain_data {
            tracing::warn!("Transitioning an image to writeonly access while retaining image data. This is likely inefficient.");
        }
        let has_layout_transition = image.layout != layout;
        let barrier = image.res.state.transition(access, has_layout_transition);
        if has_layout_transition {
            self.image_barriers.push(vk::ImageMemoryBarrier2 {
                src_stage_mask: barrier.src_stage_mask,
                dst_stage_mask: barrier.dst_stage_mask,
                src_access_mask: barrier.src_access_mask,
                dst_access_mask: barrier.dst_access_mask,
                old_layout: if retain_data {
                    image.layout
                } else {
                    vk::ImageLayout::UNDEFINED
                },
                new_layout: layout,
                image: image.raw_image(),
                subresource_range: image.subresource_range(),
                ..Default::default()
            });
            image.layout = layout;
        } else {
            self.global_barriers.src_stage_mask |= barrier.src_stage_mask;
            self.global_barriers.dst_stage_mask |= barrier.dst_stage_mask;
            self.global_barriers.src_access_mask |= barrier.src_access_mask;
            self.global_barriers.dst_access_mask |= barrier.dst_access_mask;
        }
        self
    }
    #[must_use]
    pub fn transition_resource<T: BufferLike>(
        &mut self,
        _buffer: &mut RenderRes<T>,
        _access: Access,
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
