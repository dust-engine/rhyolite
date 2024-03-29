use std::{borrow::Cow, sync::Arc};

use ash::vk;
use bevy::utils::smallvec::SmallVec;

use crate::{
    access, buffer::BufferLike, ecs::{RenderImage, RenderRes}, semaphore::TimelineSemaphore, Access, Device, HasDevice, ImageLike, QueueRef, QueueType
};

mod render;
mod transfer;
pub use render::*;
pub use transfer::*;

pub trait CommandRecorder: HasDevice {
    const QUEUE_CAP: char;
    fn cmd_buf(&mut self) -> vk::CommandBuffer;
    fn current_queue(&self) -> QueueRef;
}

pub trait CommonCommands: CommandRecorder {
    fn clear_color_image(
        &mut self,
        image: vk::Image,
        image_layout: vk::ImageLayout,
        clear_color_value: &vk::ClearColorValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device().cmd_clear_color_image(
                cmd_buf,
                image,
                image_layout,
                clear_color_value,
                ranges,
            )
        }
    }
    fn transition_resources(&mut self) -> ImmediateTransitions {
        let cmd_buf = self.cmd_buf();
        ImmediateTransitions {
            queue: self.current_queue(),
            device: self.device(),
            cmd_buf,
            global_barriers: vk::MemoryBarrier2::default(),
            image_barriers: SmallVec::new(),
            buffer_barriers: SmallVec::new(),
            dependency_flags: vk::DependencyFlags::empty(),
        }
    }
}
impl<T> CommonCommands for T where T: CommandRecorder {}

pub struct ImmediateTransitions<'w> {
    pub(crate) device: &'w Device,
    pub(crate) cmd_buf: vk::CommandBuffer,
    pub(crate) global_barriers: vk::MemoryBarrier2,
    pub(crate) image_barriers: SmallVec<[vk::ImageMemoryBarrier2; 4]>,
    pub(crate) buffer_barriers: SmallVec<[vk::BufferMemoryBarrier2; 4]>,
    pub(crate) dependency_flags: vk::DependencyFlags,
    pub(crate) queue: QueueRef,
}

impl Drop for ImmediateTransitions<'_> {
    fn drop(&mut self) {
        if self.global_barriers.src_access_mask == vk::AccessFlags2::empty()
            && self.global_barriers.dst_access_mask == vk::AccessFlags2::empty()
            && self.global_barriers.src_stage_mask == vk::PipelineStageFlags2::empty()
            && self.global_barriers.dst_stage_mask == vk::PipelineStageFlags2::empty()
            && self.image_barriers.is_empty()
            && self.buffer_barriers.is_empty()
        {
            return;
        }
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


pub trait TrackedResource {
    type State;
    fn transition(&mut self, access: Access, retain_data: bool, next_state: Self::State, commands: &mut impl ResourceTransitionCommands);
}


default impl<T> TrackedResource for RenderRes<T> {
    type State = ();
    fn transition(&mut self, access: Access, _retain_data: bool, _next_state: Self::State, commands: &mut impl ResourceTransitionCommands) {
        let mut semaphore_transitioned = false;
        let (semaphore, value) = commands.signal_semaphore(access.stage);
        if access.is_readonly() {
            self.state.add_read_semaphore(semaphore, value);
            if let Some((write_semaphore, value)) = &self.state.write_semaphore {
                semaphore_transitioned |=
                commands.wait_semaphore(Cow::Borrowed(write_semaphore), *value, access.stage);
            }
        } else {
            if let Some((sem, val)) = self.state.write_semaphore.replace((semaphore, value)) {
                semaphore_transitioned |= commands.wait_semaphore(Cow::Owned(sem), val, access.stage);
            }
            for (sem, val) in self.state.read_semaphores.drain(..) {
                semaphore_transitioned |= commands.wait_semaphore(Cow::Owned(sem), val, access.stage);
            }
        }

        let barrier = self.state.transition(access, false);
        if !semaphore_transitioned {
            commands.add_global_barrier(vk::MemoryBarrier2 {
                src_stage_mask: barrier.src_stage_mask,
                dst_stage_mask: barrier.dst_stage_mask,
                src_access_mask: barrier.src_access_mask,
                dst_access_mask: barrier.dst_access_mask,
                ..Default::default()
            });
        }
    }
}


impl<T> TrackedResource for RenderRes<T> where T: BufferLike {
    type State = ();
    fn transition(&mut self, access: Access, retain_data: bool, _next_state: Self::State, commands: &mut impl ResourceTransitionCommands) {
        let barrier = self.state.transition(access, false);
        let has_queue_family_ownership_transfer =
            if let Some(queue_family) = self.state.queue_family {
                queue_family != commands.current_queue()
            } else {
                false
            };

        let mut semaphore_transition = false;
        let (semaphore, value) = commands.signal_semaphore(access.stage);
        if access.is_readonly() {
            self.state.add_read_semaphore(semaphore, value);
            if let Some((write_semaphore, value)) = &self.state.write_semaphore {
                semaphore_transition |=
                commands.wait_semaphore(Cow::Borrowed(write_semaphore), *value, access.stage);
            }
        } else {
            if let Some((sem, val)) = self
                .state
                .write_semaphore
                .replace((semaphore.clone(), value))
            {
                semaphore_transition |= commands.wait_semaphore(Cow::Owned(sem), val, access.stage);
            }
            for (sem, val) in self.state.read_semaphores.drain(..) {
                semaphore_transition |= commands.wait_semaphore(Cow::Owned(sem), val, access.stage);
            }
        }

        if has_queue_family_ownership_transfer && retain_data {
            commands.add_buffer_barrier(vk::BufferMemoryBarrier2 {
                dst_stage_mask: barrier.dst_stage_mask,
                dst_access_mask: barrier.dst_access_mask,
                src_queue_family_index: self.state.queue_family.unwrap().family,
                dst_queue_family_index: commands.current_queue().family,
                buffer: self.raw_buffer(),
                offset: self.offset(),
                size: self.size(),
                ..Default::default()
            });
            commands.add_buffer_barrier_prev_stage(
                vk::BufferMemoryBarrier2 {
                    src_stage_mask: barrier.src_stage_mask,
                    src_access_mask: barrier.src_access_mask,
                    src_queue_family_index: self.state.queue_family.unwrap().family,
                    dst_queue_family_index: commands.current_queue().family,
                    buffer: self.raw_buffer(),
                    offset: self.offset(),
                    size: self.size(),
                    ..Default::default()
                },
                self.state.queue_family.unwrap(),
            );
        } else if !semaphore_transition {
            commands.add_global_barrier(vk::MemoryBarrier2 {
                src_stage_mask: barrier.src_stage_mask,
                dst_stage_mask: barrier.dst_stage_mask,
                src_access_mask: barrier.src_access_mask,
                dst_access_mask: barrier.dst_access_mask,
                ..Default::default()
            });
        }
        self.state.queue_family = Some(commands.current_queue());

    }
}

impl<T> TrackedResource for RenderImage<T> where T: ImageLike {
    type State = vk::ImageLayout;
    fn transition(&mut self, access: Access, retain_data: bool, layout: Self::State, commands: &mut impl ResourceTransitionCommands) {
        if access.is_readonly() && !retain_data {
            tracing::warn!("Transitioning an image to readonly access without retaining image data. This is likely an error.");
        }
        if access.is_writeonly() && retain_data {
            tracing::warn!("Transitioning an image to writeonly access while retaining image data. This is likely inefficient.");
        }
        let has_layout_transition = self.layout != layout;
        let has_queue_family_ownership_transfer =
            if let Some(queue_family) = self.res.state.queue_family {
                queue_family.family != commands.current_queue().family
            } else {
                false
            };
        let barrier = self.res.state.transition(access, has_layout_transition);

        let (semaphore, value) = commands.signal_semaphore(access.stage);
        let mut semaphore_transitioned = false;
        if access.is_readonly() {
            self.res.state.add_read_semaphore(semaphore, value);
            if let Some((write_semaphore, value)) = &self.res.state.write_semaphore {
                semaphore_transitioned |=
                commands.wait_semaphore(Cow::Borrowed(write_semaphore), *value, access.stage);
            }
        } else {
            if let Some((sem, val)) = self
                .res
                .state
                .write_semaphore
                .replace((semaphore.clone(), value))
            {
                semaphore_transitioned |= commands.wait_semaphore(Cow::Owned(sem), val, access.stage);
            }
            for (sem, val) in self.res.state.read_semaphores.drain(..) {
                semaphore_transitioned |= commands.wait_semaphore(Cow::Owned(sem), val, access.stage);
            }
        }

        if has_layout_transition || (has_queue_family_ownership_transfer && retain_data) {
            let mut barrier = vk::ImageMemoryBarrier2 {
                src_stage_mask: barrier.src_stage_mask,
                dst_stage_mask: barrier.dst_stage_mask,
                src_access_mask: barrier.src_access_mask,
                dst_access_mask: barrier.dst_access_mask,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: layout,
                image: self.raw_image(),
                subresource_range: self.subresource_range(),
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                ..Default::default()
            };
            if retain_data {
                barrier.old_layout = self.layout;
            }
            if retain_data && has_queue_family_ownership_transfer {
                barrier.src_queue_family_index = self.res.state.queue_family.unwrap().family;
                barrier.dst_queue_family_index = commands.current_queue().family;

                commands.add_image_barrier_prev_stage(
                    vk::ImageMemoryBarrier2 {
                        src_stage_mask: barrier.src_stage_mask,
                        src_access_mask: barrier.src_access_mask,
                        dst_stage_mask: if has_layout_transition {
                            // If we're transitioning the layout, layout transition needs to block semaphore signal operation.
                            // This is only applicable to QueueSubmit2. In the case of QueueSubmit1,
                            // the semaphore signal operation will happen after all other pipeline stages, and we can safely set this to
                            // BOTTOM_OF_PIPE at all times.
                            barrier.src_stage_mask
                        } else {
                            vk::PipelineStageFlags2::empty()
                        },
                        src_queue_family_index: self.state.queue_family.unwrap().family,
                        dst_queue_family_index: commands.current_queue().family,
                        image: self.raw_image(),
                        subresource_range: self.subresource_range(),
                        new_layout: barrier.new_layout,
                        old_layout: barrier.old_layout,
                        ..Default::default()
                    },
                    self.state.queue_family.unwrap(),
                );
                barrier.src_access_mask = vk::AccessFlags2::empty();
                // Block the same stage as the layout transition so that the layout transition happens after the semaphore wait.
                barrier.src_stage_mask = barrier.dst_stage_mask;
            }
            commands.add_image_barrier(barrier);
            self.layout = layout;
        } else if !semaphore_transitioned {
            commands.add_global_barrier(vk::MemoryBarrier2 {
                src_stage_mask: barrier.src_stage_mask,
                dst_stage_mask: barrier.dst_stage_mask,
                src_access_mask: barrier.src_access_mask,
                dst_access_mask: barrier.dst_access_mask,
                ..Default::default()
            });
        }
        self.res.state.queue_family = Some(commands.current_queue());

    }
}


pub trait ResourceTransitionCommands: Sized {
    fn add_image_barrier_prev_stage(
        &mut self,
        barrier: vk::ImageMemoryBarrier2,
        prev_queue: QueueRef,
    ) -> &mut Self;
    fn add_buffer_barrier_prev_stage(
        &mut self,
        barrier: vk::BufferMemoryBarrier2,
        prev_queue: QueueRef,
    ) -> &mut Self;

    fn add_global_barrier(&mut self, barrier: vk::MemoryBarrier2) -> &mut Self;
    fn add_image_barrier(&mut self, barrier: vk::ImageMemoryBarrier2) -> &mut Self;
    fn add_buffer_barrier(&mut self, barrier: vk::BufferMemoryBarrier2) -> &mut Self;
    fn set_dependency_flags(&mut self, flags: vk::DependencyFlags) -> &mut Self;

    fn current_queue(&self) -> QueueRef;

    /// Specify that the current submission must wait on a semaphore before executing.
    fn wait_semaphore(
        &mut self,
        semaphore: Cow<Arc<TimelineSemaphore>>,
        value: u64,
        stage: vk::PipelineStageFlags2,
    ) -> bool;
    fn wait_binary_semaphore(
        &mut self,
        semaphore: vk::Semaphore,
        stage: vk::PipelineStageFlags2,
    );
    /// Ask the current submission to signal a semaphore after executing.
    fn signal_semaphore(&mut self, stage: vk::PipelineStageFlags2)
        -> (Arc<TimelineSemaphore>, u64);
    fn signal_binary_semaphore_prev_stage(
        &mut self,
        semaphore: vk::Semaphore,
        stage: vk::PipelineStageFlags2,
        prev_queue: QueueRef,
    );
    fn wait_binary_semaphore_prev_stage(
        &mut self,
        semaphore: vk::Semaphore,
        stage: vk::PipelineStageFlags2,
        prev_queue: QueueRef,
    );

    fn transition<T: TrackedResource>(&mut self, res: &mut T, access: Access, retain_data: bool, next_state: T::State) -> &mut Self {
        res.transition(access, retain_data, next_state, self);
        self
    }
}
impl ResourceTransitionCommands for ImmediateTransitions<'_> {
    fn current_queue(&self) -> QueueRef {
        self.queue
    }
    fn add_image_barrier_prev_stage(
        &mut self,
        barrier: vk::ImageMemoryBarrier2,
        queue_type: QueueRef,
    ) -> &mut Self {
        panic!()
    }
    fn add_buffer_barrier_prev_stage(
        &mut self,
        barrier: vk::BufferMemoryBarrier2,
        queue_type: QueueRef,
    ) -> &mut Self {
        panic!()
    }
    fn add_global_barrier(&mut self, barrier: vk::MemoryBarrier2) -> &mut Self {
        self.global_barriers.src_stage_mask |= barrier.src_stage_mask;
        self.global_barriers.dst_stage_mask |= barrier.dst_stage_mask;
        self.global_barriers.src_access_mask |= barrier.src_access_mask;
        self.global_barriers.dst_access_mask |= barrier.dst_access_mask;
        self
    }
    fn add_image_barrier(&mut self, barrier: vk::ImageMemoryBarrier2) -> &mut Self {
        self.image_barriers.push(barrier);
        self
    }
    fn add_buffer_barrier(&mut self, barrier: vk::BufferMemoryBarrier2) -> &mut Self {
        self.buffer_barriers.push(barrier);
        self
    }
    fn set_dependency_flags(&mut self, flags: vk::DependencyFlags) -> &mut Self {
        self.dependency_flags = flags;
        self
    }

    fn wait_semaphore(
        &mut self,
        semaphore: Cow<Arc<TimelineSemaphore>>,
        value: u64,
        stage: vk::PipelineStageFlags2,
    ) -> bool {
        todo!()
    }

    fn signal_semaphore(
        &mut self,
        stage: vk::PipelineStageFlags2,
    ) -> (Arc<TimelineSemaphore>, u64) {
        todo!()
    }
    fn wait_binary_semaphore(
            &mut self,
            semaphore: vk::Semaphore,
            stage: vk::PipelineStageFlags2,
        ) {
        todo!()
    }
    fn signal_binary_semaphore_prev_stage(
        &mut self,
        semaphore: vk::Semaphore,
        stage: vk::PipelineStageFlags2,
        prev_queue: QueueRef,
    ) {
        todo!()
    }
    fn wait_binary_semaphore_prev_stage(
            &mut self,
            semaphore: vk::Semaphore,
            stage: vk::PipelineStageFlags2,
            prev_queue: QueueRef,
        ) {
        todo!()
    }
}
