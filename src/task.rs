//! Async compute tasks

use std::{
    borrow::Cow,
    mem::ManuallyDrop,
    ops::Deref,
    sync::{atomic::AtomicU32, Arc},
};

use ash::vk;
use bevy::ecs::{
    system::Resource,
    world::{FromWorld, World},
};

use crate::{
    command_pool::CommandPool,
    commands::{CommandRecorder, SemaphoreSignalCommands},
    semaphore::TimelineSemaphore,
    Device, HasDevice, QueueRef, Queues, QUEUE_FLAGS_ASYNC,
};

#[derive(Resource)]
pub struct AsyncComputeTaskPool {
    queues: Queues,
    command_pool: CommandPool,
    queue_ref: QueueRef,
    fences: Vec<vk::Fence>,
}

impl FromWorld for AsyncComputeTaskPool {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<Device>();
        let queues = world.resource::<Queues>();
        let compute_queue = queues
            .with_caps(
                vk::QueueFlags::COMPUTE | QUEUE_FLAGS_ASYNC,
                vk::QueueFlags::empty(),
            )
            .unwrap();
        let command_pool = CommandPool::new(device.clone(), compute_queue.family).unwrap();
        Self {
            command_pool,
            queues: queues.clone(),
            queue_ref: compute_queue,
            fences: Vec::new(),
        }
    }
}

impl AsyncComputeTaskPool {
    pub fn spawn(&mut self) -> AsyncComputeCommandRecorder {
        let cmd_buf = unsafe { self.command_pool.allocate().unwrap() };
        AsyncComputeCommandRecorder {
            cmd_buf,
            task_pool: self,
        }
    }
    pub fn wait_blocked<T>(&mut self, task: AsyncComputeTask<T>) -> T {
        unsafe {
            self.command_pool
                .device()
                .wait_for_fences(&[task.fence], true, u64::MAX)
                .unwrap();
            self.command_pool
                .device()
                .reset_fences(&[task.fence])
                .unwrap();
            self.fences.push(task.fence);
            self.command_pool.free(&[task.cmd_buf]);
            std::mem::forget(task.drop_marker);
        }
        task.result
    }
}

pub struct AsyncComputeTask<T> {
    device: Device,
    fence: vk::Fence,
    cmd_buf: vk::CommandBuffer,
    result: T,
    drop_marker: AsyncComputeDropMarker,
}
impl<T> AsyncComputeTask<T> {
    pub fn is_finished(&self) -> bool {
        unsafe {
            self.device
                .get_fence_status(self.fence)
                .unwrap()
        }
    }
}
struct AsyncComputeDropMarker;

impl Drop for AsyncComputeDropMarker {
    fn drop(&mut self) {
        panic!()
    }
}

pub struct AsyncComputeCommandRecorder<'a> {
    task_pool: &'a mut AsyncComputeTaskPool,
    cmd_buf: vk::CommandBuffer,
}
impl AsyncComputeCommandRecorder<'_> {
    pub fn finish<T>(self, result: T) -> AsyncComputeTask<T> {
        let fence = self.task_pool.fences.pop().unwrap_or_else(|| unsafe {
            self.task_pool
                .command_pool
                .device()
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap()
        });
        let queue = self.task_pool.queues.get(self.task_pool.queue_ref);
        unsafe {
            self.task_pool
                .command_pool
                .device()
                .queue_submit(
                    *queue,
                    &[vk::SubmitInfo {
                        command_buffer_count: 1,
                        p_command_buffers: &self.cmd_buf,
                        ..Default::default()
                    }],
                    fence,
                )
                .unwrap();
        }
        AsyncComputeTask {
            fence,
            result,
            cmd_buf: self.cmd_buf,
            drop_marker: AsyncComputeDropMarker,
            device: self.task_pool.command_pool.device().clone(),
        }
    }
}
impl HasDevice for AsyncComputeCommandRecorder<'_> {
    fn device(&self) -> &Device {
        self.task_pool.command_pool.device()
    }
}
impl Drop for AsyncComputeCommandRecorder<'_> {
    fn drop(&mut self) {
        self.task_pool.command_pool.free(&[self.cmd_buf]);
    }
}
impl CommandRecorder for AsyncComputeCommandRecorder<'_> {
    const QUEUE_CAP: char = 'c';
    fn cmd_buf(&mut self) -> vk::CommandBuffer {
        self.cmd_buf
    }
    fn current_queue(&self) -> QueueRef {
        self.task_pool.queue_ref
    }
    fn semaphore_signal(&mut self) -> &mut impl SemaphoreSignalCommands {
        todo!();
        self
    }
}

impl SemaphoreSignalCommands for AsyncComputeCommandRecorder<'_> {
    fn wait_semaphore(
        &mut self,
        semaphore: Cow<Arc<TimelineSemaphore>>,
        value: u64,
        stage: vk::PipelineStageFlags2,
    ) -> bool {
        todo!()
    }

    fn wait_binary_semaphore(&mut self, semaphore: vk::Semaphore, stage: vk::PipelineStageFlags2) {
        todo!()
    }

    fn signal_semaphore(
        &mut self,
        stage: vk::PipelineStageFlags2,
    ) -> (Arc<TimelineSemaphore>, u64) {
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
