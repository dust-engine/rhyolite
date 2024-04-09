//! Async compute tasks

use std::{mem::ManuallyDrop, ops::Deref, sync::atomic::AtomicU32};

use ash::vk;
use bevy::ecs::world::{FromWorld, World};

use crate::{
    command_pool::CommandPool, commands::CommandRecorder, Device, HasDevice, QueueRef, Queues,
    QUEUE_FLAGS_ASYNC,
};
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
    pub fn record(&mut self) -> AsyncComputeCommandRecorder {
        let cmd_buf = unsafe { self.command_pool.allocate().unwrap() };
        AsyncComputeCommandRecorder {
            cmd_buf,
            task_pool: self,
        }
    }
    pub fn wait_blocked<T>(&mut self, task: AsyncComputeTask<T>) -> T {
        unsafe {
            self.command_pool.device().wait_for_fences(&[task.fence], true, u64::MAX).unwrap();
            self.command_pool.device().reset_fences(&[task.fence]).unwrap();
            self.fences.push(task.fence);
            self.command_pool.free(&[task.cmd_buf]);
            std::mem::forget(task.drop_marker);
        }
        task.result
    }
}

pub struct AsyncComputeTask<T> {
    fence: vk::Fence,
    cmd_buf: vk::CommandBuffer,
    result: T,
    drop_marker: AsyncComputeDropMarker,
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
                .command_pool.device()
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
        AsyncComputeTask { fence, result, cmd_buf: self.cmd_buf, drop_marker: AsyncComputeDropMarker }
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
