//! Async compute tasks

use std::{borrow::Cow, sync::Arc};

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
pub struct AsyncTaskPool {
    device: Device,
    queues: Queues,

    transfer_command_pool: Option<CommandPool>,
    transfer_queue_ref: Option<QueueRef>,

    compute_command_pool: CommandPool,
    compute_queue_ref: QueueRef,

    fences: Vec<vk::Fence>,
    semaphores: Vec<vk::Semaphore>,
}

impl FromWorld for AsyncTaskPool {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<Device>();
        let queues = world.resource::<Queues>();
        let compute_queue = queues
            .with_caps(
                vk::QueueFlags::COMPUTE | QUEUE_FLAGS_ASYNC,
                vk::QueueFlags::empty(),
            )
            .unwrap();
        let transfer_queue = queues
            .with_caps(
                vk::QueueFlags::TRANSFER | QUEUE_FLAGS_ASYNC,
                vk::QueueFlags::empty(),
            )
            .unwrap();
        let command_pool = CommandPool::new(device.clone(), compute_queue.family).unwrap();
        let mut this = Self {
            compute_command_pool: command_pool,
            compute_queue_ref: compute_queue,
            queues: queues.clone(),
            transfer_command_pool: None,
            transfer_queue_ref: None,
            device: device.clone(),
            semaphores: Vec::new(),
            fences: Vec::new(),
        };
        if transfer_queue.index != compute_queue.index {
            assert_ne!(transfer_queue.family, compute_queue.family);
            this.transfer_command_pool =
                Some(CommandPool::new(device.clone(), transfer_queue.family).unwrap());
            this.transfer_queue_ref = Some(transfer_queue);
        }
        this
    }
}

impl AsyncTaskPool {
    fn get_command_pool<const Q: char>(&mut self) -> &mut CommandPool {
        if Q == 'c' {
            &mut self.compute_command_pool
        } else {
            self.transfer_command_pool
                .as_mut()
                .unwrap_or(&mut self.compute_command_pool)
        }
    }
    fn get_queue<const Q: char>(&self) -> QueueRef {
        if Q == 'c' {
            self.compute_queue_ref
        } else {
            self.transfer_queue_ref.unwrap_or(self.compute_queue_ref)
        }
    }
    pub fn spawn_compute(&mut self) -> AsyncCommandRecorder<'c'> {
        let command_pool = self.get_command_pool::<'c'>();
        let cmd_buf = unsafe { command_pool.allocate().unwrap() };
        unsafe {
            self.device
                .begin_command_buffer(
                    cmd_buf,
                    &vk::CommandBufferBeginInfo {
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        AsyncCommandRecorder {
            cmd_buf,
            task_pool: self,
            wait_semaphore: vk::Semaphore::null(),
            all_semaphores: Vec::new(),
            all_compute_cmd_buf: Vec::new(),
            all_transfer_cmd_buf: Vec::new(),
            drop_marker: AsyncComputeDropMarker,
        }
    }
    pub fn spawn_transfer(&mut self) -> AsyncCommandRecorder<'t'> {
        let command_pool = self.get_command_pool::<'t'>();
        let cmd_buf = unsafe { command_pool.allocate().unwrap() };
        unsafe {
            self.device
                .begin_command_buffer(
                    cmd_buf,
                    &vk::CommandBufferBeginInfo {
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        AsyncCommandRecorder {
            cmd_buf,
            task_pool: self,
            wait_semaphore: vk::Semaphore::null(),
            all_semaphores: Vec::new(),
            all_compute_cmd_buf: Vec::new(),
            all_transfer_cmd_buf: Vec::new(),
            drop_marker: AsyncComputeDropMarker,
        }
    }
    pub fn wait_blocked<T>(&mut self, task: AsyncComputeTask<T>) -> T {
        unsafe {
            self.device
                .wait_for_fences(&[task.fence], true, u64::MAX)
                .unwrap();
            self.device.reset_fences(&[task.fence]).unwrap();
            self.fences.push(task.fence);
            self.semaphores.extend(task.semaphores);

            self.compute_command_pool.free(&task.compute_cmd_buf);
            if let Some(transfer_command_pool) = &mut self.transfer_command_pool {
                transfer_command_pool.free(&task.transfer_cmd_buf);
            } else {
                assert!(task.transfer_cmd_buf.is_empty());
            }
            std::mem::forget(task.drop_marker);
        }
        task.result
    }
}

pub struct AsyncComputeTask<T> {
    device: Device,
    fence: vk::Fence,
    compute_cmd_buf: Vec<vk::CommandBuffer>,
    transfer_cmd_buf: Vec<vk::CommandBuffer>,
    semaphores: Vec<vk::Semaphore>,
    result: T,
    drop_marker: AsyncComputeDropMarker,
}
impl<T> AsyncComputeTask<T> {
    pub fn is_finished(&self) -> bool {
        unsafe { self.device.get_fence_status(self.fence).unwrap() }
    }
}
struct AsyncComputeDropMarker;

impl Drop for AsyncComputeDropMarker {
    fn drop(&mut self) {
        panic!()
    }
}

pub struct AsyncCommandRecorder<'a, const Q: char> {
    task_pool: &'a mut AsyncTaskPool,
    cmd_buf: vk::CommandBuffer,
    wait_semaphore: vk::Semaphore,

    all_semaphores: Vec<vk::Semaphore>,
    all_compute_cmd_buf: Vec<vk::CommandBuffer>,
    all_transfer_cmd_buf: Vec<vk::CommandBuffer>,
    drop_marker: AsyncComputeDropMarker,
}
impl<'a, const Q: char> AsyncCommandRecorder<'a, Q> {
    pub fn commit<T, const NEXT_Q: char>(
        mut self,
        wait_stages: vk::PipelineStageFlags2,
        signal_stages: vk::PipelineStageFlags2,
    ) -> AsyncCommandRecorder<'a, NEXT_Q> {
        if Q == NEXT_Q {
            tracing::warn!("Unnecessary commit");
        }
        if self.task_pool.transfer_command_pool.is_none() {
            // Unified queue.
            return AsyncCommandRecorder {
                task_pool: self.task_pool,
                cmd_buf: self.cmd_buf,
                wait_semaphore: self.wait_semaphore,
                drop_marker: self.drop_marker,
                all_semaphores: self.all_semaphores,
                all_compute_cmd_buf: self.all_compute_cmd_buf,
                all_transfer_cmd_buf: self.all_transfer_cmd_buf,
            };
        }
        // Split queue.

        let signal_semaphore = self.task_pool.semaphores.pop().unwrap_or_else(|| unsafe {
            self.task_pool
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap()
        });
        let queue = self.task_pool.get_queue::<Q>();
        let queue = self.task_pool.queues.get(queue);
        unsafe {
            self.task_pool
                .device
                .end_command_buffer(self.cmd_buf)
                .unwrap();
            self.task_pool
                .device
                .queue_submit2(
                    *queue,
                    &[vk::SubmitInfo2 {
                        command_buffer_info_count: 1,
                        p_command_buffer_infos: &[vk::CommandBufferSubmitInfo {
                            command_buffer: self.cmd_buf,
                            ..Default::default()
                        }] as *const _,
                        signal_semaphore_info_count: 1,
                        p_signal_semaphore_infos: &[vk::SemaphoreSubmitInfo {
                            semaphore: signal_semaphore,
                            stage_mask: signal_stages,
                            ..Default::default()
                        }] as *const _,
                        p_wait_semaphore_infos: &[vk::SemaphoreSubmitInfo {
                            semaphore: self.wait_semaphore,
                            stage_mask: wait_stages,
                            ..Default::default()
                        }] as *const _,
                        wait_semaphore_info_count: if self.wait_semaphore != vk::Semaphore::null() {
                            1
                        } else {
                            0
                        },
                        ..Default::default()
                    }],
                    vk::Fence::null(),
                )
                .unwrap();
        }
        drop(queue);
        self.all_semaphores.push(signal_semaphore);
        if Q == 't' && self.task_pool.transfer_command_pool.is_some() {
            self.all_transfer_cmd_buf.push(self.cmd_buf);
        } else {
            self.all_compute_cmd_buf.push(self.cmd_buf);
        }
        let new_command_buffer = unsafe {
            self.task_pool
                .get_command_pool::<NEXT_Q>()
                .allocate()
                .unwrap()
        };
        unsafe {
            self.task_pool
                .device
                .begin_command_buffer(
                    new_command_buffer,
                    &vk::CommandBufferBeginInfo {
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        AsyncCommandRecorder {
            task_pool: self.task_pool,
            cmd_buf: new_command_buffer,
            wait_semaphore: signal_semaphore,
            drop_marker: self.drop_marker,
            all_compute_cmd_buf: self.all_compute_cmd_buf,
            all_transfer_cmd_buf: self.all_transfer_cmd_buf,
            all_semaphores: self.all_semaphores,
        }
    }
    pub fn finish<T>(mut self, result: T, stages: vk::PipelineStageFlags2) -> AsyncComputeTask<T> {
        let fence = self.task_pool.fences.pop().unwrap_or_else(|| unsafe {
            self.task_pool
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap()
        });
        let queue = self.task_pool.get_queue::<Q>();
        let queue = self.task_pool.queues.get(queue);
        unsafe {
            self.task_pool
                .device
                .end_command_buffer(self.cmd_buf)
                .unwrap();
            self.task_pool
                .device
                .queue_submit2(
                    *queue,
                    &[vk::SubmitInfo2 {
                        command_buffer_info_count: 1,
                        p_command_buffer_infos: &[vk::CommandBufferSubmitInfo {
                            command_buffer: self.cmd_buf,
                            ..Default::default()
                        }] as *const _,
                        wait_semaphore_info_count: if self.wait_semaphore != vk::Semaphore::null() {
                            1
                        } else {
                            0
                        },
                        p_wait_semaphore_infos: &[vk::SemaphoreSubmitInfo {
                            semaphore: self.wait_semaphore,
                            stage_mask: stages,
                            ..Default::default()
                        }] as *const _,
                        ..Default::default()
                    }],
                    fence,
                )
                .unwrap();
        }

        if Q == 't' && self.task_pool.transfer_command_pool.is_some() {
            self.all_transfer_cmd_buf.push(self.cmd_buf);
        } else {
            self.all_compute_cmd_buf.push(self.cmd_buf);
        }
        AsyncComputeTask {
            fence,
            result,
            compute_cmd_buf: self.all_compute_cmd_buf,
            transfer_cmd_buf: self.all_transfer_cmd_buf,
            semaphores: self.all_semaphores,
            drop_marker: self.drop_marker,
            device: self.task_pool.device.clone(),
        }
    }
}
impl<const Q: char> HasDevice for AsyncCommandRecorder<'_, Q> {
    fn device(&self) -> &Device {
        &self.task_pool.device
    }
}
impl<const Q: char> CommandRecorder for AsyncCommandRecorder<'_, Q> {
    const QUEUE_CAP: char = Q;
    fn cmd_buf(&mut self) -> vk::CommandBuffer {
        self.cmd_buf
    }
    fn current_queue(&self) -> QueueRef {
        self.task_pool.get_queue::<Q>()
    }
    fn semaphore_signal(&mut self) -> &mut impl SemaphoreSignalCommands {
        todo!();
        self
    }
}

impl<const Q: char> SemaphoreSignalCommands for AsyncCommandRecorder<'_, Q> {
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
