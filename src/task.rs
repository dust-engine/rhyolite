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

/// Task pool for async tasks to be performed on the GPU.
/// Supports compute or transfer workloads.
///
/// If the compute or transfer queue families support more than one [`vk::Queue`],
/// queue submits will be made to a dedicated [`vk::Queue`]. If only one [`vk::Queue`]
/// was supported, queue submits will be syncronized with a mutex.
#[derive(Resource)]
pub struct AsyncTaskPool {
    device: Device,
    queues: Queues,

    transfer_command_pool: Option<CommandPool>,
    transfer_queue_ref: Option<QueueRef>,

    compute_command_pool: CommandPool,
    compute_queue_ref: QueueRef,

    semaphores: Vec<Arc<TimelineSemaphore>>,
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
    fn get_semaphore(&mut self) -> Arc<TimelineSemaphore> {
        self.semaphores
            .pop()
            .unwrap_or_else(|| Arc::new(TimelineSemaphore::new(self.device.clone()).unwrap()))
    }
    fn get_queue<const Q: char>(&self) -> QueueRef {
        if Q == 'c' {
            self.compute_queue_ref
        } else {
            self.transfer_queue_ref.unwrap_or(self.compute_queue_ref)
        }
    }
    pub fn spawn_compute(&mut self) -> AsyncCommandRecorder<'c'> {
        AsyncCommandRecorder {
            cmd_buf: vk::CommandBuffer::null(),
            task_pool: self,
            semaphore: None,
            all_compute_cmd_buf: Vec::new(),
            all_transfer_cmd_buf: Vec::new(),
            drop_prohibited: AsyncComputeDropMarker(false),
            semaphore_wait_value: 0,
        }
    }
    pub fn spawn_transfer(&mut self) -> AsyncCommandRecorder<'t'> {
        AsyncCommandRecorder {
            cmd_buf: vk::CommandBuffer::null(),
            task_pool: self,
            semaphore: None,
            all_compute_cmd_buf: Vec::new(),
            all_transfer_cmd_buf: Vec::new(),
            drop_prohibited: AsyncComputeDropMarker(false),
            semaphore_wait_value: 0,
        }
    }
    pub fn wait_blocked<T>(&mut self, task: AsyncComputeTask<T>) -> T {
        let Some(semaphore) = task.semaphore else {
            assert!(task.compute_cmd_buf.is_empty());
            assert!(task.transfer_cmd_buf.is_empty());
            return task.result;
        };
        semaphore
            .wait_blocked(task.semaphore_wait_value, !0)
            .unwrap();
        unsafe {
            self.semaphores.push(semaphore);

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
    compute_cmd_buf: Vec<vk::CommandBuffer>,
    transfer_cmd_buf: Vec<vk::CommandBuffer>,
    semaphore: Option<Arc<TimelineSemaphore>>,
    semaphore_wait_value: u64,
    result: T,
    drop_marker: AsyncComputeDropMarker,
}
impl<T> AsyncComputeTask<T> {
    pub fn is_finished(&self) -> bool {
        if let Some(semaphore) = &self.semaphore {
            semaphore.is_signaled(self.semaphore_wait_value)
        } else {
            true
        }
    }
}
struct AsyncComputeDropMarker(bool);

impl Drop for AsyncComputeDropMarker {
    fn drop(&mut self) {
        if self.0 {
            tracing::warn!("AsyncComputeTask dropped without waiting for completion");
        }
    }
}

pub struct AsyncCommandRecorder<'a, const Q: char> {
    task_pool: &'a mut AsyncTaskPool,
    cmd_buf: vk::CommandBuffer,
    semaphore: Option<Arc<TimelineSemaphore>>,
    semaphore_wait_value: u64,

    all_compute_cmd_buf: Vec<vk::CommandBuffer>,
    all_transfer_cmd_buf: Vec<vk::CommandBuffer>,
    drop_prohibited: AsyncComputeDropMarker,
}
impl<'a, const Q: char> AsyncCommandRecorder<'a, Q> {
    /// Submit the existing commands in the command buffer, and return a new command buffer for a different queue family.
    /// - `wait_stages`: These stages in the current command recorder will wait for the completion of the previous command recorder.
    /// - `signal_stages`: The next command recorder will begin after the completion of these stages in the current command recorder.
    pub fn commit<const NEXT_Q: char>(
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
                semaphore: self.semaphore,
                semaphore_wait_value: self.semaphore_wait_value,
                drop_prohibited: self.drop_prohibited,
                all_compute_cmd_buf: self.all_compute_cmd_buf,
                all_transfer_cmd_buf: self.all_transfer_cmd_buf,
            };
        }

        // Split queue.
        if self.cmd_buf != vk::CommandBuffer::null() {
            let semaphore = self.semaphore.take().unwrap_or_else(|| {
                let semaphore = self.task_pool.get_semaphore();
                self.semaphore_wait_value = semaphore.value();
                semaphore
            });
            let queue = self.task_pool.get_queue::<Q>();
            let queue = self.task_pool.queues.get(queue);
            unsafe {
                let wait_semaphore_infos = [vk::SemaphoreSubmitInfo {
                    semaphore: semaphore.raw(),
                    value: self.semaphore_wait_value,
                    ..Default::default()
                }];
                self.task_pool
                    .device
                    .end_command_buffer(self.cmd_buf)
                    .unwrap();
                self.task_pool
                    .device
                    .queue_submit2(
                        *queue,
                        &[vk::SubmitInfo2::default()
                            .command_buffer_infos(&[vk::CommandBufferSubmitInfo {
                                command_buffer: self.cmd_buf,
                                ..Default::default()
                            }])
                            .wait_semaphore_infos(if self.semaphore_wait_value == 0 {
                                &[]
                            } else {
                                &wait_semaphore_infos
                            })
                            .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo {
                                semaphore: semaphore.raw(),
                                stage_mask: signal_stages,
                                value: self.semaphore_wait_value + 1,
                                ..Default::default()
                            }])],
                        vk::Fence::null(),
                    )
                    .unwrap();
            }
            drop(queue);
            self.semaphore = Some(semaphore);
            if Q == 't' && self.task_pool.transfer_command_pool.is_some() {
                self.all_transfer_cmd_buf.push(self.cmd_buf);
            } else {
                self.all_compute_cmd_buf.push(self.cmd_buf);
            }
        } else {
            assert!(self.semaphore.is_none());
            tracing::warn!("Nothing was commited");
        }
        AsyncCommandRecorder {
            task_pool: self.task_pool,
            cmd_buf: vk::CommandBuffer::null(),
            semaphore: self.semaphore,
            semaphore_wait_value: self.semaphore_wait_value + 1,
            drop_prohibited: self.drop_prohibited,
            all_compute_cmd_buf: self.all_compute_cmd_buf,
            all_transfer_cmd_buf: self.all_transfer_cmd_buf,
        }
    }
    pub fn finish<T>(mut self, result: T, stages: vk::PipelineStageFlags2) -> AsyncComputeTask<T> {
        if self.cmd_buf == vk::CommandBuffer::null() {
            tracing::warn!("Nothing to finish");
            assert!(self.semaphore.is_none());
            return AsyncComputeTask {
                semaphore: None,
                result,
                compute_cmd_buf: self.all_compute_cmd_buf,
                transfer_cmd_buf: self.all_transfer_cmd_buf,
                drop_marker: self.drop_prohibited,
                semaphore_wait_value: 0,
            };
        }
        let semaphore = self.semaphore.take().unwrap_or_else(|| {
            let semaphore = self.task_pool.get_semaphore();
            self.semaphore_wait_value = semaphore.value();
            semaphore
        });
        let queue = self.task_pool.get_queue::<Q>();
        let queue = self.task_pool.queues.get(queue);
        unsafe {
            self.task_pool
                .device
                .end_command_buffer(self.cmd_buf)
                .unwrap();
            let wait_semaphore_infos = [vk::SemaphoreSubmitInfo {
                semaphore: semaphore.raw(),
                stage_mask: stages,
                value: self.semaphore_wait_value,
                ..Default::default()
            }];
            self.task_pool
                .device
                .queue_submit2(
                    *queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo {
                            command_buffer: self.cmd_buf,
                            ..Default::default()
                        }])
                        .wait_semaphore_infos(if self.semaphore_wait_value == 0 {
                            &[]
                        } else {
                            &wait_semaphore_infos
                        })
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo {
                            semaphore: semaphore.raw(),
                            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                            value: self.semaphore_wait_value + 1,
                            ..Default::default()
                        }])],
                    vk::Fence::null(),
                )
                .unwrap();
        }

        if Q == 't' && self.task_pool.transfer_command_pool.is_some() {
            self.all_transfer_cmd_buf.push(self.cmd_buf);
        } else {
            self.all_compute_cmd_buf.push(self.cmd_buf);
        }
        AsyncComputeTask {
            result,
            compute_cmd_buf: self.all_compute_cmd_buf,
            transfer_cmd_buf: self.all_transfer_cmd_buf,
            semaphore: Some(semaphore),
            drop_marker: self.drop_prohibited,
            semaphore_wait_value: self.semaphore_wait_value + 1,
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
        if self.cmd_buf == vk::CommandBuffer::null() {
            unsafe {
                let cmd_buf = self.task_pool.get_command_pool::<Q>().allocate().unwrap();
                self.task_pool
                    .device
                    .begin_command_buffer(
                        cmd_buf,
                        &vk::CommandBufferBeginInfo {
                            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                self.cmd_buf = cmd_buf;
            }
        }
        self.drop_prohibited.0 = true;
        self.cmd_buf
    }
    fn current_queue(&self) -> QueueRef {
        self.task_pool.get_queue::<Q>()
    }
    fn semaphore_signal(&mut self) -> &mut impl SemaphoreSignalCommands {
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
        _stage: vk::PipelineStageFlags2,
    ) -> (Arc<TimelineSemaphore>, u64) {
        let semaphore = self.semaphore.take().unwrap_or_else(|| {
            let semaphore = self.task_pool.get_semaphore();
            self.semaphore_wait_value = semaphore.value();
            semaphore
        });
        (semaphore, self.semaphore_wait_value + 1)
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
