use crate::{utils::unwrap_future, Device};
use ash::khr::deferred_host_operations::Meta as DeferredHostOperations;
use ash::{
    prelude::VkResult,
    vk::{self},
};
use bevy::ecs::{system::Resource, world::FromWorld};
use crossbeam_channel::Sender;
use std::{
    mem::ManuallyDrop,
    sync::{
        atomic::{AtomicBool, AtomicU32},
        Arc,
    },
    thread::JoinHandle,
};

struct DeferredHostOperation {
    device: Device,
    raw: vk::DeferredOperationKHR,
}
impl Drop for DeferredHostOperation {
    fn drop(&mut self) {
        unsafe {
            self.device
                .extension::<DeferredHostOperations>()
                .destroy_deferred_operation(self.raw, None);
        }
    }
}
impl DeferredHostOperation {
    fn new(device: Device) -> VkResult<Self> {
        let raw = unsafe {
            device
                .extension::<DeferredHostOperations>()
                .create_deferred_operation(None)?
        };
        Ok(Self { device, raw })
    }
    fn get_max_concurrency(&self) -> u32 {
        unsafe {
            self.device
                .extension::<DeferredHostOperations>()
                .get_deferred_operation_max_concurrency(self.raw)
        }
    }
    pub fn status(&self) -> Option<vk::Result> {
        match unsafe {
            (self
                .device
                .extension::<DeferredHostOperations>()
                .fp()
                .get_deferred_operation_result_khr)(self.device.handle(), self.raw)
        } {
            vk::Result::NOT_READY => None,
            result => Some(result),
        }
    }
    pub fn raw(&self) -> vk::DeferredOperationKHR {
        self.raw
    }
}

struct DHOTask {
    // Number of remaining parallelism
    concurrency: AtomicU32,
    // true if THREAD_DONE_KHR or DONE was ever returned
    done: AtomicBool,
    op: DeferredHostOperation,
}

#[derive(Resource)]
pub struct DeferredOperationTaskPool(Option<DeferredOperationTaskPoolInner>);
struct DeferredOperationTaskPoolInner {
    device: Device,
    sender: ManuallyDrop<Arc<Sender<Arc<DHOTask>>>>,
    threads: Vec<JoinHandle<()>>,
    available_parallelism: u32,
}
impl FromWorld for DeferredOperationTaskPool {
    fn from_world(world: &mut bevy::ecs::world::World) -> Self {
        let device = world.resource::<Device>().clone();
        Self::new(device)
    }
}
impl Drop for DeferredOperationTaskPoolInner {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.sender);
        }
        for i in self.threads.drain(..) {
            i.join().unwrap();
        }
    }
}

impl DeferredOperationTaskPool {
    pub fn new(device: Device) -> Self {
        if device.get_extension::<DeferredHostOperations>().is_err() {
            return Self(None);
        }
        // At most one Task may be in the channel at any given time
        let (sender, receiver) = crossbeam_channel::unbounded::<Arc<DHOTask>>();
        let sender = Arc::new(sender);
        let available_parallelism = std::thread::available_parallelism().unwrap().get() as u32;
        let threads: Vec<_> = (0..available_parallelism)
            .map(|_| {
                let sender = Arc::downgrade(&sender);
                let receiver = receiver.clone();
                let device = device.clone();
                std::thread::spawn(move || {
                    loop {
                        let task = if let Ok(task) = receiver.recv() {
                            task
                        } else {
                            // Disconnected.
                            return;
                        };
                        if task.done.load(std::sync::atomic::Ordering::Relaxed) {
                            // Other threads have signaled that this task is done
                            continue;
                        }
                        let current_concurrency = task
                            .concurrency
                            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                        if current_concurrency > 1 {
                            // The task can be handled by someone else concurrently
                            if let Some(sender) = sender.upgrade() {
                                sender.send(task.clone()).unwrap();
                            }
                        }
                        if current_concurrency == 0 {
                            continue;
                        }
                        match unsafe {
                            (device
                                .extension::<DeferredHostOperations>()
                                .fp()
                                .deferred_operation_join_khr)(
                                device.handle(), task.op.raw()
                            )
                        } {
                            vk::Result::THREAD_DONE_KHR => {
                                // A return value of VK_THREAD_DONE_KHR indicates that the deferred operation is not complete,
                                // but there is no work remaining to assign to threads. Future calls to vkDeferredOperationJoinKHR
                                // are not necessary and will simply harm performance. This situation may occur when other threads
                                // executing vkDeferredOperationJoinKHR are about to complete operation, and the implementation
                                // is unable to partition the workload any further.
                                task.done.store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                            vk::Result::THREAD_IDLE_KHR => {
                                // A return value of VK_THREAD_IDLE_KHR indicates that the deferred operation is not complete,
                                // and there is no work for the thread to do at the time of the call. This situation may occur
                                // if the operation encounters a temporary reduction in parallelism.
                                // By returning VK_THREAD_IDLE_KHR, the implementation is signaling that it expects that more
                                // opportunities for parallelism will emerge as execution progresses, and that future calls to
                                // vkDeferredOperationJoinKHR can be beneficial. In the meantime, the application can perform
                                // other work on the calling thread.
                                let current_concurrency = task
                                    .concurrency
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                if current_concurrency == 0 {
                                    if let Some(sender) = sender.upgrade() {
                                        sender.send(task.clone()).unwrap();
                                    }
                                }
                            }
                            _result => {
                                task.done.store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }
                })
            })
            .collect();
        let inner = DeferredOperationTaskPoolInner {
            device,
            sender: ManuallyDrop::new(sender),
            threads,
            available_parallelism,
        };
        Self(Some(inner))
    }
    pub fn schedule_dho<T: Send + 'static>(
        &self,
        op: impl FnOnce(vk::DeferredOperationKHR) -> (vk::Result, T) + Send + 'static,
    ) -> Task<T> {
        let mut raw_dho = vk::DeferredOperationKHR::null();
        let mut sender = None;
        let mut available_parallelism = 0;
        let dho = if let Some(inner) = &self.0 {
            let dho_task = DHOTask {
                done: AtomicBool::new(false),
                op: DeferredHostOperation::new(inner.device.clone()).unwrap(),
                concurrency: AtomicU32::new(0),
            };
            raw_dho = dho_task.op.raw();
            sender = Some(inner.sender.clone());
            available_parallelism = inner.available_parallelism;
            Some(Arc::new(dho_task))
        } else {
            None
        };
        let dho2 = dho.clone();

        let task = bevy::tasks::AsyncComputeTaskPool::get().spawn(async move {
            // We always spawn the operation on a task pool.
            // Some bad implementation will block the thread even with DeferredHostOperation extension enabled.
            let (result, item) = op(raw_dho);
            match result {
                vk::Result::SUCCESS => return Ok(item),
                vk::Result::OPERATION_DEFERRED_KHR => {
                    let dho = dho2.unwrap();
                    let concurrency = dho.op.get_max_concurrency().max(available_parallelism);
                    dho.concurrency
                        .store(concurrency, std::sync::atomic::Ordering::Relaxed);
                    sender.unwrap().send(dho.clone()).unwrap();
                    return Ok(item);
                }
                vk::Result::OPERATION_NOT_DEFERRED_KHR => {
                    return Ok(item);
                }
                other => return Err(other),
            }
        });
        Task { task, dho }
    }

    pub fn schedule<T: Send + 'static>(
        &self,
        op: impl FnOnce() -> VkResult<T> + Send + 'static,
    ) -> Task<T> {
        let task = bevy::tasks::AsyncComputeTaskPool::get().spawn(async move { op() });
        Task { task, dho: None }
    }
}

pub struct Task<T> {
    task: bevy::tasks::Task<VkResult<T>>,
    dho: Option<Arc<DHOTask>>,
}
impl<T> Task<T> {
    pub fn is_finished(&self) -> bool {
        if !self.task.is_finished() {
            return false;
        }
        if let Some(dho) = &self.dho {
            return dho.op.status().is_some();
        } else {
            return true;
        }
    }
    pub fn unwrap(self) -> VkResult<T> {
        if !self.task.is_finished() {
            return Err(vk::Result::NOT_READY);
        }
        let out = unwrap_future(self.task)?;
        if let Some(task) = self.dho {
            let status = task.op.status();
            if let Some(status) = status {
                return Err(status);
            }
        }
        Ok(out)
    }
}
