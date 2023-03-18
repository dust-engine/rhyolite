use crate::Device;
use ash::{
    prelude::VkResult,
    vk::{self},
};
use crossbeam_channel::Sender;
use event_listener::Event;
use std::{
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicI32},
        Arc,
    },
    thread::JoinHandle, future::Future,
};

pub struct DeferredOperation {
    device: Arc<Device>,
    raw: vk::DeferredOperationKHR,
}
impl Drop for DeferredOperation {
    fn drop(&mut self) {
        unsafe {
            self.device
                .deferred_host_operation_loader()
                .destroy_deferred_operation(self.raw, None);
        }
    }
}
impl DeferredOperation {
    pub fn new(device: Arc<Device>) -> VkResult<Self> {
        let raw = unsafe {
            device
                .deferred_host_operation_loader()
                .create_deferred_operation(None)?
        };
        Ok(Self { device, raw })
    }
    pub fn get_max_concurrency(&self) -> u32 {
        unsafe {
            self.device
                .deferred_host_operation_loader()
                .get_deferred_operation_max_concurrency(self.raw)
        }
    }
    pub fn status(&self) -> Option<vk::Result> {
        match unsafe {
            (self
                .device
                .deferred_host_operation_loader()
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

pub struct Task {
    op: DeferredOperation,
    concurrency: AtomicU32,
    event: Event,
}

pub struct DeferredOperationTaskPool {
    sender: Sender<Arc<Task>>,
    terminate: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
    available_parallelism: u32,
}
impl Drop for DeferredOperationTaskPool {
    fn drop(&mut self) {
        self.terminate
            .store(true, std::sync::atomic::Ordering::Relaxed);
        for i in self.threads.drain(..) {
            i.join().unwrap();
        }
    }
}

impl DeferredOperationTaskPool {
    pub fn new(device: Arc<Device>) -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded::<Arc<Task>>();
        let terminate = Arc::new(AtomicBool::new(false));
        let available_parallelism = std::thread::available_parallelism().unwrap().get() as u32;
        let threads: Vec<_> = (0..available_parallelism)
            .map(|_| {
                let sender = sender.clone();
                let receiver = receiver.clone();
                let device = device.clone();
                let terminate = terminate.clone();
                std::thread::spawn(move || {
                    while !terminate.load(std::sync::atomic::Ordering::Relaxed) {
                        let task = if let Ok(task) = receiver.recv() {
                            task
                        } else {
                            // Disconnected.
                            return;
                        };
                        let current_concurrency = task
                            .concurrency
                            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                        if current_concurrency > 1 {
                            // The task can be handled by someone else concurrently
                            sender.send(task.clone()).unwrap();
                        }
                        if current_concurrency == 0 {
                            continue;
                        }
                        match unsafe {
                            (device
                                .deferred_host_operation_loader()
                                .fp()
                                .deferred_operation_join_khr)(
                                device.handle(), task.op.raw
                            )
                        } {
                            vk::Result::THREAD_DONE_KHR => {
                                // A return value of VK_THREAD_DONE_KHR indicates that the deferred operation is not complete,
                                // but there is no work remaining to assign to threads. Future calls to vkDeferredOperationJoinKHR
                                // are not necessary and will simply harm performance. This situation may occur when other threads
                                // executing vkDeferredOperationJoinKHR are about to complete operation, and the implementation
                                // is unable to partition the workload any further.
                                task.concurrency
                                    .store(0, std::sync::atomic::Ordering::Relaxed);
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
                                    sender.send(task.clone()).unwrap();
                                }
                            }
                            result => {
                                task.concurrency
                                    .store(0, std::sync::atomic::Ordering::Relaxed);
                                task.event.notify_relaxed(1);
                            },
                        }
                    }
                })
            })
            .collect();
        Self {
            sender,
            terminate,
            threads,
            available_parallelism
        }
    }
    pub fn schedule_deferred_operation(&self, op: DeferredOperation) -> impl Future<Output = vk::Result> {
        let concurrency = op.get_max_concurrency().max(self.available_parallelism);
        let event = Event::new();
        let task = Task {
            event,
            op,
            concurrency: AtomicU32::new(concurrency),
        };
        let task = Arc::new(task);
        let listener = task.event.listen();
        self.sender.send(task.clone()).unwrap();
        async move {
            listener.await;
            task.op.status().unwrap()
        }
    }
}
