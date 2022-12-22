use std::{ops::Generator, pin::Pin, sync::Arc, task::Poll};

use ash::vk;
use futures_util::future::OptionFuture;
use pin_project::pin_project;

use crate::{Device, Queue};


#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct QueueRef(pub usize);
impl QueueRef {
    pub fn null() -> Self {
        QueueRef(usize::MAX)
    }
    pub fn is_null(&self) -> bool {
        self.0 == usize::MAX
    }
}

/// Execute a future on multiple queues.
/// Treats command! blocks as leafs.
pub struct CombinedQueueExecutor {
    device: Arc<Device>,
    queues: Vec<vk::Queue>,
}

impl CombinedQueueExecutor {
    pub fn run(&mut self, queue: QueueRef, future: impl QueueFuture) {

    }
}

pub trait QueueFuture {
    fn init(self: Pin<&mut Self>, prev_queue: QueueRef);
    /// Record all command buffers for the specified queue_index, up to the specified timeline index.
    /// The executor calls record with increasing `timeline` value, and wrap them in vk::SubmitInfo2.
    /// queue should be the queue of the parent node, or None if multiple parents with different queues.
    /// queue should be None on subsequent calls.
    fn record(self: Pin<&mut Self>) -> Poll<QueueRef>;

    fn execute(self);
}


pub trait QueueFutureBlockGenerator = Generator<QueueRef, Return = QueueRef>;

#[pin_project]
pub struct QueueFutureBlock<I> {
    #[pin]
    inner: I,
    initial_queue_ref: QueueRef,
}
impl<I: QueueFutureBlockGenerator> QueueFutureBlock<I> {
    pub fn new(inner: I) -> Self {
        Self {
            inner,
            initial_queue_ref: QueueRef::null()
        }
    }
}
impl<I: QueueFutureBlockGenerator> QueueFuture for QueueFutureBlock<I> {
    fn init(self: Pin<&mut Self>, prev_queue: QueueRef) {
        *self.project().initial_queue_ref = prev_queue;
    }
    fn record(self: Pin<&mut Self>) -> Poll<QueueRef> {
        let this = self.project();
        match this.inner.resume(*this.initial_queue_ref) {
            std::ops::GeneratorState::Yielded(_) => Poll::Pending,
            std::ops::GeneratorState::Complete(final_queue) => Poll::Ready(final_queue),
        }
    }
    fn execute(self) {
        todo!()
    }
}
// when do we know for sure that we can merge two