use std::{ops::Generator, pin::Pin, sync::Arc, task::Poll, marker::PhantomData};

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
    type Output;
    fn init(self: Pin<&mut Self>, prev_queue: QueueRef);
    /// Record all command buffers for the specified queue_index, up to the specified timeline index.
    /// The executor calls record with increasing `timeline` value, and wrap them in vk::SubmitInfo2.
    /// queue should be the queue of the parent node, or None if multiple parents with different queues.
    /// queue should be None on subsequent calls.
    fn record(self: Pin<&mut Self>) -> Poll<(QueueRef, Self::Output)>;

    fn execute(self);
}


pub trait QueueFutureBlockGenerator<R> = Generator<QueueRef, Return = (QueueRef, R)>;

#[pin_project]
pub struct QueueFutureBlock<R, I> {
    #[pin]
    inner: I,
    initial_queue_ref: QueueRef,
    _marker: PhantomData<fn() -> R>
}
impl<R, I: QueueFutureBlockGenerator<R>> QueueFutureBlock<R, I> {
    pub fn new(inner: I) -> Self {
        Self {
            inner,
            initial_queue_ref: QueueRef::null(),
            _marker: PhantomData
        }
    }
}
impl<R, I: QueueFutureBlockGenerator<R>> QueueFuture for QueueFutureBlock<R, I> {
    type Output = R;
    fn init(self: Pin<&mut Self>, prev_queue: QueueRef) {
        *self.project().initial_queue_ref = prev_queue;
    }
    fn record(self: Pin<&mut Self>) -> Poll<(QueueRef, R)> {
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

#[pin_project]
pub struct QueueFutureJoin<I1, I2> {
    #[pin]
    left: I1,
    #[pin]
    right: I2,
    initial_queue_ref: QueueRef,
}
/*
impl<I1: QueueFuture, I2: QueueFuture> QueueFuture for QueueFutureJoin<I1, I2> {
    fn init(self: Pin<&mut Self>, prev_queue: QueueRef) {
        let this = self.project();
        *this.initial_queue_ref = prev_queue;
    }

    fn record(self: Pin<&mut Self>) -> Poll<QueueRef> {
        todo!()
    }

    fn execute(self) {
        todo!()
    }
}
*/
