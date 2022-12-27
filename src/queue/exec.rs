use std::{ops::Generator, pin::Pin, sync::Arc, task::Poll, marker::PhantomData};

use ash::vk;
use futures_util::future::OptionFuture;
use pin_project::pin_project;

use crate::{Device, Queue, future::{GPUCommandFuture, GlobalContext}};


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

/// Contains info regarding one queue of a stage of the submission.
pub struct QueueSubmitInfo {
    command_buffers: Vec<vk::CommandBuffer>,
    current_ctx: GlobalContext
}
/// Contains info regarding one stage of the submission order.
pub struct QueueContext {
    queues: Vec<QueueSubmitInfo>,
}
impl QueueContext {
    pub fn new() -> Self {
        Self {
            queues: Vec::new()
        }
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

pub enum QueueFuturePoll<OUT> {
    Barrier,
    Semaphore,
    Ready {
        next_queue: QueueRef,
        output: OUT
    }
}
pub trait QueueFuture {
    type Output;
    fn init(self: Pin<&mut Self>, prev_queue: QueueRef);
    /// Record all command buffers for the specified queue_index, up to the specified timeline index.
    /// The executor calls record with increasing `timeline` value, and wrap them in vk::SubmitInfo2.
    /// queue should be the queue of the parent node, or None if multiple parents with different queues.
    /// queue should be None on subsequent calls.
    fn record(self: Pin<&mut Self>, ctx: &mut QueueContext) -> QueueFuturePoll<Self::Output>;

    fn execute(self);
}


/// On yield: true for a hard sync point (semaphore)
pub trait QueueFutureBlockGenerator<R> = Generator<(QueueRef, *mut QueueContext), Return = (QueueRef, R), Yield = bool>;

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
    fn record(self: Pin<&mut Self>, ctx: &mut QueueContext) -> QueueFuturePoll<R> {
        let this = self.project();
        match this.inner.resume((*this.initial_queue_ref, ctx)) {
            std::ops::GeneratorState::Yielded(is_semaphore) if is_semaphore => QueueFuturePoll::Semaphore,
            std::ops::GeneratorState::Yielded(is_semaphore) => QueueFuturePoll::Barrier,
            std::ops::GeneratorState::Complete((next_queue, output)) => QueueFuturePoll::Ready { next_queue, output },
        }
    }
    fn execute(self) {
        todo!()
    }
}

#[pin_project]
pub struct QueueFutureJoin<I1: QueueFuture, I2: QueueFuture> {
    #[pin]
    inner1: I1,
    inner1_result: QueueFuturePoll<I1::Output>,
    #[pin]
    inner2: I2,
    inner2_result: QueueFuturePoll<I2::Output>,
    results_taken: bool,
    initial_queue_ref: QueueRef,
}

impl<I1: QueueFuture, I2: QueueFuture> QueueFutureJoin<I1, I2> {
    pub fn new(inner1: I1, inner2: I2) -> Self {
        Self {
            inner1,
            inner1_result: QueueFuturePoll::Barrier,
            inner2,
            inner2_result: QueueFuturePoll::Barrier,
            results_taken: false,
            initial_queue_ref: QueueRef::null()
        }
    }
}

impl<I1: QueueFuture, I2: QueueFuture> QueueFuture for QueueFutureJoin<I1, I2> {
    type Output = (I1::Output, I2::Output);
    fn init(self: Pin<&mut Self>, prev_queue: QueueRef) {
        let this = self.project();
        *this.initial_queue_ref = prev_queue;
        this.inner1.init(prev_queue);
        this.inner2.init(prev_queue);
    }

    fn record(self: Pin<&mut Self>, ctx: &mut QueueContext) -> QueueFuturePoll<Self::Output> {
        let this = self.project();
        assert!(
            !*this.results_taken,
            "Attempted to record a QueueFutureJoin after it's finished"
        );
        match (&this.inner1_result, &this.inner2_result) {
            (QueueFuturePoll::Barrier, QueueFuturePoll::Barrier) |
            (QueueFuturePoll::Semaphore, QueueFuturePoll::Semaphore) => {
                *this.inner1_result = this.inner1.record(ctx);
                *this.inner2_result = this.inner2.record(ctx);
            },
            (QueueFuturePoll::Barrier, QueueFuturePoll::Semaphore) |
            (QueueFuturePoll::Barrier, QueueFuturePoll::Ready { .. }) |
            (QueueFuturePoll::Semaphore, QueueFuturePoll::Ready { .. }) => {
                *this.inner1_result = this.inner1.record(ctx);
            },
            (QueueFuturePoll::Semaphore, QueueFuturePoll::Barrier) |
            (QueueFuturePoll::Ready { .. }, QueueFuturePoll::Barrier) |
            (QueueFuturePoll::Ready { .. }, QueueFuturePoll::Semaphore) => {
                *this.inner2_result = this.inner2.record(ctx);
            },
            (QueueFuturePoll::Ready {..}, QueueFuturePoll::Ready { .. }) => {
                unreachable!();
            }
        }
        match (&this.inner1_result, &this.inner2_result) {
            (QueueFuturePoll::Barrier, QueueFuturePoll::Barrier) |
            (QueueFuturePoll::Barrier, QueueFuturePoll::Semaphore) |
            (QueueFuturePoll::Barrier, QueueFuturePoll::Ready { .. }) |
            (QueueFuturePoll::Semaphore, QueueFuturePoll::Barrier) |
            (QueueFuturePoll::Ready { .. }, QueueFuturePoll::Barrier)=> {
                QueueFuturePoll::Barrier
            },
            (QueueFuturePoll::Semaphore, QueueFuturePoll::Semaphore) |
            (QueueFuturePoll::Ready { .. }, QueueFuturePoll::Semaphore) |
            (QueueFuturePoll::Semaphore, QueueFuturePoll::Ready { .. }) => {
                QueueFuturePoll::Semaphore 
            },
            (QueueFuturePoll::Ready { .. }, QueueFuturePoll::Ready { .. }) => {
                let (r1_queue, r1_ret) = match std::mem::replace(this.inner1_result, QueueFuturePoll::Barrier) {
                    QueueFuturePoll::Ready { next_queue: r1_queue, output: r1_ret } => {
                        (r1_queue, r1_ret)
                    },
                    _ => unreachable!()
                };
                let (r2_queue, r2_ret) = match std::mem::replace(this.inner2_result, QueueFuturePoll::Barrier) {
                    QueueFuturePoll::Ready { next_queue: r2_queue, output: r2_ret } => {
                        (r2_queue, r2_ret)
                    },
                    _ => unreachable!()
                };
                let queue = if r1_queue == r2_queue {
                    r1_queue
                } else {
                    QueueRef::null()
                };
                *this.results_taken = true;
                return QueueFuturePoll::Ready {
                    next_queue: queue,
                    output: (r1_ret, r2_ret)
                };
            }
        }
    }

    fn execute(self) {
        todo!()
    }
}

#[pin_project]
struct RunCommandsQueueFuture<I: GPUCommandFuture> {
    #[pin]
    inner: I,
    initial_queue: QueueRef,
}

impl<I: GPUCommandFuture> QueueFuture for RunCommandsQueueFuture<I> {
    type Output = I::Output;

    fn init(self: Pin<&mut Self>, prev_queue: QueueRef) {
        let this = self.project();
        *this.initial_queue = prev_queue;
        // this.inner.init(&mut );
    }

    fn record(self: Pin<&mut Self>, ctx: &mut QueueContext) -> QueueFuturePoll<Self::Output> {
        let this = self.project();
        //this.inner.record(ctx)
        todo!()
    }

    fn execute(self) {
        todo!()
    }
}
