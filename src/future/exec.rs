use core::task::ContextBuilder;
use std::{cell::Cell, future::Future, ops::{Deref, DerefMut}, pin::Pin, sync::{atomic::AtomicU64, Arc, Mutex, MutexGuard}, task::Poll};

use ash::{prelude::VkResult, vk};

use crate::{command::CommandEncoder, semaphore::TimelineSemaphore, Device};

use super::{GPUFutureBlock, GPUFutureBlockReturnValue, GPUFutureContext};



fn gpu_future_poll<T: Future>(gpu_future: Pin<&mut T>, ctx: &mut GPUFutureContext) -> Poll<T::Output> {
    use std::task::{Waker, RawWaker, RawWakerVTable};

    fn null_waker_clone_fn(_ptr: *const ()) -> RawWaker {
        panic!("GPU Futures cannot be executed from regular async executors");
    }
    fn null_waker_fn(_ptr: *const ()) {
        panic!("GPU Futures cannot be executed from regular async executors");
    }

    const NULL_WAKER_VTABLE: &'static RawWakerVTable = &RawWakerVTable::new(
        null_waker_clone_fn,
        null_waker_fn,
        null_waker_fn,
        null_waker_fn,
    );
    const NULL_WAKER: &'static Waker = unsafe {
        &Waker::new(std::ptr::null(), NULL_WAKER_VTABLE)
    };
    let mut ctx = ContextBuilder::from_waker(NULL_WAKER).ext(ctx).build();
    gpu_future.poll(&mut ctx)
}



pub struct GPUFutureSubmissionStatus<T: GPUFutureBlock> {
    return_value: <T as GPUFutureBlock>::Output,
    retained_values: <T as GPUFutureBlock>::Retained,
    timeline_semaphore: Arc<TimelineSemaphore>,
    wait_value: u64
}

impl<'a> CommandEncoder<'a> {
    /// The recorded futures will be executed serially
    pub fn record<T: GPUFutureBlock>(&mut self, future: T) -> GPUFutureSubmissionStatus<T> {
        let mut future = std::pin::pin!(future);
        let GPUFutureBlockReturnValue { output, retained_values } = loop {
            match gpu_future_poll(future.as_mut(), &mut self.command_buffer.future_ctx) {
                Poll::Ready(output) => break output,
                Poll::Pending => {
                    // TODO insert pipeline barrier as needed
                },
            }
        };
        GPUFutureSubmissionStatus {
            return_value: output,
            retained_values,
            timeline_semaphore: self.command_buffer.timeline_semaphore.clone(),
            wait_value: self.command_buffer.wait_value,
        }
    }
}
