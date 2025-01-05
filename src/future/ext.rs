use super::*;
use std::{future::Future, pin::Pin, task::Poll};

pub trait GPUFutureBlockExt: GPUFutureBlock + Sized {
    fn run_in_parallel(self) -> GPUParallelFutureBlock<Self> {
        GPUParallelFutureBlock(self)
    }
}

impl<T: GPUFutureBlock + Sized> GPUFutureBlockExt for T {}

/// Wrapper that executes everything in a GPUFutureBlock in parallel with no barriers inbetween.
pub struct GPUParallelFutureBlock<T: GPUFutureBlock>(T);
impl<T: GPUFutureBlock> Future for GPUParallelFutureBlock<T> {
    type Output =
        GPUFutureBlockReturnValue<<T as GPUFutureBlock>::Returned, <T as GPUFutureBlock>::Retained>;
    fn poll(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let mut this = unsafe { self.map_unchecked_mut(|x| &mut x.0) };
        loop {
            match <T as Future>::poll(this.as_mut(), cx) {
                Poll::Ready(result) => return Poll::Ready(result),
                Poll::Pending => continue,
            }
        }
    }
}
