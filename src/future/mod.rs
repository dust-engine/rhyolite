use ash::vk;
use std::marker::PhantomPinned;
use std::ops::GeneratorState;
use std::pin::Pin;
use std::task::Poll;
use std::{ops::Generator, sync::Arc};

use crate::Device;

mod block;
mod exec;
mod ext;
pub use block::*;
pub use exec::*;
pub use ext::*;

pub trait GPUCommandFuture {
    type Output;

    /// Attempt to record as many commands as possible into the provided
    /// command_buffer until a pipeline barrier is needed.
    ///
    /// Commands recorded inbetween two pipeline barriers are considered
    /// to be in the same "stage". These commands do not have any dependencies
    /// between each other, and they should run independently of each other.
    ///
    /// # Return value
    ///
    /// This function returns:
    ///
    /// - [`Poll::Pending`] if it's possible to record more commands
    /// - [`Poll::Ready(val)`] with the return value `val` of this future,
    ///   if no more commands can be recorded.
    ///
    /// Once a future has finished, clients should not call `record` on it again.
    ///
    /// # Runtime characteristics
    ///
    /// Futures alone are *inert*; they must be `record`ed into a command
    /// buffer and submitted to a queue in order to do work on the GPU.
    fn record(self: Pin<&mut Self>, command_buffer: vk::CommandBuffer) -> Poll<Self::Output>;

    /// Returns the context for the operations recorded into the command buffer
    /// next time `record` was called.
    fn context(&self) -> GPUCommandFutureContext;

    /// Initialize the pinned future.
    /// This method is mostly a hook for `GPUCommandFutureBlock` to initialize
    /// the context in a pinned future.
    ///
    /// For executors, this method should be called once, and as soon as the future was pinnned.
    /// For implementations of `GPUCommandFuture`, this method can be ignored in most cases.
    /// For combinators, this method should be called recursively for all inner futures.
    fn init(self: Pin<&mut Self>) {}
}

#[test]
fn test() {
    let block1 = async_ash_macro::commands! {
        let fut1 = CopyBufferFuture{ str: "prev1"};
        fut1.await;
        let fut1 = CopyBufferFuture{ str: "prev2"};
        fut1.await;
        1_u32
    };

    let block2 = async_ash_macro::commands! {
        let fut1 = CopyBufferFuture{ str: "A"};
        fut1.await;
        let fut1 = CopyBufferFuture{ str: "B"};
        fut1.await;
        2_u64
    };
    let block3 = CopyBufferFuture { str: "special" };
    let block = async_ash_macro::commands! {
        let (a, b, c) = async_ash_macro::join!(block1, block2, block3).await;
        println!("a: {:?}, b: {:?}, c: {:?}", a, b, c);
    };

    let mut block = std::pin::pin!(block);
    block.as_mut().init();
    for i in 0..4 {
        match block.as_mut().record(vk::CommandBuffer::null()) {
            Poll::Ready(()) => {
                println!("Ready");
            }
            Poll::Pending => {
                println!("Pending");
            }
        }
    }
}

struct CopyBufferFuture {
    str: &'static str,
}
impl GPUCommandFuture for CopyBufferFuture {
    type Output = ();
    fn record(self: Pin<&mut Self>, command_buffer: vk::CommandBuffer) -> Poll<Self::Output> {
        println!("{}", self.str);
        Poll::Ready(())
    }
    fn context(&self) -> GPUCommandFutureContext {
        let mut ctx = GPUCommandFutureContext::default();
        ctx.write(
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
        ctx.read(
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
        ctx
    }
}
