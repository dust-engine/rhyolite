use ash::vk;
use std::pin::Pin;
use std::task::Poll;

mod block;
mod exec;
mod ext;
pub use block::*;
pub use exec::*;
pub use ext::*;

pub use async_ash_macro::{commands, join};

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
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext);

    /// Initialize the pinned future.
    /// This method is mostly a hook for `GPUCommandFutureBlock` to initialize
    /// the context in a pinned future.
    ///
    /// For executors, this method should be called once, and as soon as the future was pinnned.
    /// For implementations of `GPUCommandFuture`, this method can be ignored in most cases.
    /// For combinators, this method should be called recursively for all inner futures.
    fn init(self: Pin<&mut Self>) {}
}
