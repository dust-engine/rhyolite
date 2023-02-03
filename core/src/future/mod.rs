use std::pin::Pin;
use std::task::Poll;

mod block;
mod exec;
mod ext;
mod state;
pub use block::*;
pub use exec::*;
pub use ext::*;
pub use state::*;

pub trait GPUCommandFuture {
    type Output;

    /// Objects with lifetimes that need to be extended until the future was executed on the GPU.
    type RetainedState;

    /// Optional object to be passed in at record time that collects reused states.
    type RecycledState: Default;

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
    fn record<'a, 'b: 'a>(
        self: Pin<&mut Self>,
        ctx: &'a mut CommandBufferRecordContext<'b>,
        recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)>;

    /// Returns the context for the operations recorded into the command buffer
    /// next time `record` was called.
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext);

    /// Initialize the pinned future.
    /// This method is mostly a hook for `GPUCommandFutureBlock` to move forward to its first
    /// yield point. `GPUCommandFutureBlock` would then yields the function pointer to its
    /// first future to be awaited, allowing us to call the `context` method to retrieve the
    /// context.
    /// 
    /// Returns a boolean indicating if this future should be run. If the implementation returns
    /// false, the entire future will be skipped, and no further calls to `record` or `context`
    /// will be made.
    ///
    /// For executors, this method should be called once, and as soon as the future was pinnned.
    /// For implementations of `GPUCommandFuture`, this method can be ignored in most cases.
    /// For combinators, this method should be called recursively for all inner futures.
    fn init<'a, 'b: 'a>(
        self: Pin<&mut Self>,
        _ctx: &'a mut CommandBufferRecordContext<'b>,
        _recycled_state: &mut Self::RecycledState,
    ) -> Option<(Self::Output, Self::RetainedState)> {
        None
    }
}
