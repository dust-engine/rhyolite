use super::{GPUCommandFuture, GPUCommandFutureContext};
use ash::vk;
use pin_project::pin_project;
use std::ops::{Generator, GeneratorState};
use std::pin::Pin;
use std::task::Poll;

pub trait GPUCommandGenerator = Generator<vk::CommandBuffer, Yield = GPUCommandFutureContext>;

#[pin_project]
pub struct GPUCommandBlock<G: GPUCommandGenerator> {
    #[pin]
    inner: G,
    next_ctx: Option<GPUCommandFutureContext>,
}
impl<G: GPUCommandGenerator> GPUCommandBlock<G> {
    pub fn new(inner: G) -> Self {
        Self {
            inner,
            next_ctx: None,
        }
    }
}
impl<G: GPUCommandGenerator> GPUCommandFuture for GPUCommandBlock<G> {
    type Output = G::Return;
    fn record(self: Pin<&mut Self>, command_buffer: vk::CommandBuffer) -> Poll<G::Return> {
        let this = self.project();
        match this.inner.resume(command_buffer) {
            GeneratorState::Yielded(ctx) => {
                *this.next_ctx = Some(ctx);
                Poll::Pending
            }
            GeneratorState::Complete(r) => Poll::Ready(r),
        }
    }
    fn context(self: Pin<&mut Self>, ctx: &mut GPUCommandFutureContext) {
        let next_ctx = self.project().next_ctx.take().expect("Attempted to take the context multiple times");
        ctx.merge(next_ctx);
    }
    fn init(mut self: Pin<&mut Self>) {
        // Reach the first yield point to get the context of the first awaited future.
        if let Poll::Pending = self.as_mut().record(vk::CommandBuffer::null()) {
        } else {
            unreachable!()
        }
    }
}
