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
    next_ctx: GPUCommandFutureContext,
}
impl<G: GPUCommandGenerator> GPUCommandBlock<G> {
    pub fn new(inner: G) -> Self {
        Self {
            inner,
            next_ctx: GPUCommandFutureContext::default(),
        }
    }
}
impl<G: GPUCommandGenerator> GPUCommandFuture for GPUCommandBlock<G> {
    type Output = G::Return;
    fn record(self: Pin<&mut Self>, command_buffer: vk::CommandBuffer) -> Poll<G::Return> {
        let this = self.project();
        match this.inner.resume(command_buffer) {
            GeneratorState::Yielded(ctx) => {
                *this.next_ctx = ctx;
                Poll::Pending
            }
            GeneratorState::Complete(r) => Poll::Ready(r),
        }
    }
    fn context(&self) -> GPUCommandFutureContext {
        self.next_ctx.clone()
    }
    fn init(mut self: Pin<&mut Self>) {
        if let Poll::Pending = self.as_mut().record(vk::CommandBuffer::null()) {
        } else {
            unreachable!()
        }
    }
}
