use super::{GPUCommandFuture, GPUCommandFutureContext};
use ash::vk;
use pin_project::pin_project;
use std::marker::PhantomData;
use std::ops::{Generator, GeneratorState};
use std::pin::Pin;
use std::task::Poll;

pub trait GPUCommandGenerator<R> = Generator<vk::CommandBuffer, Yield = GeneratorState<GPUCommandFutureContext, R>, Return = ()>;

#[pin_project]
pub struct GPUCommandBlock<R, G> {
    #[pin]
    inner: G,
    next_ctx: Option<GPUCommandFutureContext>,
    _marker: std::marker::PhantomData<R>,
}
impl<R, G: GPUCommandGenerator<R>> GPUCommandBlock<R, G>{
    pub fn new(inner: G) -> Self {
        Self {
            inner,
            next_ctx: None,
            _marker: PhantomData
        }
    }
}
impl<R, G: GPUCommandGenerator<R>> GPUCommandFuture for GPUCommandBlock<R, G> {
    type Output = R;
    fn record(self: Pin<&mut Self>, command_buffer: vk::CommandBuffer) -> Poll<R> {
        let this = self.project();
        match this.inner.resume(command_buffer) {
            GeneratorState::Yielded(ctx) => {
                match ctx {
                    GeneratorState::Yielded(ctx) => {
                        *this.next_ctx = Some(ctx);
                        Poll::Pending
                    },
                    GeneratorState::Complete(r) => {
                        Poll::Ready(r)
                    }
                }
            }
            // Block generators should never be driven to completion. Instead, they should be dropped before completion.
            GeneratorState::Complete(_) => unreachable!(),
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
