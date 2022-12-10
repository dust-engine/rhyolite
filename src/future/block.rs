use super::{GPUCommandFuture, GlobalContext, StageContext};
use pin_project::pin_project;
use std::marker::PhantomData;
use std::ops::{Generator, GeneratorState};
use std::pin::Pin;
use std::task::Poll;

//S Generator takes a raw pointer as the argument. https://github.com/rust-lang/rust/issues/68923
pub trait GPUCommandGenerator<R, State> =
    for<'a> Generator<*mut GlobalContext, Yield = StageContext, Return = (R, State)>;

#[pin_project]
pub struct GPUCommandBlock<R, State, G> {
    #[pin]
    inner: G,
    next_ctx: Option<StageContext>,
    _marker: std::marker::PhantomData<(R, State)>,
}
impl<R, State, G: GPUCommandGenerator<R, State>> GPUCommandBlock<R, State, G> {
    pub fn new(inner: G) -> Self {
        Self {
            inner,
            next_ctx: None,
            _marker: PhantomData,
        }
    }
}
impl<R, State, G: GPUCommandGenerator<R, State>> GPUCommandFuture for GPUCommandBlock<R, State, G> {
    type Output = R;
    type RetainedState = State;
    fn record(
        self: Pin<&mut Self>,
        ctx: &mut GlobalContext,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        match this.inner.resume(ctx) {
            GeneratorState::Yielded(ctx) => {
                *this.next_ctx = Some(ctx);
                Poll::Pending
            }
            // Block generators should never be driven to completion. Instead, they should be dropped before completion.
            GeneratorState::Complete((ret, state)) => {
                *this.next_ctx = None;
                Poll::Ready((ret, state))
            }
        }
    }
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        let next_ctx = self
            .project()
            .next_ctx
            .take()
            .expect("Attempted to take the context multiple times");
        ctx.merge(next_ctx);
    }
    fn init(mut self: Pin<&mut Self>, ctx: &mut GlobalContext) {
        // Reach the first yield point to get the context of the first awaited future.
        assert!(self.next_ctx.is_none());
        if let Poll::Pending = self.as_mut().record(ctx) {
        } else {
            unreachable!()
        }
    }
}
