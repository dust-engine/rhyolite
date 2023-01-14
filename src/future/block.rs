use super::{CommandBufferRecordContext, GPUCommandFuture, StageContext};
use pin_project::pin_project;
use std::marker::PhantomData;
use std::ops::{Generator, GeneratorState};
use std::pin::Pin;
use std::task::Poll;

pub struct GPUCommandGeneratorContextFetchPtr {
    this: *mut (),
    fetch: fn(*mut (), ctx: &mut StageContext),
}
impl GPUCommandGeneratorContextFetchPtr {
    pub fn new<T: GPUCommandFuture>(this: Pin<&mut T>) -> Self {
        Self {
            this: unsafe {
                this.get_unchecked_mut() as *mut T as *mut ()
            },
            fetch: |ptr, stage| unsafe {
                let ptr = std::pin::Pin::new_unchecked(&mut *(ptr as *mut T));
                T::context(ptr, stage)
            }
        }
    }
    pub fn call(&mut self, ctx: &mut StageContext) {
        (self.fetch)(self.this, ctx);
    }
}
//S Generator takes a raw pointer as the argument. https://github.com/rust-lang/rust/issues/68923
pub trait GPUCommandGenerator<R, State> = Generator<
    *mut CommandBufferRecordContext,
    Yield = GPUCommandGeneratorContextFetchPtr,
    Return = (R, State),
>;

#[pin_project]
pub struct GPUCommandBlock<R, State, G> {
    #[pin]
    inner: G,
    next_ctx: Option<GPUCommandGeneratorContextFetchPtr>,
    _marker: std::marker::PhantomData<fn() -> (R, State)>,
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
        ctx: &mut CommandBufferRecordContext,
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
            .as_mut()
            .expect("Calling context without calling init");
        next_ctx.call(ctx);
    }
    fn init(mut self: Pin<&mut Self>, ctx: &mut CommandBufferRecordContext) {
        // Reach the first yield point to get the context of the first awaited future.
        assert!(self.next_ctx.is_none());
        if let Poll::Pending = self.as_mut().record(ctx) {
        } else {
            unreachable!()
        }
    }
}
