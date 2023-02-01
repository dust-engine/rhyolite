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
            this: unsafe { this.get_unchecked_mut() as *mut T as *mut () },
            fetch: |ptr, stage| unsafe {
                let ptr = std::pin::Pin::new_unchecked(&mut *(ptr as *mut T));
                T::context(ptr, stage)
            },
        }
    }
    pub fn call(&mut self, ctx: &mut StageContext) {
        (self.fetch)(self.this, ctx);
    }
}
//S Generator takes a raw pointer as the argument. https://github.com/rust-lang/rust/issues/68923
// Should be:
// pub trait GPUCommandGenerator<'retain, R, State, Recycle: Default> = for<'a, 'b> Generator<
// (&'a mut CommandBufferRecordContext<'b>, &'a mut Recycle),
// Yield = GPUCommandGeneratorContextFetchPtr,
// Return = (R, State, PhantomData<&'retain ()>),
// >;
pub trait GPUCommandGenerator<'retain, R, State, Recycle: Default> = Generator<
    (*mut (), *mut Recycle),
    Yield = GPUCommandGeneratorContextFetchPtr,
    Return = (R, State, PhantomData<&'retain ()>),
>;

#[pin_project]
pub struct GPUCommandBlock<R, State, Recycle: Default, G> {
    #[pin]
    inner: G,
    next_ctx: Option<GPUCommandGeneratorContextFetchPtr>,
    _marker: std::marker::PhantomData<fn(*mut Recycle) -> (R, State)>,
}
impl<'retain, R, State, Recycle: Default, G: GPUCommandGenerator<'retain, R, State, Recycle>>
    GPUCommandBlock<R, State, Recycle, G>
{
    pub fn new(inner: G) -> Self {
        Self {
            inner,
            next_ctx: None,
            _marker: PhantomData,
        }
    }
}
impl<'retain, R, State, Recycle: Default, G: GPUCommandGenerator<'retain, R, State, Recycle>>
    GPUCommandFuture for GPUCommandBlock<R, State, Recycle, G>
{
    type Output = R;
    type RetainedState = State;
    type RecycledState = Recycle;
    fn record<'a, 'b: 'a>(
        self: Pin<&mut Self>,
        ctx: &'a mut CommandBufferRecordContext<'b>,
        recycled_state: &mut Recycle,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        match this
            .inner
            .resume((ctx as *mut _ as *mut (), recycled_state))
        {
            GeneratorState::Yielded(ctx) => {
                *this.next_ctx = Some(ctx);
                Poll::Pending
            }
            GeneratorState::Complete((ret, state, _)) => {
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
    fn init<'a, 'b: 'a>(
        self: Pin<&mut Self>,
        ctx: &'a mut CommandBufferRecordContext<'b>,
        recycled_state: &mut Recycle,
    ) {
        // Reach the first yield point to get the context of the first awaited future.
        assert!(self.next_ctx.is_none());
        let this = self.project();
        match this
            .inner
            .resume((ctx as *mut _ as *mut (), recycled_state))
        {
            GeneratorState::Yielded(ctx) => {
                *this.next_ctx = Some(ctx);
            }
            GeneratorState::Complete(_) => unreachable!(),
        }
    }
}
