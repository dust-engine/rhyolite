use crate::{RunCommandsQueueFuture, QueueRef};

use super::{CommandBufferRecordContext, GPUCommandFuture, StageContext};
use pin_project::pin_project;
use std::cell::{Cell, RefCell};
use std::pin::Pin;
use std::task::Poll;

pub trait GPUCommandFutureExt: GPUCommandFuture + Sized {
    fn join<G: GPUCommandFuture>(self, other: G) -> GPUCommandJoin<Self, G> {
        GPUCommandJoin {
            inner1: self,
            inner1_result: None,
            inner2: other,
            inner2_result: None,
            results_taken: false,
        }
    }
    fn map<R, F: FnOnce(Self::Output) -> R>(self, mapper: F) -> GPUCommandMap<Self, F> {
        GPUCommandMap {
            inner: self,
            mapper: Some(mapper),
        }
    }
    fn schedule_in_queue(self, queue: QueueRef) -> RunCommandsQueueFuture<Self> {
        RunCommandsQueueFuture::new(self, queue)
    }
    fn schedule(self) -> RunCommandsQueueFuture<Self> {
        RunCommandsQueueFuture::new(self, QueueRef::null())
    }
}

impl<T: GPUCommandFuture> GPUCommandFutureExt for T {}

#[pin_project]
pub struct GPUCommandJoin<G1, G2>
where
    G1: GPUCommandFuture,
    G2: GPUCommandFuture,
{
    #[pin]
    inner1: G1,
    inner1_result: Option<(G1::Output, G1::RetainedState)>,
    #[pin]
    inner2: G2,
    inner2_result: Option<(G2::Output, G2::RetainedState)>,

    results_taken: bool,
}

impl<G1, G2> GPUCommandFuture for GPUCommandJoin<G1, G2>
where
    G1: GPUCommandFuture,
    G2: GPUCommandFuture,
{
    type Output = (G1::Output, G2::Output);
    type RetainedState = (G1::RetainedState, G2::RetainedState);
    type RecycledState = (G1::RecycledState, G2::RecycledState);
    #[inline]
    fn record<'a, 'b: 'a>(
        self: Pin<&mut Self>,
        command_buffer: &'a mut CommandBufferRecordContext<'b>,
        recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        assert!(
            !*this.results_taken,
            "Attempted to record a GPUCommandJoin after it's finished"
        );
        let (recycle1, recycle2) = recycled_state;
        if this.inner1_result.is_none() {
            if let Poll::Ready(r) = this.inner1.record(command_buffer, recycle1) {
                *this.inner1_result = Some(r);
            }
        }
        if this.inner2_result.is_none() {
            if let Poll::Ready(r) = this.inner2.record(command_buffer, recycle2) {
                *this.inner2_result = Some(r);
            }
        }
        if this.inner1_result.is_some() && this.inner2_result.is_some() {
            let (r1_ret, r1_retained_state) = this.inner1_result.take().unwrap();
            let (r2_ret, r2_retained_state) = this.inner2_result.take().unwrap();
            *this.results_taken = true;
            Poll::Ready(((r1_ret, r2_ret), (r1_retained_state, r2_retained_state)))
        } else {
            Poll::Pending
        }
    }

    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        let this = self.project();
        assert!(
            !*this.results_taken,
            "Attempted to take the context of a GPUCommandJoin after it's finished"
        );
        if this.inner1_result.is_none() {
            this.inner1.context(ctx);
        }
        if this.inner2_result.is_none() {
            this.inner2.context(ctx);
        }
    }

    fn init<'a, 'b: 'a>(self: Pin<&mut Self>, ctx: &'a mut CommandBufferRecordContext<'b>,
        recycled_state: &mut Self::RecycledState,) {
        let this = self.project();
        this.inner1.init(ctx, &mut recycled_state.0);
        this.inner2.init(ctx, &mut recycled_state.1);
    }
}

#[pin_project]
pub struct GPUCommandMap<G, F> {
    mapper: Option<F>,
    #[pin]
    inner: G,
}

impl<G, R, F> GPUCommandFuture for GPUCommandMap<G, F>
where
    G: GPUCommandFuture,
    F: FnOnce(G::Output) -> R,
{
    type Output = R;
    type RetainedState = G::RetainedState;
    type RecycledState = G::RecycledState;
    #[inline]
    fn record<'a, 'b: 'a>(
        self: Pin<&mut Self>,
        ctx: &'a mut CommandBufferRecordContext<'b>,
        recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        match this.inner.record(ctx, recycled_state) {
            Poll::Pending => Poll::Pending,
            Poll::Ready((r, retained_state)) => {
                let mapper = this
                    .mapper
                    .take()
                    .expect("Attempted to poll GPUCommandMap after completion");
                Poll::Ready(((mapper)(r), retained_state))
            }
        }
    }
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        self.project().inner.context(ctx);
    }
    fn init<'a, 'b: 'a>(self: Pin<&mut Self>, ctx: &'a mut CommandBufferRecordContext<'b>,
        recycled_state: &mut Self::RecycledState) {
        let this = self.project();
        this.inner.init(ctx, recycled_state);
    }
}

#[pin_project(project = GPUCommandForkedStateInnerProj)]
pub enum GPUCommandForkedStateInner<G: GPUCommandFuture> {
    Some(#[pin] G),
    Resolved(G::Output),
}
impl<G: GPUCommandFuture> GPUCommandForkedStateInner<G> {
    pub fn unwrap_pinned(self: Pin<&mut Self>) -> Pin<&mut G> {
        use GPUCommandForkedStateInnerProj::*;
        match self.project() {
            Some(g) => g,
            Resolved(_) => panic!(),
        }
    }
}

// The shared structure between all branches
pub struct GPUCommandForkedInner<'a, G: GPUCommandFuture, const N: usize> {
    inner: Pin<&'a mut GPUCommandForkedStateInner<G>>,
    last_stage: u32,
    ready: [bool; N],
}
impl<'a, G: GPUCommandFuture, const N: usize> GPUCommandForkedInner<'a, G, N> {
    pub fn new(inner: Pin<&'a mut GPUCommandForkedStateInner<G>>) -> RefCell<Self> {
        RefCell::new(Self {
            inner,
            last_stage: 0,
            ready: [false; N],
        })
    }
}

/// The structure specific to a single branch
#[pin_project]
pub struct GPUCommandForked<'a, 'r, G: GPUCommandFuture, const N: usize> {
    inner: &'r RefCell<GPUCommandForkedInner<'a, G, N>>,
    id: usize,
}
impl<'a, 'r, G: GPUCommandFuture, const N: usize> GPUCommandForked<'a, 'r, G, N> {
    pub fn new(inner: &'r RefCell<GPUCommandForkedInner<'a, G, N>>, id: usize) -> Self {
        Self { inner, id }
    }
}

impl<'a, 'r, G, const N: usize> GPUCommandFuture for GPUCommandForked<'a, 'r, G, N>
where
    G: GPUCommandFuture,
    G::Output: Clone,
{
    type Output = G::Output;
    type RetainedState = Option<G::RetainedState>;
    type RecycledState = G::RecycledState;
    #[inline]
    fn record<'fa, 'fb: 'fa>(
        self: Pin<&mut Self>,
        ctx: &'fa mut CommandBufferRecordContext<'fb>,
        recycled_state: &mut Self::RecycledState,
    ) -> Poll<(Self::Output, Self::RetainedState)> {
        let mut this = &mut *self.project().inner.borrow_mut();
        if !this.ready.iter().all(|a| *a) {
            // no op.
            return Poll::Pending;
        }
        match this.inner.as_mut().project() {
            GPUCommandForkedStateInnerProj::Resolved(result) => Poll::Ready((result.clone(), None)),
            GPUCommandForkedStateInnerProj::Some(inner) => {
                if this.last_stage < ctx.current_stage_index() {
                    // do the work
                    this.last_stage = ctx.current_stage_index();
                    match inner.record(ctx, recycled_state) {
                        Poll::Pending => Poll::Pending,
                        Poll::Ready((result, retained)) => {
                            this.inner
                                .set(GPUCommandForkedStateInner::Resolved(result.clone()));
                            Poll::Ready((result, Some(retained)))
                        }
                    }
                } else {
                    Poll::Pending
                }
            }
        }
    }
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        let id = self.id;
        let mut this = self.project().inner.borrow_mut();
        this.ready[id] = true;
        if !this.ready.iter().all(|a| *a) {
            // no op. only the one that finishes the latest will actually record and generate dependencies.
            return;
        }
        this.inner.as_mut().unwrap_pinned().context(ctx);
    }
    fn init<'fa, 'fb: 'fa>(self: Pin<&mut Self>, ctx: &'fa mut CommandBufferRecordContext<'fb>,
        recycled_state: &mut Self::RecycledState,) {
        // Noop. The inner command will be initialized when fork was called on it.
    }
}
