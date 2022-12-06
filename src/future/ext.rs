use super::{GPUCommandFuture, StageContext};
use ash::vk;
use pin_project::pin_project;
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
}

impl<T: GPUCommandFuture> GPUCommandFutureExt for T {}

#[pin_project(project = GPUCommandJoinStateProj)]
enum GPUCommandJoinState<G: GPUCommandFuture> {
    Pending(#[pin] G),
    Ready(G::Output),
    Taken,
}

#[pin_project]
pub struct GPUCommandJoin<G1, G2>
where
    G1: GPUCommandFuture,
    G2: GPUCommandFuture,
{
    #[pin]
    inner1: G1,
    inner1_result: Option<G1::Output>,
    #[pin]
    inner2: G2,
    inner2_result: Option<G2::Output>,

    results_taken: bool,
}

impl<G1, G2> GPUCommandFuture for GPUCommandJoin<G1, G2>
where
    G1: GPUCommandFuture,
    G2: GPUCommandFuture,
{
    type Output = (G1::Output, G2::Output);
    #[inline]
    fn record(self: Pin<&mut Self>, command_buffer: vk::CommandBuffer) -> Poll<Self::Output> {
        let this = self.project();
        assert!(
            !*this.results_taken,
            "Attempted to record a GPUCommandJoin after it's finished"
        );
        if this.inner1_result.is_none() {
            if let Poll::Ready(r) = this.inner1.record(command_buffer) {
                *this.inner1_result = Some(r);
            }
        }
        if this.inner2_result.is_none() {
            if let Poll::Ready(r) = this.inner2.record(command_buffer) {
                *this.inner2_result = Some(r);
            }
        }
        if this.inner1_result.is_some() && this.inner2_result.is_some() {
            let r1 = this.inner1_result.take().unwrap();
            let r2 = this.inner2_result.take().unwrap();
            *this.results_taken = true;
            Poll::Ready((r1, r2))
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

    fn init(self: Pin<&mut Self>) {
        let this = self.project();
        this.inner1.init();
        this.inner2.init();
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
    #[inline]
    fn record(self: Pin<&mut Self>, command_buffer: vk::CommandBuffer) -> Poll<Self::Output> {
        let this = self.project();
        match this.inner.record(command_buffer) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(r) => {
                let mapper = this
                    .mapper
                    .take()
                    .expect("Attempted to poll GPUCommandMap after completion");
                Poll::Ready((mapper)(r))
            }
        }
    }
    fn context(self: Pin<&mut Self>, ctx: &mut StageContext) {
        self.project().inner.context(ctx);
    }
    fn init(self: Pin<&mut Self>) {
        let this = self.project();
        this.inner.init();
    }
}
