use super::{GPUCommandFuture, GPUCommandFutureContext};
use ash::vk;
use pin_project::pin_project;
use std::pin::Pin;
use std::task::Poll;

pub trait GPUCommandFutureExt: GPUCommandFuture + Sized {
    fn join<G: GPUCommandFuture>(self, other: G) -> GPUCommandJoin<Self, G> {
        GPUCommandJoin {
            inner1: GPUCommandJoinState::Pending(self),
            inner2: GPUCommandJoinState::Pending(other),
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
    inner1: GPUCommandJoinState<G1>,
    #[pin]
    inner2: GPUCommandJoinState<G2>,
}

impl<G1, G2> GPUCommandFuture for GPUCommandJoin<G1, G2>
where
    G1: GPUCommandFuture,
    G2: GPUCommandFuture,
{
    type Output = (G1::Output, G2::Output);
    #[inline]
    fn record(self: Pin<&mut Self>, command_buffer: vk::CommandBuffer) -> Poll<Self::Output> {
        let mut this = self.project();
        if let GPUCommandJoinStateProj::Pending(g) = this.inner1.as_mut().project() {
            if let Poll::Ready(r) = g.record(command_buffer) {
                unsafe {
                    // Safety: It's ok to drop the future here because it's already completed.
                    *this.inner1.as_mut().get_unchecked_mut() = GPUCommandJoinState::Ready(r);
                }
            }
        }
        if let GPUCommandJoinStateProj::Pending(g) = this.inner2.as_mut().project() {
            if let Poll::Ready(r) = g.record(command_buffer) {
                unsafe {
                    // Safety: It's ok to drop the future here because it's already completed.
                    *this.inner2.as_mut().get_unchecked_mut() = GPUCommandJoinState::Ready(r);
                }
            }
        }
        match (
            this.inner1.as_mut().project(),
            this.inner2.as_mut().project(),
        ) {
            (GPUCommandJoinStateProj::Ready(_), GPUCommandJoinStateProj::Ready(_)) => {
                // Take ownership
                let rs = unsafe {
                    // Safety: It's ok to drop them now, because they're completed.
                    (
                        std::mem::replace(
                            this.inner1.get_unchecked_mut(),
                            GPUCommandJoinState::Taken,
                        ),
                        std::mem::replace(
                            this.inner2.get_unchecked_mut(),
                            GPUCommandJoinState::Taken,
                        ),
                    )
                };
                if let (GPUCommandJoinState::Ready(r1), GPUCommandJoinState::Ready(r2)) = rs {
                    Poll::Ready((r1, r2))
                } else {
                    unreachable!()
                }
            }
            (GPUCommandJoinStateProj::Taken, GPUCommandJoinStateProj::Taken) => {
                panic!("Attempted to poll GPUCommandJoin after completion")
            }
            _ => Poll::Pending,
        }
    }

    fn context(&self) -> GPUCommandFutureContext {
        let mut ctx = GPUCommandFutureContext::default();
        if let GPUCommandJoinState::Pending(g) = &self.inner1 {
            ctx = ctx.merge(&g.context());
        }
        if let GPUCommandJoinState::Pending(g) = &self.inner2 {
            ctx = ctx.merge(&g.context());
        }
        ctx
    }

    fn init(self: Pin<&mut Self>) {
        let this = self.project();
        if let GPUCommandJoinStateProj::Pending(g) = this.inner1.project() {
            g.init();
        }
        if let GPUCommandJoinStateProj::Pending(g) = this.inner2.project() {
            g.init();
        }
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
    fn context(&self) -> GPUCommandFutureContext {
        self.inner.context()
    }
    fn init(self: Pin<&mut Self>) {
        let this = self.project();
        this.inner.init();
    }
}
