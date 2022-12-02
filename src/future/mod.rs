use std::marker::PhantomPinned;
use std::ops::GeneratorState;
use std::{ops::Generator, sync::Arc};
use ash::vk;
use std::pin::Pin;
use std::task::Poll;

use crate::Device;

pub trait GPUCommandFuture {
    type Output;
    fn record(self: Pin<&mut Self>, command_buffer: vk::CommandBuffer) -> Poll<Self::Output>;
}

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

pub trait GPUCommandGenerator = Generator<vk::CommandBuffer, Yield = ()>;

pub struct GPUCommandBlock<G: GPUCommandGenerator> {
    inner: G,
}
impl<G: GPUCommandGenerator> GPUCommandBlock<G> {
    fn into_inner(self: Pin<&mut Self>) -> Pin<&mut G> {
        unsafe {
            self.map_unchecked_mut(|a| &mut a.inner)
        }
    }
}
impl<G: GPUCommandGenerator> GPUCommandFuture for GPUCommandBlock<G> {
    type Output = G::Return;
    fn record(self: Pin<&mut Self>, command_buffer: vk::CommandBuffer) -> Poll<G::Return> {
        let generator = self.into_inner();
        match generator.resume(command_buffer) {
            GeneratorState::Yielded(()) => Poll::Pending,
            GeneratorState::Complete(r) => Poll::Ready(r),
        }
    }
}

enum GPUCommandJoinState<G: GPUCommandFuture> {
    Pending(G),
    Ready(G::Output),
    Taken
}
pub struct GPUCommandJoin<G1, G2>
where G1: GPUCommandFuture,
    G2: GPUCommandFuture {
    inner1: GPUCommandJoinState<G1>,
    inner2: GPUCommandJoinState<G2>,
}

impl<G1, G2> GPUCommandFuture for GPUCommandJoin<G1, G2>
where G1: GPUCommandFuture,
    G2: GPUCommandFuture {
    type Output = (G1::Output, G2::Output);
    fn record(self: Pin<&mut Self>, command_buffer: vk::CommandBuffer) -> Poll<Self::Output> {
        unsafe {
            let this = self.get_unchecked_mut();
            match &mut this.inner1 {
                GPUCommandJoinState::Pending(g) => {
                    let g = Pin::new_unchecked(g);
                    match g.record(command_buffer) {
                        Poll::Pending => (),
                        Poll::Ready(r) => {
                            this.inner1 = GPUCommandJoinState::Ready(r);
                        }
                    }
                },
                _ => {}
            }
            match &mut this.inner2 {
                GPUCommandJoinState::Pending(g) => {
                    let g = Pin::new_unchecked(g);
                    match g.record(command_buffer) {
                        Poll::Pending => (),
                        Poll::Ready(r) => {
                            this.inner2 = GPUCommandJoinState::Ready(r);
                        }
                    }
                },
                _ => {}
            }
            match (&mut this.inner1, &mut this.inner2) {
                (GPUCommandJoinState::Ready(r1), GPUCommandJoinState::Ready(r2)) => {
                    match (
                        std::mem::replace(&mut this.inner1, GPUCommandJoinState::Taken), 
                        std::mem::replace(&mut this.inner2, GPUCommandJoinState::Taken)
                    ) {
                        (GPUCommandJoinState::Ready(r1), GPUCommandJoinState::Ready(r2)) => {
                            Poll::Ready((r1, r2))
                        },
                        _ => unreachable!()
                    }
                },
                (GPUCommandJoinState::Taken, GPUCommandJoinState::Taken) => panic!("Attempted to poll GPUCommandJoin after completion"),
                _ => Poll::Pending,
            }
        }
    }
}

pub struct GPUCommandMap<G, F> {
    mapper: Option<F>,
    inner: G,
}

impl<G, R, F> GPUCommandFuture for GPUCommandMap<G, F>
    where G: GPUCommandFuture,
        F: FnOnce(G::Output) -> R {
    type Output = R;
    fn record(mut self: Pin<&mut Self>, command_buffer: vk::CommandBuffer) -> Poll<Self::Output> {
        let inner = unsafe {
            self.as_mut().map_unchecked_mut(|a| &mut a.inner)
        };
        match inner.record(command_buffer) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(r) => {
                let mapper = unsafe {
                    self.get_unchecked_mut().mapper.take().expect("Attempted to poll GPUCommandMap after completion")
                };
                Poll::Ready((mapper)(r))
            }
        }
    }
}

#[test]
fn test() {
    let block1 = async_ash_macro::commands! {
        let fut1 = CopyBufferFuture{ str: "prev1"};
        fut1.await;
        let fut1 = CopyBufferFuture{ str: "prev2"};
        fut1.await;
        1_u32
    };

    let block2 = async_ash_macro::commands! {
        let fut1 = CopyBufferFuture{ str: "A"};
        fut1.await;
        let fut1 = CopyBufferFuture{ str: "B"};
        fut1.await;
        2_u64
    };
    let block3 =  CopyBufferFuture{ str: "special"};
    let block = async_ash_macro::commands! {
        let (a, b, c) = async_ash_macro::join!(block1, block2, block3).await;
        println!("a: {:?}, b: {:?}, c: {:?}", a, b, c);
    };

    let mut block = std::pin::pin!(block);
    for i in 0..4 {
        match block.as_mut().record(vk::CommandBuffer::null()) {
            Poll::Ready(()) =>  {
                println!("Ready");
            }
            Poll::Pending => {
                println!("Pending");
            }
        }
    }
}

struct CopyBufferFuture {
    str: &'static str,
}
impl GPUCommandFuture for CopyBufferFuture {
    type Output = ();
    fn record(self: Pin<&mut Self>, command_buffer: vk::CommandBuffer) -> Poll<Self::Output> {
        println!("{}", self.str);
        Poll::Ready(())
    }
}

/*

let branch1 = commands! {
    A.await;
    B.await;
    (C, D, E).join().await;
    F.await;
}
let branch2 = commands! {
    (X, Y).join().await;
    Z.await;
}
let fianl = commands! {
    (branch1, branch2).join().await;
    G.await;
}

Result should be:
A, X, Y
barrier
B, Z
barrier
C, D, E
barrier
F
barrier
G


alternatively
A
barrier
B X Y
barrier
C D E Z
barrier
F
barrier
G

alternatively

A
barrier
B
barrier
C D E X Y
barrier
F Z
barrier
G

*/