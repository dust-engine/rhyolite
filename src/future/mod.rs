pub mod commands;
mod ctx;
mod exec;
mod res;
use core::task::ContextBuilder;
use std::{
    any::Any,
    future::{Future, IntoFuture},
    marker::PhantomData,
    mem::MaybeUninit,
    pin::Pin,
    str::FromStr,
    task::Poll,
};

pub use ctx::*;
pub use res::*;

pub use rhyolite_macros::gpu_future;

use crate::semaphore::SemaphoreDeferredValue;

pub trait GPUFuture: Sized + Unpin {
    type Output;
    type Retained = ();
    fn barrier(&mut self, ctx: BarrierContext) {}
    fn record(self, ctx: RecordContext) -> (Self::Output, Self::Retained);
}

pub enum GPUFutureState<T: GPUFuture> {
    Barrier(T),
    Record(T),
    Finished,
}

#[macro_export]
macro_rules! define_future {
    ($type_name: ty) => {
        impl std::future::IntoFuture for $type_name {
            type Output = rhyolite::future::GPUFutureBlockReturnValue<<Self as GPUFuture>::Output, <Self as GPUFuture>::Retained>;
            type IntoFuture = rhyolite::future::GPUFutureState<Self>;

            fn into_future(self) -> Self::IntoFuture {
                rhyolite::future::GPUFutureState::Barrier(self)
            }
        }
    };
    ($type_name: ty, $($args:tt)*) => {
        impl<$($args)*> std::future::IntoFuture for $type_name {
            type Output = rhyolite::future::GPUFutureBlockReturnValue<<Self as GPUFuture>::Output, <Self as GPUFuture>::Retained>;
            type IntoFuture = rhyolite::future::GPUFutureState<Self>;

            fn into_future(self) -> Self::IntoFuture {
                rhyolite::future::GPUFutureState::Barrier(self)
            }
        }
    };
}

impl<T: GPUFuture> Future for GPUFutureState<T> {
    type Output = GPUFutureBlockReturnValue<<T as GPUFuture>::Output, <T as GPUFuture>::Retained>;

    fn poll(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match std::mem::replace(this, Self::Finished) {
            GPUFutureState::Barrier(mut future) => {
                // Emit barrier
                let ctx = cx
                    .ext()
                    .downcast_mut::<GPUFutureContext>()
                    .expect("Attempting to run a regular future in a GPU context");
                future.barrier(ctx.barrier_ctx());
                *this = Self::Record(future);
                std::task::Poll::Pending
            }
            GPUFutureState::Record(future) => {
                let ctx = cx
                    .ext()
                    .downcast_mut::<GPUFutureContext>()
                    .expect("Attempting to run a regular future in a GPU context");
                let (output, retained_values) = future.record(ctx.record_ctx());
                std::task::Poll::Ready(GPUFutureBlockReturnValue {
                    output,
                    retained_values,
                })
            }
            GPUFutureState::Finished => {
                panic!()
            }
        }
    }
}

pub struct GPUFutureBlockReturnValue<O, R> {
    pub output: O,
    pub retained_values: R,
}

pub trait GPUFutureBlock:
    Future<
    Output = GPUFutureBlockReturnValue<
        <Self as GPUFutureBlock>::Returned,
        <Self as GPUFutureBlock>::Retained,
    >,
>
{
    type Returned;
    type Retained;
}
impl<O, R, T: Future<Output = GPUFutureBlockReturnValue<O, R>>> GPUFutureBlock for T {
    type Returned = O;
    type Retained = R;
}

unsafe fn maybe_uninit_new<'a, T>(obj: &'a mut T) -> &'a mut MaybeUninit<T> {
    std::mem::transmute(obj)
}

pub struct InFlightFrameMananger<T, const C: usize = 3> {
    current_frame: usize,
    frame_data: [Option<T>; C],
}
impl<T, const C: usize> InFlightFrameMananger<T, C> {
    pub fn take(&mut self) -> Option<T> {
        self.frame_data[self.current_frame].take()
    }

    pub fn next_frame(&mut self, frame_data: T) {
        let dst = &mut self.frame_data[self.current_frame];
        assert!(
            dst.is_none(),
            "You must call `take` before calling `add_frame`"
        );
        *dst = Some(frame_data);
        self.current_frame += 1;
        if self.current_frame >= C {
            self.current_frame = 0;
        }
    }
    pub fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.frame_data.iter_mut().filter_map(|x| x.take())
    }
}
impl<T, const C: usize> Default for InFlightFrameMananger<T, C> {
    fn default() -> Self {
        Self {
            current_frame: 0,
            frame_data: [0; C].map(|_| None),
        }
    }
}
