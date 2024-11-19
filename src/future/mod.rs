mod ctx;
mod res;
mod exec;
pub mod commands;
use core::task::ContextBuilder;
use std::{any::Any, future::{Future, IntoFuture}, marker::PhantomData, mem::MaybeUninit, pin::Pin, str::FromStr, task::Poll};

pub use ctx::*;

pub use rhyolite_macros::gpu_future;

pub trait GPUFuture: Sized + Unpin {
    type Output;
    type Retained = ();
    fn barrier(&mut self, ctx: BarrierContext) {
    }
    fn record(self, ctx: RecordContext) -> (Self::Output, Self::Retained);
}

pub enum GPUFutureState<T: GPUFuture> {
    Barrier(T),
    Record(T),
    Finished
}

#[macro_export]
macro_rules! define_future {
    ($type_name: ty) => {
        impl std::future::IntoFuture for $type_name {
            type Output = crate::GPUFutureBlockReturnValue<<Self as GPUFuture>::Output, <Self as GPUFuture>::Retained>;
            type IntoFuture = crate::GPUFutureState<Self>;
        
            fn into_future(self) -> Self::IntoFuture {
                crate::GPUFutureState::Barrier(self)
            }
        }
    };
    ($type_name: ty, $($args:tt)*) => {
        impl<$($args)*> std::future::IntoFuture for $type_name {
            type Output = crate::GPUFutureBlockReturnValue<<Self as GPUFuture>::Output, <Self as GPUFuture>::Retained>;
            type IntoFuture = crate::GPUFutureState<Self>;
        
            fn into_future(self) -> Self::IntoFuture {
                crate::GPUFutureState::Barrier(self)
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
                let ctx = cx.ext().downcast_mut::<GPUFutureContext>().expect("Attempting to run a regular future in a GPU context");
                future.barrier(ctx.barrier_ctx());
                *this = Self::Record(future);
                std::task::Poll::Pending
            },
            GPUFutureState::Record(future) => {
                let ctx = cx.ext().downcast_mut::<GPUFutureContext>().expect("Attempting to run a regular future in a GPU context");
                let (output, retained_values) = future.record(ctx.record_ctx());
                std::task::Poll::Ready(GPUFutureBlockReturnValue {
                    output,
                    retained_values,
                })
            },
            GPUFutureState::Finished => {
                panic!()
            },
        }
    }
}

pub struct GPUFutureBlockReturnValue<O, R> {
    output: O,
    retained_values: R,
}

pub trait GPUFutureBlock: Future<Output = GPUFutureBlockReturnValue<<Self as GPUFutureBlock>::Output, <Self as GPUFutureBlock>::Retained>> {
    type Output;
    type Retained;
}
impl<O, R, T: Future<Output = GPUFutureBlockReturnValue<O, R>>> GPUFutureBlock for T {
    type Output = O;
    type Retained = R;
}

pub struct GPURetained<'a, T>(&'a mut T);
impl<'a, T> GPURetained<'a, T> {
    pub fn __retain(r: &'a mut T) -> Self {
        Self(r)
    }
}
impl<'a, T> std::ops::Deref for GPURetained<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.0
    }
}
impl<'a, T> std::ops::DerefMut for GPURetained<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}


unsafe fn maybe_uninit_new<'a, T>(obj: &'a mut T) -> &'a mut MaybeUninit<T> {
    std::mem::transmute(obj)
}

fn test() {
    let a = gpu_future! {
        let mut a: Option<GPURetained<'_, Vec<u32>>> = None;
        if true {

            let mut retained = retain!(Vec::<u32>::new());
            retained.push(12_u32);
        }
    };
}
