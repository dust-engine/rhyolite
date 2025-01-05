mod ctx;
mod exec;
mod ext;
mod res;
use std::{future::Future, mem::MaybeUninit, pin::Pin, task::Poll};

pub use ctx::*;
pub use exec::*;
pub use ext::GPUFutureBlockExt;
pub use res::*;

pub use rhyolite_macros::gpu_future;

pub trait GPUFuture: Sized + Unpin {
    type Output;
    type Retained = ();
    fn barrier(&mut self, _ctx: BarrierContext) {}
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
                future.barrier(ctx.barrier_ctx_barrier());
                *this = Self::Record(future);
                std::task::Poll::Pending
            }
            GPUFutureState::Record(mut future) => {
                let ctx = cx
                    .ext()
                    .downcast_mut::<GPUFutureContext>()
                    .expect("Attempting to run a regular future in a GPU context");
                future.barrier(ctx.barrier_ctx_record());
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
    type Returned: Send + Sync + 'static;
    type Retained: Send + Sync + 'static;
}
impl<O, R, T: Future<Output = GPUFutureBlockReturnValue<O, R>>> GPUFutureBlock for T
where
    O: Send + Sync + 'static,
    R: Send + Sync + 'static,
{
    type Returned = O;
    type Retained = R;
}

unsafe fn maybe_uninit_new<'a, T>(obj: &'a mut T) -> &'a mut MaybeUninit<T> {
    std::mem::transmute(obj)
}
