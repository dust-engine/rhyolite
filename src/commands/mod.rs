mod combinator;
mod image;

pub use combinator::*;
pub use image::*;

use crate::define_future;

use crate::future::{GPUFuture, GPUFutureBarrierContext, RecordContext};

pub struct Yield;
define_future!(Yield);
impl GPUFuture for Yield {
    type Output = ();

    fn barrier<Ctx: GPUFutureBarrierContext>(&mut self, ctx: Ctx) {}

    fn record(self, ctx: RecordContext) -> (Self::Output, Self::Retained) {
        Default::default()
    }
}
pub fn yield_now() -> Yield {
    Yield
}
