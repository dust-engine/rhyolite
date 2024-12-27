mod closure;
mod combinator;
mod image;

pub use closure::*;
pub use combinator::*;
pub use image::*;

use crate::define_future;

use crate::future::{BarrierContext, GPUFuture, RecordContext};

pub struct Yield;
define_future!(Yield);
impl GPUFuture for Yield {
    type Output = ();

    fn barrier(&mut self, _ctx: BarrierContext) {}

    fn record(self, _ctx: RecordContext) -> (Self::Output, Self::Retained) {
        Default::default()
    }
}
