use crate::define_future;
use crate::future::{BarrierContext, GPUFuture, RecordContext};

pub fn record_commands<Ctx: Unpin, F: Unpin, B: Unpin, Out>(
    context: Ctx,
    record: F,
    barrier: B,
) -> ClosureFuture<Ctx, F, B, Out>
where
    F: FnOnce(RecordContext, Ctx) -> Out,
    B: FnMut(BarrierContext, &mut Ctx),
{
    ClosureFuture {
        context,
        record,
        barrier,
    }
}

define_future!(ClosureFuture<Ctx, F, B, Out>, Ctx: Unpin, F: Unpin + FnOnce(RecordContext, Ctx) -> Out,B: Unpin + FnMut(BarrierContext, &mut Ctx),Out);
pub struct ClosureFuture<Ctx, F, B, Out>
where
    F: FnOnce(RecordContext, Ctx) -> Out,
    B: FnMut(BarrierContext, &mut Ctx),
{
    context: Ctx,
    record: F,
    barrier: B,
}

impl<Ctx: Unpin, F: Unpin, B: Unpin, Out> GPUFuture for ClosureFuture<Ctx, F, B, Out>
where
    F: FnOnce(RecordContext, Ctx) -> Out,
    B: FnMut(BarrierContext, &mut Ctx),
{
    type Output = Out;

    fn barrier(&mut self, ctx: BarrierContext) {
        (self.barrier)(ctx, &mut self.context);
    }
    fn record(self, ctx: RecordContext) -> (Self::Output, Self::Retained) {
        let output = (self.record)(ctx, self.context);
        (output, ())
    }
}
