use std::task::{RawWaker, RawWakerVTable, Waker};

use futures_util::Future;

use crate::QueueType;

use super::GPUFuture;

unsafe fn clone_noop(data: *const ()) -> RawWaker {
    RawWaker::new(data, GPU_VTABLE)
}
unsafe fn noop(_data: *const ()) {}
pub(super) static GPU_VTABLE: &'static RawWakerVTable =
    &RawWakerVTable::new(clone_noop, noop, noop, noop);

pub struct GPUContext {
    pub current_priority: u64,
    pub(super) current_queue: Option<u32>,
}

impl GPUContext {
    pub unsafe fn waker(&mut self) -> Waker {
        let raw_waker = std::task::RawWaker::new(self as *mut _ as *const (), GPU_VTABLE);
        let waker = std::task::Waker::from_raw(raw_waker);
        waker
    }

    pub fn queue_type_to_index(&self, queue_type: QueueType) -> u32 {
        match queue_type {
            QueueType::Graphics => 0,
            QueueType::Compute => 1,
            QueueType::Transfer => 2,
            QueueType::SparseBinding => todo!(),
        }
    }
}

/// Provides convenient method to retreive GPUContext from `std::task::Context`
pub trait GPUTaskContext {
    fn get(&mut self) -> &mut GPUContext;
}

impl<'a> GPUTaskContext for std::task::Context<'a> {
    fn get(&mut self) -> &mut GPUContext {
        if self.waker().as_raw().vtable() as *const RawWakerVTable
            != GPU_VTABLE as *const RawWakerVTable
        {
            panic!("Trying to execute a GPUFuture in a regular executor");
        }
        unsafe { &mut *(self.waker().as_raw().data() as *mut GPUContext) }
    }
}

pub struct QueueTransferFuture {
    queue_type: QueueType,
}
impl Future for QueueTransferFuture {
    type Output = ();

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let cx = cx.get();
        let index = cx.queue_type_to_index(self.queue_type);
        cx.current_queue = Some(index);
        std::task::Poll::Ready(())
    }
}
impl GPUFuture for QueueTransferFuture {
    fn priority(&self) -> u64 {
        0
    }
}
pub fn queue_transfer(queue_type: QueueType) -> QueueTransferFuture {
    QueueTransferFuture { queue_type }
}
