use std::task::{RawWaker, RawWakerVTable, Waker};

use crate::QueueType;


unsafe fn clone_noop(data: *const ()) -> RawWaker {
    RawWaker::new(data, GPU_VTABLE)
}
unsafe fn noop(_data: *const ()) {}
pub(super) static GPU_VTABLE: &'static RawWakerVTable = &RawWakerVTable::new(clone_noop, noop, noop, noop);

pub struct GPUContext {
    pub current_priority: u64,
    pub(super) current_queue: u32,
}

impl GPUContext {
    pub unsafe fn waker(&mut self) -> Waker {
        let raw_waker = std::task::RawWaker::new(self as *mut _ as *const (), GPU_VTABLE);
        let waker = std::task::Waker::from_raw(raw_waker);
        waker
    }

    pub fn set_queue_type(&mut self, queue: QueueType) {
        todo!()
    }
}



/// Provides convenient method to retreive GPUContext from `std::task::Context`
pub trait GPUTaskContext {
    fn get(&mut self) -> &mut GPUContext;
}

impl<'a> GPUTaskContext for std::task::Context<'a> {
    fn get(&mut self) -> &mut GPUContext {
        if self.waker().as_raw().vtable() as *const RawWakerVTable != GPU_VTABLE as *const RawWakerVTable {
            panic!("Trying to execute a GPUFuture in a regular executor");
        }
        unsafe {
            &mut *(self.waker().as_raw().data() as *mut GPUContext)
        }
    }
}
