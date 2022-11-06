// queue.do(|command_recorder| xxxx) -> future
// queue.run_graph(graph) -> future
// swapchain.acquire_next_image() -> future
// queue.present(future) doesn't return future. instead, it takes a future.

/*
gpu! {

} should return an opaque impl GPUFuture.


1. What we're really trying to do, is to build a graph.
*/

use ash::vk;
use async_ash_macro::gpu;
use std::future::Future;
use std::pin::Pin;

pub trait GPUFuture: Future {
    fn priority(&self) -> u64;
}

enum GPUFutureState<F: GPUFuture> {
    Ready(F::Output),
    Pending(F),
    None,
}
struct JoinedGPUFuture<F1: GPUFuture, F2: GPUFuture> {
    f1: GPUFutureState<F1>,
    f2: GPUFutureState<F2>,
}

impl<F1: GPUFuture, F2: GPUFuture> JoinedGPUFuture<F1, F2> {
    fn new(f1: F1, f2: F2) -> Self {
        Self {
            f1: GPUFutureState::Pending(f1),
            f2: GPUFutureState::Pending(f2),
        }
    }
}
use std::task::{Poll, RawWakerVTable};
impl<F1: GPUFuture, F2: GPUFuture> Future for JoinedGPUFuture<F1, F2> {
    type Output = (F1::Output, F2::Output);

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        unsafe {
            let this = self.get_unchecked_mut();
            // Both are pending: run the highest priority one.

            match (&mut this.f1, &mut this.f2) {
                (GPUFutureState::Ready(f1), GPUFutureState::Ready(f2)) => {
                    panic!("Continued to call pull after finished")
                }
                (GPUFutureState::Ready(_), GPUFutureState::Pending(f2)) => {
                    let f2 = Pin::new_unchecked(f2);
                    match f2.poll(cx) {
                        Poll::Ready(result) => {
                            this.f2 = GPUFutureState::Ready(result);
                        }
                        _ => (),
                    }
                }
                (GPUFutureState::Pending(f1), GPUFutureState::Ready(_)) => {
                    let f1 = Pin::new_unchecked(f1);
                    match f1.poll(cx) {
                        Poll::Ready(result) => {
                            this.f1 = GPUFutureState::Ready(result);
                        }
                        _ => (),
                    }
                }
                (GPUFutureState::Pending(f1), GPUFutureState::Pending(f2)) => {
                    if f1.priority() > f2.priority() {
                        let f1 = Pin::new_unchecked(f1);
                        match f1.poll(cx) {
                            Poll::Ready(result) => {
                                this.f1 = GPUFutureState::Ready(result);
                            }
                            _ => (),
                        }
                    } else {
                        let f2 = Pin::new_unchecked(f2);
                        match f2.poll(cx) {
                            Poll::Ready(result) => {
                                this.f2 = GPUFutureState::Ready(result);
                            }
                            _ => (),
                        }
                    }
                }
                _ => unreachable!(),
            }

            match (&mut this.f1, &mut this.f2) {
                (GPUFutureState::Ready(_), GPUFutureState::Ready(_)) => {
                    let f1 = std::mem::replace(&mut this.f1, GPUFutureState::None);
                    let f2 = std::mem::replace(&mut this.f2, GPUFutureState::None);
                    match (f1, f2) {
                        (GPUFutureState::Ready(f1), GPUFutureState::Ready(f2)) => {
                            return Poll::Ready((f1, f2))
                        }
                        _ => unreachable!(),
                    }
                }
                _ => {
                    let gpu_ctx = &mut *(cx.waker().as_raw().data() as *mut waker::GPUContext);
                    gpu_ctx.current_priority = this.priority();
                    Poll::Pending
                }
            }
        }
    }
}
impl<F1: GPUFuture, F2: GPUFuture> GPUFuture for JoinedGPUFuture<F1, F2> {
    fn priority(&self) -> u64 {
        match (&self.f1, &self.f2) {
            (GPUFutureState::Pending(f1), GPUFutureState::Pending(f2)) => {
                f1.priority().max(f2.priority())
            }
            (GPUFutureState::Ready(_), GPUFutureState::Pending(f2)) => f2.priority(),
            (GPUFutureState::Pending(f1), GPUFutureState::Ready(_)) => f1.priority(),
            _ => unreachable!(),
        }
    }
}

mod waker {
    use std::task::{RawWaker, RawWakerVTable, Waker};

    use super::GPUFuture;
    unsafe fn clone(data: *const ()) -> RawWaker {
        panic!("Trying to execute a regular Future in a GPU executor");
    }
    unsafe fn wake(data: *const ()) {
        panic!("Trying to execute a regular Future in a GPU executor");
    }
    unsafe fn wake_by_ref(data: *const ()) {
        panic!("Trying to execute a regular Future in a GPU executor");
    }
    unsafe fn drop(data: *const ()) {
        panic!("Trying to execute a regular Future in a GPU executor: Waker dropped");
    }
    pub(super) const GPU_VTABLE: RawWakerVTable =
        RawWakerVTable::new(clone, wake, wake_by_ref, drop);

    pub struct GPUContext {
        pub current_priority: u64,
        pub was_hook_run: bool,
    }

    impl GPUContext {
        pub unsafe fn waker(&mut self) -> Waker {
            let raw_waker = std::task::RawWaker::new(self as *mut _ as *const (), &GPU_VTABLE);
            let waker = std::task::Waker::from_raw(raw_waker);
            waker
        }
    }
}
pub use waker::GPUContext;

/// This needs to be awaited before awaiting any GPUFuture.
/// This provides us an opportunity to grab any initial states from the GPUFuture.
struct GPUFutureHook<'a, F: GPUFuture> {
    future: &'a F,
    executed: bool,
}
impl<'a, F: GPUFuture> GPUFutureHook<'a, F> {
    fn new(future: &'a F) -> Self {
        Self {
            future,
            executed: false,
        }
    }
}
impl<'a, F: GPUFuture> Future for GPUFutureHook<'a, F> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        if self.executed {
            return Poll::Ready(());
        }
        unsafe {
            if cx.waker().as_raw().vtable() != &waker::GPU_VTABLE {
                panic!("Trying to execute a GPUFuture in a regular executor");
            }
            let gpu_ctx = &mut *(cx.waker().as_raw().data() as *mut waker::GPUContext);
            gpu_ctx.current_priority = self.future.priority();
            gpu_ctx.was_hook_run = true;
            self.executed = true;
        }
        Poll::Pending
    }
}

pub struct GPUFutureBlock<F: Future> {
    priority: u64,
    f: F,
}

impl<F: Future> GPUFutureBlock<F> {
    pub fn new(f: F) -> Self {
        Self {
            f,
            priority: u64::MAX, // Initial priority is always max. The initial hook would then get triggered.
        }
    }
}

impl<F: Future> Future for GPUFutureBlock<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        unsafe {
            let gpu_cx = &mut *(cx.waker().as_raw().data() as *mut waker::GPUContext);

            let this = self.get_unchecked_mut();
            let f = Pin::new_unchecked(&mut this.f);

            gpu_cx.current_priority = this.priority;
            gpu_cx.was_hook_run = false;
            let result = f.poll(cx);

            let gpu_cx = &mut *(cx.waker().as_raw().data() as *mut waker::GPUContext);
            this.priority = gpu_cx.current_priority;
            result
        }
    }
}
impl<F: Future> GPUFuture for GPUFutureBlock<F> {
    fn priority(&self) -> u64 {
        self.priority
    }
}

mod tests {
    use super::*;
    struct MyFuture {
        str: String,
        executed: bool,
        priority: u64,
    }
    impl MyFuture {
        fn new(str: String, priority: u64) -> Self {
            Self {
                executed: false,
                str,
                priority,
            }
        }
    }
    impl Future for MyFuture {
        type Output = ();
        fn poll(
            mut self: Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Self::Output> {
            let gpu_cx = unsafe { &mut *(cx.waker().as_raw().data() as *mut waker::GPUContext) };

            println!("Running {}", self.str);
            Poll::Ready(())
        }
    }
    impl GPUFuture for MyFuture {
        fn priority(&self) -> u64 {
            self.priority
        }
    }

    fn traaay() {
        let future = gpu! {
            MyFuture::new("myfuture".into(), 12253).await;
            MyFuture::new("myfuture1".into(), 12254).await;
            MyFuture::new("myfuture2".into(), 12255).await;
            MyFuture::new("myfuture1112".into(), 12222255).await
        };
    }

    #[test]
    fn test2() {
        GPUFutureBlock::new(async {
            MyFuture::new("myfuture".into(), 12253).await;
        });
    }

    #[test]
    fn test() {
        let mut future = gpu! {
            let f1 = gpu! {

                MyFuture::new("Hello".into(), 100).await;
                MyFuture::new("World".into(), 2).await;
            };
            let f2 = gpu! {
                MyFuture::new("Hello2".into(), 1000).await;
                MyFuture::new("World2".into(), 1).await;
            };
            JoinedGPUFuture::new(f1, f2).await;
            MyFuture::new("Now".into(), 1).await;
        };

        let mut count = 0;
        let mut gpuctx = waker::GPUContext {
            current_priority: 100,
            was_hook_run: false,
        };
        let waker = unsafe { gpuctx.waker() };
        let mut ctx = std::task::Context::from_waker(&waker);
        loop {
            use std::future::Future;
            count += 1;
            let a = unsafe { Pin::new_unchecked(&mut future) };
            if a.poll(&mut ctx).is_ready() {
                break;
            }
        }
        println!("Pulled {} times", count);

        // Don't drop the waker. It's just a pointer to gpuctx and a pointer to the vtable.
        // Only regular executors would drop the waker here.
        std::mem::forget(waker);
    }
}
