mod context;
mod join;

pub use context::*;
pub use join::*;

use async_ash_macro::gpu;
use std::future::Future;
use std::pin::Pin;
use std::task::Poll;

pub trait GPUFuture: Future {
    fn priority(&self) -> u64;
}

/// This needs to be awaited before awaiting any GPUFuture.
/// This provides us an opportunity to grab any initial states from the GPUFuture.
pub struct GPUFutureHook<'a, F: GPUFuture> {
    future: &'a F,
    executed: bool,
}
impl<'a, F: GPUFuture> GPUFutureHook<'a, F> {
    pub fn new(future: &'a F) -> Self {
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
        let gpu_ctx = cx.get();
        gpu_ctx.current_priority = self.future.priority();
        self.executed = true;
        Poll::Pending
    }
}

pub mod tests {
    use super::*;
    pub struct MyFuture {
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
    pub fn test() {
        let mut future = gpu! {
            let f1 = gpu! {
                MyFuture::new("hello".into(), 12253).await;
            };
            f1.await;
        };

        let mut count = 0;
        let mut gpuctx = GPUContext {
            current_priority: 100,
            current_queue: None,
        };
        let waker = unsafe { gpuctx.waker() };
        let mut ctx = std::task::Context::from_waker(&waker);
        loop {
            use std::future::Future;
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

/*
Option 1
gpu! {
    common_task().await;
    separate_task1();
    separate_task2();
}

Option 2
gpu! {
    common_task().awaitl
}.shared()

gpu! {
    swapchain.acquire().await

}

*/
