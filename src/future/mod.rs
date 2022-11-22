use std::{future::Future, pin::Pin, ops::Generator, marker::PhantomData};
use ash::vk;
use async_ash_macro::gpu;

use crate::{Queue, swapchain::SwapchainImage};

pub enum QueueOperation {
    Submit,
    BindSparse,
}


pub trait GPUFuture {
    type Output;

    /// Produce work on the GPU side. Will be executed in one go.
    fn schedule(&mut self) -> impl Iterator<Item = QueueOperation> + '_;

    /// Execute on the host side. Tasks will be executed as they become available.
    async fn execute(self) -> Self::Output;
}

pub struct CopyBuffer {

}

impl GPUFuture for CopyBuffer {
    type Output = ();

    fn schedule(&mut self) -> impl Iterator<Item = QueueOperation> {
        vec![QueueOperation::Submit].into_iter()
    }

    async fn execute(self) -> Self::Output {
        () // just need to await semaphore
    }
}

#[derive(Debug)]
struct Apple;

fn copy_buffer(src: &Apple, dst: &mut Apple) -> CopyBuffer {
    CopyBuffer {}
}

/// Generators cannot implement `for<'a, 'b> Generator<&'a mut Context<'b>>`, so we need to pass
/// a raw pointer (see https://github.com/rust-lang/rust/issues/68923).
/// The correct way to write this would be:
/// ```rs
/// trait CompositeGPUFutureGenerator<E> = for<'a> Generator<&'a mut E, Return=(), Yield = QueueOperation> + Unpin;
/// ```
trait CompositeGPUFutureGenerator<E> = Generator<*mut E, Return=(), Yield = QueueOperation> + Unpin;
struct CompositeGPUFuture<E, G: CompositeGPUFutureGenerator<E>, F: Future<Output = ()>, C: FnOnce(E) -> F> {
    i: G,
    c: C,
    environment: E
}
impl<E, G: CompositeGPUFutureGenerator<E>, F: Future<Output = ()>, C: FnOnce(E) -> F> GPUFuture for CompositeGPUFuture<E, G, F, C> {
    type Output = F::Output;

    fn schedule(&mut self) -> impl Iterator<Item = QueueOperation> + '_ {
        CompositeGPUFutureIterator {
            g: &mut self.i,
            e: &mut self.environment
        }
    }

    fn execute(self) -> impl Future<Output = Self::Output> {
        (self.c)(self.environment)
    }
}
struct CompositeGPUFutureIterator<'a, E, G: CompositeGPUFutureGenerator<E>> {
    g: &'a mut G,
    e: &'a mut E,
}

impl<'a, E, G: CompositeGPUFutureGenerator<E>> Iterator for CompositeGPUFutureIterator<'a, E, G> {
    type Item = QueueOperation;

    fn next(&mut self) -> Option<Self::Item> {
        use std::ops::GeneratorState;
        let g: &mut G = self.g;
        let g = Pin::new(g);
        match g.resume(self.e) {
            GeneratorState::Yielded(n) => Some(n),
            GeneratorState::Complete(()) => None,
        }
    }
}

/// Helper function to reference a pointer. This is to help Rust with type inference.

fn get_apple() -> Apple {
    Apple {}
}

struct TypeHint<T> {
    _marker: PhantomData<T>
}
impl<T> Clone for TypeHint<T> {
    fn clone(&self) -> Self {
        Self { _marker: PhantomData }
    }
}
impl<T> Copy for TypeHint<T> {
}
impl<T> TypeHint<T> {
    pub fn new(var: &T) -> Self {
        Self {
            _marker: PhantomData
        }
    }
    pub fn pin_ptr_mut(&self, v: *mut T) {
    }
    pub fn pin_ref(&self, v: &T) {
    }
}


fn test() {
    let mut e1 = get_apple();
    let mut e2 = get_apple();
    let future = gpu! {
        copy_buffer(&e1, &mut e2).await;
    };
    // 1. get all the futures, owned
    // 2. call schedule on all of them, without taking the ownership, before hand.
    // 3. call execute on all of them, async, taking ownership.

    let mut a1 = get_apple();
    let mut a2 = get_apple();
    let mut marker: i32 = 0;
    let futures = {
        let f1 = copy_buffer(&a1, &mut a2);
        let f2 = copy_buffer(&a2, &mut a1); // <- create a let binding for each future awaited.
        (f1, f2)
    };


    let captures = (&mut marker, futures);

    let type_hint = TypeHint::new(&captures);
    let i = move |aaa| {
        type_hint.pin_ptr_mut(aaa);
        let (marker, futures) = unsafe { &mut *aaa };
        let (f1, f2) = futures;
        for i in f1.schedule() {
            yield i;
        }
        for i in f2.schedule() {
            yield i;
        }
    };
    let c = |aaa| async move {
        type_hint.pin_ref(&aaa);
        let (marker, futures) = aaa;
        let (f1, f2) = futures;
        f1.execute().await;
        f2.execute().await;
    };

    let future = CompositeGPUFuture {
        i,
        c,
        environment: captures
    };
}
/*
async {
    let image = swapchain.acquire_image().await;
    copy_stuff_to_image(&mut image);
    gpu_post_processing(&mut image).await;
    queue.present(image);

}

converted to
{
    waitSemaphore(0); // elided
    // Do Nothing
    // GPU is responsible for signalSemaphore(1)
    waitSemaphore(1);
    copy_stuff_to_image(&mut image);
    signalSemaphore(2);
    // gpu_post_processing; no op on host
    waitSemaphore(3);
    signal_semaphore(4);
    // queue.present();
    wait_semaphore(5);
    drop some stuff.
} to be executed asyncly

and
{
swapchain.acquire_image(signal 1)
gpu_post_processing(&mut image, signal 3).await
queue.present(image)
}
to be executed now


*/
