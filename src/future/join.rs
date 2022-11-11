use std::{pin::Pin, task::Poll};

use futures_util::Future;

use super::{GPUContext, GPUFuture, GPUTaskContext};

pub struct JoinedGPUFuture<F1: GPUFuture, F2: GPUFuture> {
    f1: GPUFutureState<F1>,
    f2: GPUFutureState<F2>,
}
enum GPUFutureState<F: GPUFuture> {
    Ready {
        output: F::Output,
        queue: Option<u32>,
    },
    Pending(F),
    None,
}
impl<F1: GPUFuture, F2: GPUFuture> JoinedGPUFuture<F1, F2> {
    pub fn new(f1: F1, f2: F2) -> Self {
        Self {
            f1: GPUFutureState::Pending(f1),
            f2: GPUFutureState::Pending(f2),
        }
    }
}
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
                (GPUFutureState::Ready { .. }, GPUFutureState::Ready { .. }) => {
                    panic!("Continued to call pull after finished")
                }
                (GPUFutureState::Ready { .. }, GPUFutureState::Pending(f2)) => {
                    let f2 = Pin::new_unchecked(f2);
                    match f2.poll(cx) {
                        Poll::Ready(output) => {
                            let cx = cx.get();
                            this.f2 = GPUFutureState::Ready {
                                output,
                                queue: cx.current_queue,
                            };
                        }
                        _ => (),
                    }
                }
                (GPUFutureState::Pending(f1), GPUFutureState::Ready { .. }) => {
                    let f1 = Pin::new_unchecked(f1);
                    match f1.poll(cx) {
                        Poll::Ready(output) => {
                            let cx = cx.get();
                            this.f1 = GPUFutureState::Ready {
                                output,
                                queue: cx.current_queue,
                            };
                        }
                        _ => (),
                    }
                }
                (GPUFutureState::Pending(f1), GPUFutureState::Pending(f2)) => {
                    if f1.priority() > f2.priority() {
                        let f1 = Pin::new_unchecked(f1);
                        match f1.poll(cx) {
                            Poll::Ready(output) => {
                                let cx = cx.get();
                                this.f1 = GPUFutureState::Ready {
                                    output,
                                    queue: cx.current_queue,
                                };
                            }
                            _ => (),
                        }
                    } else {
                        let f2 = Pin::new_unchecked(f2);
                        match f2.poll(cx) {
                            Poll::Ready(output) => {
                                let cx = cx.get();
                                this.f2 = GPUFutureState::Ready {
                                    output,
                                    queue: cx.current_queue,
                                };
                            }
                            _ => (),
                        }
                    }
                }
                _ => unreachable!(),
            }

            match (&mut this.f1, &mut this.f2) {
                (GPUFutureState::Ready { .. }, GPUFutureState::Ready { .. }) => {
                    let f1 = std::mem::replace(&mut this.f1, GPUFutureState::None);
                    let f2 = std::mem::replace(&mut this.f2, GPUFutureState::None);
                    match (f1, f2) {
                        (
                            GPUFutureState::Ready {
                                output: f1,
                                queue: f1_queue,
                            },
                            GPUFutureState::Ready {
                                output: f2,
                                queue: f2_queue,
                            },
                        ) => {
                            let cx = cx.get();
                            cx.current_queue = match (f1_queue, f2_queue) {
                                (Some(f1_queue), Some(f2_queue)) if f1_queue == f2_queue => {
                                    Some(f1_queue)
                                }
                                _ => None,
                            };
                            return Poll::Ready((f1, f2));
                        }
                        _ => unreachable!(),
                    }
                }
                _ => {
                    let cx = cx.get();
                    cx.current_priority = this.priority();
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
            (GPUFutureState::Ready { .. }, GPUFutureState::Pending(f2)) => f2.priority(),
            (GPUFutureState::Pending(f1), GPUFutureState::Ready { .. }) => f1.priority(),
            _ => unreachable!(),
        }
    }
}


pub struct GPUFutureBlock<F: Future> {
    priority: u64,
    queue_index: Option<u32>,
    f: F,
}

impl<F: Future> GPUFutureBlock<F> {
    pub fn new(f: F) -> Self {
        Self {
            f,
            // Initial priority is always max.
            // This ensures that the initial hook would always get triggered first,
            // which then sets the priority correctly based on the head future.
            priority: u64::MAX,
            queue_index: None,
        }
    }
}

impl<F: Future> Future for GPUFutureBlock<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        unsafe {
            let gpu_cx = &mut *(cx.waker().as_raw().data() as *mut GPUContext);

            let this = self.get_unchecked_mut();
            let f = Pin::new_unchecked(&mut this.f);

            gpu_cx.current_priority = this.priority;
            gpu_cx.current_queue = this.queue_index;
            let result = f.poll(cx);

            let gpu_cx = &mut *(cx.waker().as_raw().data() as *mut GPUContext);
            this.priority = gpu_cx.current_priority;
            this.queue_index = gpu_cx.current_queue;
            result
        }
    }
}
impl<F: Future> GPUFuture for GPUFutureBlock<F> {
    fn priority(&self) -> u64 {
        self.priority
    }
}
