use std::{cell::RefCell, marker::PhantomData, ops::Generator, pin::Pin, sync::Arc, task::Poll};

use ash::vk;
use futures_util::future::OptionFuture;
use pin_project::pin_project;

use crate::{
    future::{CommandBufferRecordContext, GPUCommandFuture, StageContext, PerFrameState},
    Device, TimelineSemaphore, commands::SharedCommandPool,
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct QueueRef(pub u8);
impl QueueRef {
    pub fn null() -> Self {
        QueueRef(u8::MAX)
    }
    pub fn is_null(&self) -> bool {
        self.0 == u8::MAX
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct QueueMask(u64);
impl QueueMask {
    pub fn set_queue(&mut self, queue: QueueRef) {
        self.0 |= 1 << queue.0;
    }
    pub fn clear_queue(&mut self, queue: QueueRef) {
        self.0 &= !(1 << queue.0);
    }
    pub fn iter(&self) -> QueueMaskIterator {
        QueueMaskIterator(self.0)
    }
    pub fn empty() -> Self {
        Self(0)
    }
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }
    pub fn merge_with(&mut self, other: Self) {
        self.0 |= other.0
    }
    pub fn merge(&self, other: &Self) -> Self {
        Self(self.0 | other.0)
    }
}
pub struct QueueMaskIterator(u64);
impl Iterator for QueueMaskIterator {
    type Item = QueueRef;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            return None;
        }
        let t = self.0 & self.0.overflowing_neg().0;
        let r = self.0.trailing_zeros();
        self.0 ^= t;
        Some(QueueRef(r as u8))
    }
}

/// One for each submission.
pub struct SubmissionContext<'a> {
    // Indexed by queue family.s
    shared_command_pools: &'a mut [Option<SharedCommandPool>],
    queues: Vec<QueueSubmissionContext>,
}

impl<'a> SubmissionContext<'a> {
    pub fn of_queue_mut(&mut self, queue: QueueRef) -> &mut QueueSubmissionContext {
        &mut self.queues[queue.0 as usize]
    }
}

/// One per queue per submission
pub struct QueueSubmissionContext {
    queue_family_index: u32,
    stage_index: u32,
    timeline_index: u64,
    timeline_semaphore: TimelineSemaphore,
    dependencies: QueueMask,
    command_buffers: Vec<vk::CommandBuffer>,
    recording_command_buffer: Option<vk::CommandBuffer>
}
impl QueueSubmissionContext {
    pub fn depends_on_queues(&mut self, queue_mask: QueueMask) {
        self.dependencies.merge_with(queue_mask)
    }
}

/// Queues returned by the device.
pub struct Queues {
    device: Arc<Device>,

    /// Queues, and their queue family index.
    queues: Vec<(vk::Queue, u32)>,

    /// Mapping from queue families to queue refs
    families: Vec<QueueMask>
}

impl Queues {
    /// This function can only be called once per device.
    pub(crate) unsafe fn new(device: &Arc<Device>, num_queue_family: u32, queue_create_infos: &[vk::DeviceQueueCreateInfo]) -> Self {
        let mut families = vec![QueueMask::empty(); num_queue_family as usize];
        
        let mut queues: Vec<(vk::Queue, u32)> = Vec::new();
        
        for info in queue_create_infos.iter() {
            for i in 0..info.queue_count {
                families[info.queue_family_index as usize].set_queue(QueueRef(queues.len() as u8));
                queues.push(
                    (device.get_device_queue(info.queue_family_index, info.queue_family_index), info.queue_family_index));
            }
        }
        

        Self {
            device: device.clone(),
            queues,
            families
        }
    }
    pub fn make_shared_command_pools(&self) -> Vec<Option<SharedCommandPool>> {
        self.families.iter().enumerate().map(|(queue_family_index, queues)| {
            if queues.is_empty() {
                None
            } else {
                Some(SharedCommandPool::new(self.device.clone(), queue_family_index as u32))
            }
        }).collect()
    }
    pub fn submit<F: QueueFuture>(
        &mut self,
        future: F,
        shared_command_pools: &mut [Option<SharedCommandPool>],
        recycled_state: &mut F::RecycledState
    ) -> impl std::future::Future<Output = F::Output> {
        let mut future = std::pin::pin!(future);
        let mut submission_context = SubmissionContext {
            shared_command_pools,
            queues: self.queues.iter().map(|(queue, queue_family_index)| QueueSubmissionContext {
                queue_family_index: *queue_family_index,
                stage_index: 0,
                timeline_index: 0,
                timeline_semaphore: TimelineSemaphore::new(self.device.clone(), 0).unwrap(),
                dependencies: QueueMask::empty(),
                command_buffers: Vec::new(),
                recording_command_buffer: None,
            }).collect(),
        };

        let mut submissions: Vec<Vec<vk::SubmitInfo2>> = vec![Vec::new(); self.queues.len()];
        let output = loop {
            match future.as_mut().record(&mut submission_context, recycled_state) {
                QueueFuturePoll::Barrier => {
                    continue;
                }
                QueueFuturePoll::Semaphore => {
                    for (i, ctx) in submission_context.queues.iter().enumerate() {
                        Self::submit_for_queue(ctx, &submission_context, &mut submissions[i]);
                    }
                    for ctx in submission_context.queues.iter_mut() {
                        ctx.stage_index = 0;
                        ctx.timeline_index += 1;
                    }
                }
                QueueFuturePoll::Ready { next_queue, output } => {
                    break output;
                }
            }
        };

        // actually submit
        for (i, queue) in self.queues.iter().enumerate() {
            let queue_submissions = submissions[i].as_slice();
            unsafe {
                /*
                self.device.queue_submit2(
                    *queue,
                    queue_submissions,
                    vk::Fence::null()
                ).unwrap();
                */
                println!("Submit");
                for submission in queue_submissions.iter() {
                    Self::submit_for_queue_cleanup(submission);
                }
            }
        }
        let fut_dispose = future.dispose();
        async {
            fut_dispose.await;
            output
        }
    }
    fn submit_for_queue(
        queue_ctx: &QueueSubmissionContext,
        ctx: &SubmissionContext,
        submissions: &mut Vec<vk::SubmitInfo2>,
    ) {
        if queue_ctx.stage_index == 0 {
            // Nothing was submitted for this queue.
            return;
        }

        // Also need to wait on other queues.
        let wait_semaphore_infos = queue_ctx
            .dependencies
            .iter()
            .map(|queue| {
                let dep = &ctx.queues[queue.0 as usize];
                assert!(dep.timeline_index > 0);
                vk::SemaphoreSubmitInfo {
                    semaphore: dep.timeline_semaphore.semaphore,
                    value: dep.timeline_index,
                    stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS, // TODO: Optimization
                    device_index: 0,
                    ..Default::default()
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let signal_semaphore_infos = Box::new(vk::SemaphoreSubmitInfo {
            semaphore: queue_ctx.timeline_semaphore.semaphore,
            value: queue_ctx.timeline_index + 1,
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS, // TODO: Optimization
            device_index: 0,
            ..Default::default()
        });
        
        let command_buffer_infos = queue_ctx.command_buffers.iter()
        .map(|buf| vk::CommandBufferSubmitInfo {
            command_buffer: *buf,
            ..Default::default()
        }).collect::<Vec<_>>().into_boxed_slice();

        // Now, create submission.
        let submission = vk::SubmitInfo2 {
            flags: vk::SubmitFlags::empty(),
            wait_semaphore_info_count: 1,
            p_wait_semaphore_infos: wait_semaphore_infos.as_ptr(),
            command_buffer_info_count: command_buffer_infos.len() as u32,
            p_command_buffer_infos: command_buffer_infos.as_ptr(),
            signal_semaphore_info_count: 1,
            p_signal_semaphore_infos: signal_semaphore_infos.as_ref(),
            ..Default::default()
        };
        submissions.push(submission);
        std::mem::forget(wait_semaphore_infos);
        std::mem::forget(signal_semaphore_infos);
        std::mem::forget(command_buffer_infos);
    }
    unsafe fn submit_for_queue_cleanup(submission: &vk::SubmitInfo2) {
        let wait_semaphore_infos: Box<[vk::SemaphoreSubmitInfo]> =
            Box::from_raw(std::slice::from_raw_parts_mut(
                submission.p_wait_semaphore_infos as *mut _,
                submission.wait_semaphore_info_count as usize,
            ));
        assert_eq!(submission.signal_semaphore_info_count, 1);
        let signal_semaphore_infos: Box<vk::SemaphoreSubmitInfo> =
            Box::from_raw(submission.p_signal_semaphore_infos as *mut _);
        let command_buffer_infos: Box<[vk::CommandBufferSubmitInfo]> =
            Box::from_raw(std::slice::from_raw_parts_mut(
                submission.p_command_buffer_infos as *mut _,
                submission.command_buffer_info_count as usize,
            ));
        drop(wait_semaphore_infos);
        drop(signal_semaphore_infos);
        drop(command_buffer_infos);
    }
}

pub enum QueueFuturePoll<OUT> {
    Barrier,
    Semaphore,
    Ready {
        next_queue: QueueMask, // Should really be "queues used"
        output: OUT,
    },
}
pub trait QueueFuture {
    type Output;
    type RecycledState: Default;
    fn init(self: Pin<&mut Self>, ctx: &mut SubmissionContext, recycled_state: &mut Self::RecycledState, prev_queue: QueueMask);
    /// Record all command buffers for the specified queue_index, up to the specified timeline index.
    /// The executor calls record with increasing `timeline` value, and wrap them in vk::SubmitInfo2.
    /// queue should be the queue of the parent node, or None if multiple parents with different queues.
    /// queue should be None on subsequent calls.
    fn record(self: Pin<&mut Self>, ctx: &mut SubmissionContext, recycled_state: &mut Self::RecycledState) -> QueueFuturePoll<Self::Output>;

    /// Runs when the future is ready to be disposed.s
    fn dispose(self: Pin<&mut Self>) -> impl std::future::Future<Output = ()>;
}

/// On yield: true for a hard sync point (semaphore)
pub trait QueueFutureBlockGenerator<R, Fut, RecycledState> =
    Generator<(QueueMask, *mut (), *mut RecycledState), Return = (QueueMask, Fut, R), Yield = bool>;

#[pin_project]
pub struct QueueFutureBlock<Ret, Inner, Fut, Recycle>
where
    Inner: QueueFutureBlockGenerator<Ret, Fut, Recycle>,
    Fut: std::future::Future<Output = ()>,
{
    #[pin]
    inner: Inner,
    dispose: Option<Fut>,

    /// This is more of a hack to save the dependent queue mask when `init` was called
    /// so we can initialize `__current_queue_mask` with this value.
    initial_queue_mask: QueueMask,
    _marker: PhantomData<fn(&mut Recycle) -> Ret>,
}
impl<Ret, Inner, Fut, Recycle> QueueFutureBlock<Ret, Inner, Fut, Recycle>
where
    Inner: QueueFutureBlockGenerator<Ret, Fut, Recycle>,
    Fut: std::future::Future<Output = ()>,
{
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            initial_queue_mask: QueueMask::empty(),
            dispose: None,
            _marker: PhantomData,
        }
    }
}
impl<Ret, Inner, Fut, Recycle: Default> QueueFuture for QueueFutureBlock<Ret, Inner, Fut, Recycle>
where
    Inner: QueueFutureBlockGenerator<Ret, Fut, Recycle>,
    Fut: std::future::Future<Output = ()>,
{
    type Output = Ret;
    type RecycledState = Recycle;
    fn init(self: Pin<&mut Self>, ctx: &mut SubmissionContext, recycled_state: &mut Self::RecycledState, prev_queue: QueueMask) {
        *self.project().initial_queue_mask = prev_queue;
    }
    fn record(self: Pin<&mut Self>, ctx: &mut SubmissionContext, recycled_state: &mut Self::RecycledState) -> QueueFuturePoll<Ret> {
        let this = self.project();
        assert!(
            this.dispose.is_none(),
            "Calling record after returning Complete"
        );
        match this.inner.resume((*this.initial_queue_mask, ctx as *mut _ as *mut (), recycled_state)) {
            std::ops::GeneratorState::Yielded(is_semaphore) if is_semaphore => {
                QueueFuturePoll::Semaphore
            }
            std::ops::GeneratorState::Yielded(is_semaphore) => QueueFuturePoll::Barrier,
            std::ops::GeneratorState::Complete((next_queue, dispose, output)) => {
                *this.dispose = Some(dispose);
                QueueFuturePoll::Ready { next_queue, output }
            }
        }
    }
    fn dispose(self: Pin<&mut Self>) -> impl std::future::Future<Output = ()> {
        let this = self.project();
        let dispose = this
            .dispose
            .take()
            .expect("Dispose can only be called once after recording is finished");
        dispose
    }
}

#[pin_project]
pub struct QueueFutureJoin<I1: QueueFuture, I2: QueueFuture> {
    #[pin]
    inner1: I1,
    inner1_result: QueueFuturePoll<I1::Output>,
    #[pin]
    inner2: I2,
    inner2_result: QueueFuturePoll<I2::Output>,
    results_taken: bool,
}

impl<I1: QueueFuture, I2: QueueFuture> QueueFutureJoin<I1, I2> {
    pub fn new(inner1: I1, inner2: I2) -> Self {
        Self {
            inner1,
            inner1_result: QueueFuturePoll::Barrier,
            inner2,
            inner2_result: QueueFuturePoll::Barrier,
            results_taken: false,
        }
    }
}

impl<I1: QueueFuture, I2: QueueFuture> QueueFuture for QueueFutureJoin<I1, I2> {
    type Output = (I1::Output, I2::Output);
    type RecycledState = (I1::RecycledState, I2::RecycledState);
    fn init(self: Pin<&mut Self>, ctx: &mut SubmissionContext, recycled_state: &mut Self::RecycledState, prev_queue: QueueMask) {
        let this = self.project();
        this.inner1.init(ctx, &mut recycled_state.0, prev_queue);
        this.inner2.init(ctx,  &mut recycled_state.1, prev_queue);
    }

    fn record(self: Pin<&mut Self>, ctx: &mut SubmissionContext, recycled_state: &mut Self::RecycledState) -> QueueFuturePoll<Self::Output> {
        let this = self.project();
        assert!(
            !*this.results_taken,
            "Attempted to record a QueueFutureJoin after it's finished"
        );
        match (&this.inner1_result, &this.inner2_result) {
            (QueueFuturePoll::Barrier, QueueFuturePoll::Barrier)
            | (QueueFuturePoll::Semaphore, QueueFuturePoll::Semaphore) => {
                *this.inner1_result = this.inner1.record(ctx, &mut recycled_state.0);
                *this.inner2_result = this.inner2.record(ctx, &mut recycled_state.1);
            }
            (QueueFuturePoll::Barrier, QueueFuturePoll::Semaphore)
            | (QueueFuturePoll::Barrier, QueueFuturePoll::Ready { .. })
            | (QueueFuturePoll::Semaphore, QueueFuturePoll::Ready { .. }) => {
                *this.inner1_result = this.inner1.record(ctx, &mut recycled_state.0);
            }
            (QueueFuturePoll::Semaphore, QueueFuturePoll::Barrier)
            | (QueueFuturePoll::Ready { .. }, QueueFuturePoll::Barrier)
            | (QueueFuturePoll::Ready { .. }, QueueFuturePoll::Semaphore) => {
                *this.inner2_result = this.inner2.record(ctx, &mut recycled_state.1);
            }
            (QueueFuturePoll::Ready { .. }, QueueFuturePoll::Ready { .. }) => {
                unreachable!();
            }
        }
        match (&this.inner1_result, &this.inner2_result) {
            (QueueFuturePoll::Barrier, QueueFuturePoll::Barrier)
            | (QueueFuturePoll::Barrier, QueueFuturePoll::Semaphore)
            | (QueueFuturePoll::Barrier, QueueFuturePoll::Ready { .. })
            | (QueueFuturePoll::Semaphore, QueueFuturePoll::Barrier)
            | (QueueFuturePoll::Ready { .. }, QueueFuturePoll::Barrier) => QueueFuturePoll::Barrier,
            (QueueFuturePoll::Semaphore, QueueFuturePoll::Semaphore)
            | (QueueFuturePoll::Ready { .. }, QueueFuturePoll::Semaphore)
            | (QueueFuturePoll::Semaphore, QueueFuturePoll::Ready { .. }) => {
                QueueFuturePoll::Semaphore
            }
            (QueueFuturePoll::Ready { .. }, QueueFuturePoll::Ready { .. }) => {
                let (r1_queue_mask, r1_ret) =
                    match std::mem::replace(this.inner1_result, QueueFuturePoll::Barrier) {
                        QueueFuturePoll::Ready {
                            next_queue: r1_queue,
                            output: r1_ret,
                        } => (r1_queue, r1_ret),
                        _ => unreachable!(),
                    };
                let (r2_queue_mask, r2_ret) =
                    match std::mem::replace(this.inner2_result, QueueFuturePoll::Barrier) {
                        QueueFuturePoll::Ready {
                            next_queue: r2_queue,
                            output: r2_ret,
                        } => (r2_queue, r2_ret),
                        _ => unreachable!(),
                    };
                *this.results_taken = true;
                return QueueFuturePoll::Ready {
                    next_queue: r1_queue_mask.merge(&r2_queue_mask),
                    output: (r1_ret, r2_ret),
                };
            }
        }
    }

    fn dispose(self: Pin<&mut Self>) -> impl std::future::Future<Output = ()> {
        use futures_util::FutureExt;
        let this = self.project();
        futures_util::future::join(this.inner1.dispose(), this.inner2.dispose()).map(|_| ())
    }
}


#[pin_project]
pub struct RunCommandsQueueFuture<I: GPUCommandFuture> {
    #[pin]
    inner: I,

    /// Specified queue to run on.
    queue: QueueRef, // If null, use the previous queue.
    /// Retained state and the timeline index
    retained_state: Option<(I::RetainedState, u64)>,
    prev_queue: QueueMask,
}
impl<I: GPUCommandFuture> RunCommandsQueueFuture<I> {
    pub fn new(future: I, queue: QueueRef) -> Self {
        Self {
            inner: future,
            queue,
            retained_state: None,
            prev_queue: QueueMask::empty()
        }
    }
}
impl<I: GPUCommandFuture> QueueFuture for RunCommandsQueueFuture<I> {
    type Output = I::Output;
    type RecycledState = I::RecycledState;

    fn init(self: Pin<&mut Self>, ctx: &mut SubmissionContext, recycled_state: &mut Self::RecycledState, prev_queue: QueueMask) {
        let this = self.project();
        if this.queue.is_null() {
            let mut iter = prev_queue.iter();
            if let Some(inherited_queue) = prev_queue.iter().next() {
                *this.queue = inherited_queue;
                assert!(
                    iter.next().is_none(),
                    "Cannot use derived queue when the future depends on more than one queues"
                );
            } else {
                // Default to the first queue, if the queue does not have predecessor.
                *this.queue = QueueRef(0);
            }

        }
        *this.prev_queue = prev_queue;
        let r = &mut ctx.queues[this.queue.0 as usize];
        r.dependencies.merge_with(prev_queue);
        let mut command_ctx = CommandBufferRecordContext {
            stage_index: r.stage_index,
            command_buffers: &mut r.command_buffers,
            command_pool: ctx.shared_command_pools[r.queue_family_index as usize].as_mut().unwrap(),
            recording_command_buffer: &mut r.recording_command_buffer,
        };
        this.inner.init(&mut command_ctx, recycled_state);
    }

    fn record(self: Pin<&mut Self>, ctx: &mut SubmissionContext, recycled_state: &mut Self::RecycledState) -> QueueFuturePoll<Self::Output> {
        let this = self.project();

        let queue = {
            let mut mask = QueueMask::empty();
            mask.set_queue(*this.queue);
            mask
        };

        if !this.prev_queue.is_empty() && *this.prev_queue != queue {
            *this.prev_queue = queue;
            return QueueFuturePoll::Semaphore;
        }

        let r = &mut ctx.queues[this.queue.0 as usize];
        let mut command_ctx = CommandBufferRecordContext {
            stage_index: r.stage_index,
            command_buffers: &mut r.command_buffers,
            command_pool: ctx.shared_command_pools[r.queue_family_index as usize].as_mut().unwrap(),
            recording_command_buffer: &mut r.recording_command_buffer,
        };

        let result = match command_ctx.record_one_step(this.inner, recycled_state) {
            Poll::Ready((output, retained_state)) => {
                *this.retained_state = Some((retained_state, r.timeline_index));
                QueueFuturePoll::Ready {
                    next_queue: {
                        let mut mask = QueueMask::empty();
                        mask.set_queue(*this.queue);
                        mask
                    },
                    output,
                }
            }
            Poll::Pending => QueueFuturePoll::Barrier,
        };
        r.stage_index += 1;
        result
    }

    fn dispose(self: Pin<&mut Self>) -> impl std::future::Future<Output = ()> {
        let this = self.project();
        let (retained_state, timelie_index) = this
            .retained_state
            .take()
            .expect("Dispose should be called after recording finish");
        async move {
            // await timeline
            println!("Await timeline: {}", timelie_index);
            drop(retained_state);
        }
    }
}
