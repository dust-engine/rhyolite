use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    fmt::{Debug, Write},
    marker::PhantomData,
    ops::Generator,
    pin::Pin,
    sync::Arc,
    task::Poll,
};

use ash::vk;
use futures_util::future::OptionFuture;
use pin_project::pin_project;

use crate::{
    commands::SharedCommandPool,
    future::{CommandBufferRecordContext, GPUCommandFuture, PerFrameState, StageContext},
    Device, TimelineSemaphore,
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
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
impl std::fmt::Debug for QueueMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
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
    // Indexed by queue family.
    shared_command_pools: &'a mut [Option<SharedCommandPool>],
    // Indexed by queue id.
    queues: Vec<QueueSubmissionContext>,
    // Indexed by queue id.
    submission: Vec<QueueSubmissionType>,
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
    timeline_index: u32,
    timeline_semaphore: TimelineSemaphore,

    /// Stages that the previous stage of the current queue needs to signal on.
    signals: BTreeSet<vk::PipelineStageFlags2>,

    /// Stages that the current stage needs to wait on.
    waits: Vec<(vk::PipelineStageFlags2, QueueRef, vk::PipelineStageFlags2)>, // the last u32 indexes into the first signals array.
}

#[derive(Debug)]
enum QueueSubmissionType {
    Submit {
        command_buffers: Vec<vk::CommandBuffer>,
        recording_command_buffer: Option<vk::CommandBuffer>,
    },
    SparseBind {
        memory_binds: Vec<vk::SparseBufferMemoryBindInfo>,
        image_opaque_binds: Vec<vk::SparseImageOpaqueMemoryBindInfo>,
        image_binds: Vec<vk::SparseImageMemoryBindInfo>,
    },
    Acquire,
    Present,
    Unknown,
}
impl Default for QueueSubmissionType {
    fn default() -> Self {
        Self::Unknown
    }
}
impl QueueSubmissionType {
    fn end(&mut self) {
        match self {
            QueueSubmissionType::Submit {
                command_buffers,
                recording_command_buffer,
            } => {
                if let Some(cmd_buf) = recording_command_buffer.take() {
                    command_buffers.push(cmd_buf);
                }
            }
            _ => {}
        }
    }
}
/// Queues returned by the device.
pub struct Queues {
    device: Arc<Device>,

    /// Queues, and their queue family index.
    queues: Vec<(vk::Queue, u32)>,

    /// Mapping from queue families to queue refs
    families: Vec<QueueMask>,
}

impl Queues {
    /// This function can only be called once per device.
    pub(crate) unsafe fn new(
        device: &Arc<Device>,
        num_queue_family: u32,
        queue_create_infos: &[vk::DeviceQueueCreateInfo],
    ) -> Self {
        let mut families = vec![QueueMask::empty(); num_queue_family as usize];

        let mut queues: Vec<(vk::Queue, u32)> = Vec::new();

        for info in queue_create_infos.iter() {
            for i in 0..info.queue_count {
                families[info.queue_family_index as usize].set_queue(QueueRef(queues.len() as u8));
                queues.push((
                    device.get_device_queue(info.queue_family_index, i),
                    info.queue_family_index,
                ));
            }
        }

        Self {
            device: device.clone(),
            queues,
            families,
        }
    }
    pub fn make_shared_command_pools(&self) -> Vec<Option<SharedCommandPool>> {
        self.families
            .iter()
            .enumerate()
            .map(|(queue_family_index, queues)| {
                if queues.is_empty() {
                    None
                } else {
                    Some(SharedCommandPool::new(
                        self.device.clone(),
                        queue_family_index as u32,
                    ))
                }
            })
            .collect()
    }
    pub fn submit<F: QueueFuture>(
        &mut self,
        mut future: F,
        // These two pools are passed in as argument so that they can be cleaned on a regular basis (per frame) externally
        shared_command_pools: &mut [Option<SharedCommandPool>],
        semaphore_pool: &mut TimelineSemaphorePool,
        fence_pool: &mut FencePool,
        recycled_state: &mut F::RecycledState,
    ) -> impl std::future::Future<Output = F::Output> {
        let mut future_pinned = unsafe { std::pin::Pin::new_unchecked(&mut future) };
        let mut submission_context = SubmissionContext {
            shared_command_pools,
            queues: self
                .queues
                .iter()
                .map(|(queue, queue_family_index)| QueueSubmissionContext {
                    queue_family_index: *queue_family_index,
                    stage_index: 0,
                    timeline_index: 0,
                    timeline_semaphore: TimelineSemaphore::new(self.device.clone(), 0).unwrap(),
                    signals: Default::default(),
                    waits: Default::default(),
                })
                .collect(),
            submission: self
                .queues
                .iter()
                .map(|_| QueueSubmissionType::Unknown)
                .collect(),
        };

        let mut current_stage = CachedStageSubmissions::new(self.queues.len());
        let mut submission_batch = SubmissionBatch::new(self.queues.len());

        future_pinned
            .as_mut()
            .init(&mut submission_context, recycled_state, QueueMask::empty());
        let output = loop {
            match future_pinned
                .as_mut()
                .record(&mut submission_context, recycled_state)
            {
                QueueFuturePoll::Barrier => {
                    println!("Saw barrier");
                    continue;
                }
                QueueFuturePoll::Semaphore => {
                    println!("Saw semaphore");
                    let mut last_stage = std::mem::replace(
                        &mut current_stage,
                        CachedStageSubmissions::new(self.queues.len()),
                    );
                    last_stage.apply_signals(&submission_context, semaphore_pool);
                    current_stage.apply_submissions(&submission_context, &last_stage);

                    for (src, dst) in submission_context
                        .submission
                        .iter_mut()
                        .zip(last_stage.queues.iter_mut())
                    {
                        src.end();
                        dst.ty = std::mem::replace(src, QueueSubmissionType::Unknown);
                    }

                    submission_batch.add_stage(last_stage);
                    for ctx in submission_context.queues.iter_mut() {
                        ctx.stage_index = 0;
                        ctx.timeline_index += 1;
                    }
                }
                QueueFuturePoll::Ready { next_queue, output } => {
                    println!("Saw ready");
                    let mut last_stage = std::mem::replace(
                        &mut current_stage,
                        CachedStageSubmissions::new(self.queues.len()),
                    );
                    last_stage.apply_signals(&submission_context, semaphore_pool);

                    for (src, dst) in submission_context
                        .submission
                        .iter_mut()
                        .zip(last_stage.queues.iter_mut())
                    {
                        src.end();
                        dst.ty = std::mem::replace(src, QueueSubmissionType::Unknown);
                    }
                    submission_batch.add_stage(last_stage);
                    break output;
                }
            }
        };

        let fences_to_wait = self.submit_batch(submission_batch, fence_pool);

        drop(future_pinned); // No more touching of future! It's getting moved.
        let fut_dispose = future.dispose();

        let device = self.device.clone();
        async {
            blocking::unblock(move || unsafe {
                device.wait_for_fences(&fences_to_wait, true, !0);
            })
            .await;
            drop(fut_dispose);
            output
        }
    }
    fn submit_batch(
        &mut self,
        batch: SubmissionBatch,
        fence_pool: &mut FencePool,
    ) -> Vec<vk::Fence> {
        println!("{:?}", batch);
        let fences: Vec<vk::Fence> = Vec::new();
        for (queue, queue_batch) in self
            .queues
            .iter()
            .map(|(queue, _)| *queue)
            .zip(batch.queues.iter())
        {
            unsafe {
                if !queue_batch.submits.is_empty() {
                    let fence = fence_pool.get();
                    self.device
                        .queue_submit2(queue, &queue_batch.submits, fence)
                        .unwrap();
                }
                if !queue_batch.sparse_binds.is_empty() {
                    self.device
                        .queue_bind_sparse(queue, &queue_batch.sparse_binds, vk::Fence::null())
                        .unwrap();
                }
            }
        }
        fences
    }
}

pub struct TimelineSemaphorePool {
    device: Arc<Device>,
    semaphore_ops: Vec<(vk::Semaphore, u64)>,
}
impl TimelineSemaphorePool {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            semaphore_ops: Vec::new(),
        }
    }
    fn signal(&mut self) -> (vk::Semaphore, u64) {
        let (semaphore, timeline) = self.semaphore_ops.pop().unwrap_or_else(|| {
            let mut type_info = vk::SemaphoreTypeCreateInfo::builder()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0)
                .build();
            let create_info = vk::SemaphoreCreateInfo::builder()
                .push_next(&mut type_info)
                .build();
            let semaphore = unsafe { self.device.create_semaphore(&create_info, None).unwrap() };
            (semaphore, 0)
        });
        (semaphore, timeline + 1)
    }
    fn waited(&mut self, semaphore: vk::Semaphore, timeline: u64) {
        self.semaphore_ops.push((semaphore, timeline));
    }
}

pub struct FencePool {
    device: Arc<Device>,
    all_fences: Vec<vk::Fence>,
    indice: usize,
}
impl FencePool {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            all_fences: Vec::new(),
            indice: 0,
        }
    }
    fn get(&mut self) -> vk::Fence {
        if self.indice <= self.all_fences.len() {
            let fence = unsafe {
                self.device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .unwrap()
            };
            self.all_fences.push(fence);
        }
        let fence = self.all_fences[self.indice];
        self.indice += 1;
        fence
    }
    fn reset(&mut self) {
        if self.indice > 0 {
            unsafe {
                self.device
                    .reset_fences(&self.all_fences[0..self.indice])
                    .unwrap()
            }
        }
        self.indice = 0;
    }
}

/// Represents one queue of one stage of submissions
#[derive(Default)]
struct CachedQueueStageSubmissions {
    // Indexed
    ty: QueueSubmissionType,
    waits: Vec<(vk::Semaphore, u64, vk::PipelineStageFlags2)>,
    signals: BTreeMap<vk::PipelineStageFlags2, (vk::Semaphore, u64)>,
}
/// Represents one stage of submissions
struct CachedStageSubmissions {
    // Indexed by QueueId
    queues: Vec<CachedQueueStageSubmissions>,
}
impl CachedStageSubmissions {
    pub fn new(num_queues: usize) -> Self {
        Self {
            queues: (0..num_queues)
                .map(|_| CachedQueueStageSubmissions::default())
                .collect(),
        }
    }
    /// Called on the previous stage.
    fn apply_signals(
        &mut self,
        submission_context: &SubmissionContext,
        semaphore_pool: &mut TimelineSemaphorePool,
    ) {
        for (i, (ctx, cache)) in submission_context
            .queues
            .iter()
            .zip(self.queues.iter_mut())
            .enumerate()
        {
            cache.signals.extend(ctx.signals.iter().map(|stage| {
                let (semaphore, value) = semaphore_pool.signal();
                (*stage, (semaphore, value))
            }));
        }
    }
    /// Called on the current stage.
    fn apply_submissions(&mut self, submission_context: &SubmissionContext, prev_stage: &Self) {
        for (i, (ctx, cache)) in submission_context
            .queues
            .iter()
            .zip(self.queues.iter_mut())
            .enumerate()
        {
            cache
                .waits
                .extend(ctx.waits.iter().map(|(dst_stage, src_queue, src_stage)| {
                    let (semaphore, value) = prev_stage.queues[src_queue.0 as usize]
                        .signals
                        .get(src_stage)
                        .unwrap();
                    (*semaphore, *value, *dst_stage)
                }));
        }
    }
}

#[derive(Default)]
pub struct QueueSubmissionBatch {
    submits: Vec<vk::SubmitInfo2>,
    sparse_binds: Vec<vk::BindSparseInfo>,
}
pub struct SubmissionBatch {
    queues: Vec<QueueSubmissionBatch>,
}
impl Debug for SubmissionBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SubmissionBatch {")?;

        for (i, queue) in self.queues.iter().enumerate() {
            if queue.submits.len() > 0 {
                f.write_char('\n')?;
                f.write_fmt(format_args!("  Queue {} Submits:", i))?;
                for submit in queue.submits.iter() {
                    f.write_char('\n');
                    f.write_fmt(format_args!(
                        "    Waits {} semaphores",
                        submit.wait_semaphore_info_count
                    ))?;
                    f.write_fmt(format_args!("    {} command buffers, ", i))?;
                    f.write_fmt(format_args!(
                        "    Signal {} semaphores",
                        submit.signal_semaphore_info_count
                    ))?;
                }
            }
        }
        f.write_str("\n}")?;
        Ok(())
    }
}
impl SubmissionBatch {
    pub fn new(num_queues: usize) -> Self {
        Self {
            queues: (0..num_queues).map(|_| Default::default()).collect(),
        }
    }
    fn add_stage(&mut self, stage: CachedStageSubmissions) {
        for (ctx, self_ctx) in stage.queues.into_iter().zip(self.queues.iter_mut()) {
            println!("{:?}", ctx.ty);
            match ctx.ty {
                QueueSubmissionType::Submit {
                    command_buffers,
                    recording_command_buffer,
                } => {
                    assert!(recording_command_buffer.is_none());
                    let waits = ctx
                        .waits
                        .into_iter()
                        .map(|(semaphore, value, stage_mask)| vk::SemaphoreSubmitInfo {
                            semaphore,
                            value,
                            stage_mask,
                            ..Default::default()
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    let signals = ctx
                        .signals
                        .into_iter()
                        .map(|(stage_mask, (semaphore, value))| vk::SemaphoreSubmitInfo {
                            semaphore,
                            value,
                            stage_mask,
                            ..Default::default()
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    let command_buffers = command_buffers
                        .into_iter()
                        .map(|buf| vk::CommandBufferSubmitInfo {
                            command_buffer: buf,
                            ..Default::default()
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    self_ctx.submits.push(
                        vk::SubmitInfo2::builder()
                            .command_buffer_infos(&command_buffers)
                            .wait_semaphore_infos(&waits)
                            .signal_semaphore_infos(&signals)
                            .build(),
                    );
                    std::mem::forget(waits);
                    std::mem::forget(signals);
                    std::mem::forget(command_buffers);
                }
                QueueSubmissionType::SparseBind {
                    memory_binds,
                    image_opaque_binds,
                    image_binds,
                } => {
                    for (_, _, stages) in ctx.waits.iter() {
                        assert!(stages.is_empty());
                    }
                    for (stages, _) in ctx.signals.iter() {
                        assert!(stages.is_empty());
                    }
                    let waits = ctx
                        .waits
                        .iter()
                        .map(|(semaphore, _, _)| *semaphore)
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    let signals = ctx
                        .signals
                        .iter()
                        .map(|(_, (semaphore, _))| *semaphore)
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    let wait_values = ctx
                        .waits
                        .iter()
                        .map(|(_, value, _)| *value)
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    let signal_values = ctx
                        .signals
                        .iter()
                        .map(|(_, (_, value))| *value)
                        .collect::<Vec<_>>()
                        .into_boxed_slice();

                    let memory_binds = memory_binds.into_boxed_slice();
                    let image_opaque_binds = image_opaque_binds.into_boxed_slice();
                    let image_binds = image_binds.into_boxed_slice();
                    let mut timeline_info = Box::new(
                        vk::TimelineSemaphoreSubmitInfo::builder()
                            .wait_semaphore_values(&wait_values)
                            .signal_semaphore_values(&signal_values)
                            .build(),
                    );
                    self_ctx.sparse_binds.push(
                        vk::BindSparseInfo::builder()
                            .buffer_binds(&memory_binds)
                            .image_binds(&image_binds)
                            .image_opaque_binds(&image_opaque_binds)
                            .wait_semaphores(&waits)
                            .signal_semaphores(&signals)
                            .push_next(timeline_info.as_mut())
                            .build(),
                    );

                    std::mem::forget(waits);
                    std::mem::forget(signals);
                    std::mem::forget(wait_values);
                    std::mem::forget(signal_values);
                    std::mem::forget(memory_binds);
                    std::mem::forget(image_opaque_binds);
                    std::mem::forget(image_binds);
                    std::mem::forget(timeline_info);
                }
                QueueSubmissionType::Acquire => todo!(),
                QueueSubmissionType::Present => todo!(),
                QueueSubmissionType::Unknown => (),
            }
        }
    }
}
impl Drop for SubmissionBatch {
    fn drop(&mut self) {
        unsafe {
            for queue in self.queues.iter() {
                for submit in &queue.submits {
                    let waits = Box::from_raw(std::slice::from_raw_parts_mut(
                        submit.p_wait_semaphore_infos as *mut vk::SemaphoreSubmitInfo,
                        submit.wait_semaphore_info_count as usize,
                    ));
                    let signals = Box::from_raw(std::slice::from_raw_parts_mut(
                        submit.p_signal_semaphore_infos as *mut vk::SemaphoreSubmitInfo,
                        submit.signal_semaphore_info_count as usize,
                    ));
                    let commands = Box::from_raw(std::slice::from_raw_parts_mut(
                        submit.p_command_buffer_infos as *mut vk::SemaphoreSubmitInfo,
                        submit.command_buffer_info_count as usize,
                    ));
                    drop(waits);
                    drop(signals);
                    drop(commands);
                }
                for sparse_bind in &queue.sparse_binds {
                    drop(Box::from_raw(std::slice::from_raw_parts_mut(
                        sparse_bind.p_wait_semaphores as *mut vk::Semaphore,
                        sparse_bind.wait_semaphore_count as usize,
                    )));
                    drop(Box::from_raw(std::slice::from_raw_parts_mut(
                        sparse_bind.signal_semaphore_count as *mut vk::SemaphoreSubmitInfo,
                        sparse_bind.signal_semaphore_count as usize,
                    )));
                    drop(Box::from_raw(std::slice::from_raw_parts_mut(
                        sparse_bind.p_buffer_binds as *mut vk::SemaphoreSubmitInfo,
                        sparse_bind.buffer_bind_count as usize,
                    )));
                    drop(Box::from_raw(std::slice::from_raw_parts_mut(
                        sparse_bind.p_image_binds as *mut vk::SemaphoreSubmitInfo,
                        sparse_bind.image_bind_count as usize,
                    )));
                    drop(Box::from_raw(std::slice::from_raw_parts_mut(
                        sparse_bind.p_image_opaque_binds as *mut vk::SemaphoreSubmitInfo,
                        sparse_bind.image_opaque_bind_count as usize,
                    )));
                    let timeline =
                        Box::from_raw(sparse_bind.p_next as *mut vk::TimelineSemaphoreSubmitInfo);
                    drop(Box::from_raw(std::slice::from_raw_parts_mut(
                        timeline.p_signal_semaphore_values as *mut vk::SemaphoreSubmitInfo,
                        timeline.signal_semaphore_value_count as usize,
                    )));
                    drop(Box::from_raw(std::slice::from_raw_parts_mut(
                        timeline.p_wait_semaphore_values as *mut vk::SemaphoreSubmitInfo,
                        timeline.wait_semaphore_value_count as usize,
                    )));
                    drop(timeline);
                }
            }
        }
    }
}

pub enum QueueFuturePoll<OUT> {
    Barrier,
    Semaphore,
    Ready { next_queue: QueueMask, output: OUT },
}

/// We don't know what are the semaphores to signal, until later stages tell us.
/// When future2 depends on future1, the following execution order occures:
/// - future1.init()
/// - let ctx = future1.context();
/// - future1.record().......
///
///
/// - future2.init()
/// - let ctx = future2.context();
/// - future1.end(ctx)
/// - future2.record().......
pub trait QueueFuture {
    type Output;
    type RecycledState: Default;
    type RetainedState;
    fn init(
        self: Pin<&mut Self>,
        ctx: &mut SubmissionContext,
        recycled_state: &mut Self::RecycledState,
        prev_queue: QueueMask,
    );
    /// Record all command buffers for the specified queue_index, up to the specified timeline index.
    /// The executor calls record with increasing `timeline` value, and wrap them in vk::SubmitInfo2.
    /// queue should be the queue of the parent node, or None if multiple parents with different queues.
    /// queue should be None on subsequent calls.
    fn record(
        self: Pin<&mut Self>,
        ctx: &mut SubmissionContext,
        recycled_state: &mut Self::RecycledState,
    ) -> QueueFuturePoll<Self::Output>;

    /// Runs when the future is ready to be disposed.s
    fn dispose(self) -> Self::RetainedState;
}

/// On yield: true for a hard sync point (semaphore)
pub trait QueueFutureBlockGenerator<Return, RecycledState, RetainedState> = Generator<
    (QueueMask, *mut (), *mut RecycledState),
    Return = (QueueMask, RetainedState, Return),
    Yield = bool,
>;

#[pin_project]
pub struct QueueFutureBlock<Ret, Inner, Recycle, Retain>
where
    Inner: QueueFutureBlockGenerator<Ret, Recycle, Retain>,
{
    #[pin]
    inner: Inner,
    retained_state: Option<Retain>,

    /// This is more of a hack to save the dependent queue mask when `init` was called
    /// so we can initialize `__current_queue_mask` with this value.
    initial_queue_mask: QueueMask,
    _marker: PhantomData<fn(&mut Recycle) -> Ret>,
}
impl<Ret, Inner, Fut, Recycle> QueueFutureBlock<Ret, Inner, Fut, Recycle>
where
    Inner: QueueFutureBlockGenerator<Ret, Fut, Recycle>,
{
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            initial_queue_mask: QueueMask::empty(),
            retained_state: None,
            _marker: PhantomData,
        }
    }
}
impl<Ret, Inner, Retain, Recycle: Default> QueueFuture
    for QueueFutureBlock<Ret, Inner, Recycle, Retain>
where
    Inner: QueueFutureBlockGenerator<Ret, Recycle, Retain>,
{
    type Output = Ret;
    type RecycledState = Recycle;
    type RetainedState = Retain;
    fn init(
        self: Pin<&mut Self>,
        ctx: &mut SubmissionContext,
        recycled_state: &mut Self::RecycledState,
        prev_queue: QueueMask,
    ) {
        *self.project().initial_queue_mask = prev_queue;
    }
    fn record(
        self: Pin<&mut Self>,
        ctx: &mut SubmissionContext,
        recycled_state: &mut Self::RecycledState,
    ) -> QueueFuturePoll<Ret> {
        let this = self.project();
        assert!(
            this.retained_state.is_none(),
            "Calling record after returning Complete"
        );
        match this.inner.resume((
            *this.initial_queue_mask,
            ctx as *mut _ as *mut (),
            recycled_state,
        )) {
            std::ops::GeneratorState::Yielded(is_semaphore) => {
                if is_semaphore {
                    QueueFuturePoll::Semaphore
                } else {
                    QueueFuturePoll::Barrier
                }
            }
            std::ops::GeneratorState::Complete((next_queue, dispose, output)) => {
                *this.retained_state = Some(dispose);
                QueueFuturePoll::Ready { next_queue, output }
            }
        }
    }
    fn dispose(mut self) -> Retain {
        self.retained_state.take().unwrap()
    }
}

/*
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

*/
#[pin_project]
pub struct RunCommandsQueueFuture<I: GPUCommandFuture> {
    #[pin]
    inner: I,

    /// Specified queue to run on.
    queue: QueueRef, // If null, use the previous queue.
    /// Retained state and the timeline index
    retained_state: Option<I::RetainedState>,
    prev_queue: QueueMask,
}
impl<I: GPUCommandFuture> RunCommandsQueueFuture<I> {
    pub fn new(future: I, queue: QueueRef) -> Self {
        Self {
            inner: future,
            queue,
            retained_state: None,
            prev_queue: QueueMask::empty(),
        }
    }
}
impl<I: GPUCommandFuture> QueueFuture for RunCommandsQueueFuture<I> {
    type Output = I::Output;
    type RecycledState = I::RecycledState;
    type RetainedState = I::RetainedState;

    fn init(
        self: Pin<&mut Self>,
        ctx: &mut SubmissionContext,
        recycled_state: &mut Self::RecycledState,
        prev_queue: QueueMask,
    ) {
        println!("init");
        let this = self.project();
        if this.queue.is_null() {
            let mut iter = prev_queue.iter();
            if let Some(inherited_queue) = iter.next() {
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
        let mut r = &mut ctx.submission[this.queue.0 as usize];
        match &r {
            QueueSubmissionType::Unknown => {
                *r = QueueSubmissionType::Submit {
                    command_buffers: Vec::new(),
                    recording_command_buffer: None,
                };
            }
            QueueSubmissionType::Submit { .. } => (),
            _ => panic!(),
        };
        let (command_buffers, recording_command_buffer) = match &mut r {
            QueueSubmissionType::Submit {
                command_buffers,
                recording_command_buffer,
            } => (command_buffers, recording_command_buffer),
            _ => unreachable!(),
        };
        let q = &mut ctx.queues[this.queue.0 as usize];
        let mut command_ctx = CommandBufferRecordContext {
            stage_index: q.stage_index,
            command_buffers,
            command_pool: ctx.shared_command_pools[q.queue_family_index as usize]
                .as_mut()
                .unwrap(),
            recording_command_buffer,
            timeline_index: q.timeline_index,
            queue: *this.queue,
        };
        this.inner.init(&mut command_ctx, recycled_state);
    }

    fn record(
        self: Pin<&mut Self>,
        ctx: &mut SubmissionContext,
        recycled_state: &mut Self::RecycledState,
    ) -> QueueFuturePoll<Self::Output> {
        println!("record");
        let this = self.project();

        let queue = {
            let mut mask = QueueMask::empty();
            mask.set_queue(*this.queue);
            mask
        };

        if !this.prev_queue.is_empty() {
            let ret = if *this.prev_queue == queue {
                QueueFuturePoll::Barrier
            } else {
                QueueFuturePoll::Semaphore
            };
            *this.prev_queue = QueueMask::empty();
            return ret;
        }
        // Ok, let's reframework this.
        // A queue future can signal any number of things.
        // Wether we want to wait on that, depends on resource access conflicts.

        let mut r = &mut ctx.submission[this.queue.0 as usize];

        match &r {
            QueueSubmissionType::Unknown => {
                *r = QueueSubmissionType::Submit {
                    command_buffers: Vec::new(),
                    recording_command_buffer: None,
                };
            }
            QueueSubmissionType::Submit { .. } => (),
            _ => panic!(),
        };

        let (command_buffers, recording_command_buffer) = match &mut r {
            QueueSubmissionType::Submit {
                command_buffers,
                recording_command_buffer,
            } => (command_buffers, recording_command_buffer),
            _ => unreachable!(),
        };
        let q = &mut ctx.queues[this.queue.0 as usize];
        let mut command_ctx = CommandBufferRecordContext {
            stage_index: q.stage_index,
            command_buffers,
            command_pool: ctx.shared_command_pools[q.queue_family_index as usize]
                .as_mut()
                .unwrap(),
            recording_command_buffer,
            timeline_index: q.timeline_index,
            queue: *this.queue,
        };

        let poll = if q.stage_index == 0 {
            command_ctx.record_first_step(this.inner, recycled_state, |a| {
                for (src_queue, dst_queue, src_stages, dst_stages) in &a.semaphore_transitions {
                    let id = ctx.queues[src_queue.0 as usize].signals.len() as u32;
                    ctx.queues[src_queue.0 as usize].signals.insert(*src_stages);
                    ctx.queues[dst_queue.0 as usize].waits.push((
                        *dst_stages,
                        *src_queue,
                        *src_stages,
                    ));
                }
            })
        } else {
            command_ctx.record_one_step(this.inner, recycled_state)
        };
        let result = match poll {
            Poll::Ready((output, retained_state)) => {
                *this.retained_state = Some(retained_state);
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
        let r = &mut ctx.queues[this.queue.0 as usize];
        r.stage_index += 1;
        result
    }

    fn dispose(mut self) -> Self::RetainedState {
        self.retained_state.take().unwrap()
    }
}
