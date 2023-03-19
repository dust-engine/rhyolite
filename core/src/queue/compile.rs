use std::{any::Any, marker::PhantomData};

use ash::vk;

use crate::{QueueFuture, SubmissionContext, QueueSubmissionContext, QueueMask, QueueFuturePoll, QueueSubmissionType, SubmissionBatch, commands::SharedCommandPool, TimelineSemaphorePool, FencePool, utils::retainer::Retainer, HasDevice};
use super::exec::CachedStageSubmissions;

pub(super) trait QueueCompileExt: QueueFuture {
    fn compile<'a>(
        mut self,
        // These pools are passed in as argument so that they can be cleaned on a regular basis (per frame) externally.
        // The lifetime parameter prevents the caller from dropping the polls before awaiting the returned future.
        shared_command_pools: &'a mut [Option<SharedCommandPool>],
        semaphore_pool: &'a mut TimelineSemaphorePool,
        recycled_state: &mut Self::RecycledState,
        apply_final_signal: bool,
    ) -> CompiledQueueFuture<'a, Self>
    where
        Self::RetainedState: 'a,
        Self::Output: 'a,
        Self: Sized,
    {
        let device = semaphore_pool.device().clone();
        let mut future_pinned = unsafe { std::pin::Pin::new_unchecked(&mut self) };
        let mut submission_context = SubmissionContext {
            shared_command_pools,
            queues: device.queue_info()
                .queues
                .iter()
                .map(|(queue_family_index, _)| QueueSubmissionContext {
                    queue_family_index: *queue_family_index,
                    stage_index: 0,
                    timeline_index: 1,
                    signals: Default::default(),
                    waits: Default::default(),
                    exports: Default::default(),
                })
                .collect(),
            submission: device.queue_info().queues
                .iter()
                .map(|_| QueueSubmissionType::Unknown)
                .collect(),
        };

        let mut current_stage = CachedStageSubmissions::new(device.queue_info().queues.len());
        let mut submission_stages: Vec<CachedStageSubmissions> = Vec::new();
        let mut last_submit_stage_index = vec![usize::MAX; device.queue_info().queues.len()];

        future_pinned
            .as_mut()
            .setup(&mut submission_context, recycled_state, QueueMask::empty());
        let (output, final_signals) = loop {
            match future_pinned
                .as_mut()
                .record(&mut submission_context, recycled_state)
            {
                QueueFuturePoll::Barrier => {
                    continue;
                }
                QueueFuturePoll::Semaphore(additional_semaphores_to_wait) => {
                    let mut last_stage = std::mem::replace(
                        &mut current_stage,
                        CachedStageSubmissions::new(device.queue_info().queues.len()),
                    );
                    last_stage.apply_signals(&submission_context, semaphore_pool);
                    current_stage.apply_submissions(
                        &submission_context,
                        &last_stage,
                        semaphore_pool,
                    );
                    current_stage.wait_additional_signals(
                        &submission_context,
                        additional_semaphores_to_wait,
                    );
                    submission_stages.push(last_stage);

                    for (i, (src, dst)) in submission_context
                        .submission
                        .iter_mut()
                        .zip(current_stage.queues.iter_mut())
                        .enumerate()
                    {
                        dst.ty = std::mem::replace(src, QueueSubmissionType::Unknown);
                        if let QueueSubmissionType::Submit { .. } = &dst.ty {
                            last_submit_stage_index[i] = submission_stages.len();
                        }
                    }

                    for (i, ctx) in submission_context.queues.iter().enumerate() {
                        if ctx.exports.is_empty() {
                            continue;
                        }

                        let last_accessed_stage = last_submit_stage_index[i];
                        submission_stages[last_accessed_stage].queues[i]
                            .apply_exports(ctx, &device);
                    }

                    for ctx in submission_context.queues.iter_mut() {
                        ctx.stage_index = 0;
                        ctx.timeline_index += 1;
                        ctx.waits.clear();
                        ctx.signals.clear();
                        ctx.exports.clear();
                    }
                }
                QueueFuturePoll::Ready {
                    next_queue: _,
                    output,
                } => {
                    let mut last_stage = std::mem::replace(
                        &mut current_stage,
                        CachedStageSubmissions::new(device.queue_info().queues.len()),
                    );
                    last_stage.apply_signals(&submission_context, semaphore_pool);
                    current_stage.apply_submissions(
                        &submission_context,
                        &last_stage,
                        semaphore_pool,
                    );
                    submission_stages.push(last_stage);

                    for (i, (src, dst)) in submission_context
                        .submission
                        .iter_mut()
                        .zip(current_stage.queues.iter_mut())
                        .enumerate()
                    {
                        dst.ty = std::mem::replace(src, QueueSubmissionType::Unknown);
                        if let QueueSubmissionType::Submit { .. } = &dst.ty {
                            last_submit_stage_index[i] = submission_stages.len();
                        }
                    }
                    for (i, ctx) in submission_context.queues.iter().enumerate() {
                        if ctx.exports.is_empty() {
                            continue;
                        }

                        let last_accessed_stage = last_submit_stage_index[i];
                        submission_stages[last_accessed_stage].queues[i]
                            .apply_exports(ctx, &device);
                    }
                    let final_signals = if apply_final_signal {
                        Some(current_stage.apply_final_signals(&submission_context, semaphore_pool))
                    } else {
                        None
                    };
                    submission_stages.push(current_stage);
                    break (output, final_signals);
                }
            }
        };

        for mut stage in submission_stages.iter_mut() {
            for q in stage.queues.iter_mut() {
                q.ty.end(&device);
            }
        }

        // No more touching of future! It's getting moved.
        let fut_dispose = self.dispose();
        CompiledQueueFuture {
            submission_batch: submission_stages,
            fut_dispose,
            final_signals,
            output,
            _marker: PhantomData
        }
    }
}

pub(super) struct CompiledQueueFuture<'a, F: QueueFuture> {
    pub(super) submission_batch: Vec<CachedStageSubmissions>,
    pub(super) fut_dispose: F::RetainedState,
    pub(super) final_signals: Option<Vec<(vk::Semaphore, u64)>>,
    pub(super) output: F::Output,
    _marker: PhantomData<&'a ()>
}

impl<T: QueueFuture> QueueCompileExt for T {

}