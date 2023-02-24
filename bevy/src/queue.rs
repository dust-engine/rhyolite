use std::ops::{Deref, DerefMut};

use bevy_ecs::system::Resource;
use rhyolite::{
    commands::SharedCommandPool,
    future::{use_per_frame_state_blocking, PerFrameContainer, PerFrameState},
    utils::retainer::Retainer,
    FencePool, HasDevice, QueueFuture, TimelineSemaphorePool,
};

pub struct Frame {
    shared_command_pools: Vec<Option<SharedCommandPool>>,
    pub(crate) shared_semaphore_pool: TimelineSemaphorePool,
    shared_fence_pool: FencePool,
}

#[derive(Resource)]
pub struct Queues {
    queues: rhyolite::Queues,
    frames: PerFrameState<Frame>,
    current_frame: Option<Retainer<PerFrameContainer<Frame>>>,
    max_frame_in_flight: usize,
    task_pool: bevy_tasks::TaskPool,
}

impl Queues {
    pub fn new(queues: rhyolite::Queues, max_frame_in_flight: usize) -> Self {
        Self {
            queues,
            frames: Default::default(),
            current_frame: None,
            max_frame_in_flight,
            task_pool: bevy_tasks::TaskPool::new(),
        }
    }
    /// May block.
    pub fn next_frame(&mut self) {
        self.current_frame = Some(Retainer::new(use_per_frame_state_blocking(
            &mut self.frames,
            self.max_frame_in_flight,
            || Frame {
                shared_command_pools: self.queues.make_shared_command_pools(),
                shared_fence_pool: rhyolite::FencePool::new(self.queues.device().clone()),
                shared_semaphore_pool: rhyolite::TimelineSemaphorePool::new(
                    self.queues.device().clone(),
                ),
            },
            |frame| {
                for i in frame.shared_command_pools.iter_mut() {
                    if let Some(i) = i {
                        i.reset(false);
                    }
                }
                frame.shared_fence_pool.reset();
                frame.shared_semaphore_pool.reset();
            },
        )));
    }
    pub fn current_frame(&mut self) -> &mut Frame {
        self.current_frame.as_mut().unwrap().deref_mut()
    }
    pub fn num_frame_in_flight(&self) -> u32 {
        self.max_frame_in_flight as u32
    }
    pub fn submit<F: QueueFuture<Output = ()>>(
        &mut self,
        future: F,
        recycled_state: &mut F::RecycledState,
    ) where
        F::Output: 'static,
        F::RetainedState: 'static,
    {
        let current_frame: &mut Frame = self.current_frame.as_mut().unwrap().deref_mut();
        let future = self.queues.submit(
            future,
            &mut current_frame.shared_command_pools,
            &mut current_frame.shared_semaphore_pool,
            &mut current_frame.shared_fence_pool,
            recycled_state,
        );

        let guard = self.current_frame.as_ref().unwrap().handle();
        let task = async {
            let out = future.await;
            drop(guard);
            out
        };
        self.task_pool.spawn(task).detach();
    }
}

#[derive(Resource)]
pub struct QueuesRouter(rhyolite::QueuesRouter);
impl QueuesRouter {
    pub fn new(inner: rhyolite::QueuesRouter) -> Self {
        Self(inner)
    }
}

impl Deref for QueuesRouter {
    type Target = rhyolite::QueuesRouter;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
