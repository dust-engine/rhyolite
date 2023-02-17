use std::ops::Deref;

use bevy_ecs::system::Resource;
use rhyolite::{
    commands::SharedCommandPool,
    future::{use_per_frame_state_blocking, PerFrameState},
    FencePool, HasDevice, QueueFuture, TimelineSemaphorePool,
};

use self::frame_guard::FrameGuard;

pub struct Frame {
    shared_command_pools: Vec<Option<SharedCommandPool>>,
    pub(crate) shared_semaphore_pool: TimelineSemaphorePool,
    shared_fence_pool: FencePool,
}

#[derive(Resource)]
pub struct Queues {
    queues: rhyolite::Queues,
    frames: PerFrameState<Frame>,
    current_frame: Option<FrameGuard>,
    max_frame_in_flight: usize,
    task_pool: bevy_tasks::TaskPool,
}

mod frame_guard {
    use std::sync::Arc;

    use rhyolite::future::PerFrameContainer;

    use super::Frame;

    pub(super) struct FrameGuard {
        inner: Arc<PerFrameContainer<Frame>>,
    }
    impl FrameGuard {
        pub(super) fn new(frame: PerFrameContainer<Frame>) -> Self {
            FrameGuard {
                inner: Arc::new(frame),
            }
        }
        pub(super) fn guard(&self) -> FrameGuardHandle {
            FrameGuardHandle {
                inner: self.inner.clone(),
            }
        }
        pub(super) fn inner_mut(&mut self) -> &mut Frame {
            // Safety: All other references should be FrameGuardHandle with no access to the internals.
            let frame: &mut PerFrameContainer<Frame> =
                unsafe { Arc::get_mut_unchecked(&mut self.inner) };
            frame
        }
    }
    pub(super) struct FrameGuardHandle {
        inner: Arc<PerFrameContainer<Frame>>,
    }
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
        self.current_frame = Some(FrameGuard::new(use_per_frame_state_blocking(
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
        self.current_frame.as_mut().unwrap().inner_mut()
    }
    pub fn submit<F: QueueFuture<Output = ()> + 'static>(
        &mut self,
        future: F,
        recycled_state: &mut F::RecycledState,
    ) {
        let current_frame: &mut Frame = self.current_frame.as_mut().unwrap().inner_mut();
        let future = self.queues.submit(
            future,
            &mut current_frame.shared_command_pools,
            &mut current_frame.shared_semaphore_pool,
            &mut current_frame.shared_fence_pool,
            recycled_state,
        );

        let guard = self.current_frame.as_ref().unwrap().guard();
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
