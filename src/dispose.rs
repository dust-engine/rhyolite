use std::{
    mem::ManuallyDrop,
    sync::{Arc, Weak},
};

use ash::vk;
use bevy::{app::Plugin, ecs::system::Resource};
use crossbeam_channel::Sender;

use crate::{commands::SemaphoreSignalCommands, semaphore::TimelineSemaphore};
use smallvec::SmallVec;

pub trait DisposeObject: Send + Sync {
    fn wait_blocked(&mut self);
}

static DISPOSER: once_cell::sync::OnceCell<Disposer> = once_cell::sync::OnceCell::new();

pub struct Disposer {
    join_handle: std::thread::JoinHandle<()>,
    sender: Weak<Sender<Box<dyn DisposeObject>>>,
}

pub(crate) struct DisposerPlugin;
impl Plugin for DisposerPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        let (sender, receiver) = crossbeam_channel::unbounded::<Box<dyn DisposeObject>>();
        let join_handle = std::thread::Builder::new()
            .name("Disposer thread".into())
            .spawn(move || loop {
                let Ok(mut handle) = receiver.recv() else {
                    return;
                };
                handle.wait_blocked();
                drop(handle);
            })
            .unwrap();

        let sender = Arc::new(sender);
        let weak = Arc::downgrade(&sender);

        let mut disposer = Some(Disposer {
            join_handle,
            sender: weak,
        });

        DISPOSER.get_or_init(|| disposer.take().unwrap());
        assert!(
            disposer.is_none(),
            "DisposerPlugin should only be added once"
        );

        #[derive(Resource)]
        struct DisposerTeardown {
            _sender: Arc<Sender<Box<dyn DisposeObject>>>,
        }

        let teardown = DisposerTeardown { _sender: sender };
        app.insert_resource(teardown);
    }
}

pub fn dispose(mut handle: Box<dyn DisposeObject>) {
    let Some(sender) = DISPOSER
        .get()
        .expect("DisposerPlugin not added")
        .sender
        .upgrade()
    else {
        // Application teardown. Block on the spot.
        handle.wait_blocked();
        drop(handle);
        return;
    };
    sender.send(handle).unwrap();
}

/// Objects that may be used on the GPU timeline typically includes [`DisposeHandle`] and implements [`Dispose`].
pub struct RenderObject<T: Send + Sync + 'static>(ManuallyDrop<RenderObjectInner<T>>);
struct RenderObjectInner<T> {
    inner: T,
    pub(crate) semaphores: SmallVec<[(Arc<TimelineSemaphore>, u64); 2]>,
}
impl<T: Send + Sync> DisposeObject for RenderObjectInner<T> {
    fn wait_blocked(&mut self) {
        if self.semaphores.is_empty() {
            return;
        }
        TimelineSemaphore::wait_all_blocked(
            self.semaphores
                .iter()
                .map(|(sem, val)| (sem.as_ref(), *val)),
            !0,
        )
        .unwrap();
    }
}
impl<T: Send + Sync + 'static> RenderObject<T> {
    pub fn new(item: T) -> Self {
        Self(ManuallyDrop::new(RenderObjectInner {
            inner: item,
            semaphores: SmallVec::new(),
        }))
    }
    pub fn use_on(&mut self, queue: &mut impl SemaphoreSignalCommands) -> &mut T {
        let (sem, val) = queue.signal_semaphore(vk::PipelineStageFlags2::ALL_COMMANDS);
        for (curr_sem, curr_val) in self.0.semaphores.iter_mut() {
            if Arc::ptr_eq(curr_sem, &sem) {
                *curr_val = val.max(*curr_val);
                return &mut self.0.inner;
            }
        }
        self.0.semaphores.push((sem, val));
        &mut self.0.inner
    }
    pub fn get(&self) -> &T {
        &self.0.inner
    }
}
impl<T: Send + Sync + 'static> Drop for RenderObject<T> {
    fn drop(&mut self) {
        if self.0.semaphores.is_empty() {
            // drop in place
            unsafe {
                ManuallyDrop::drop(&mut self.0);
            }
            return;
        }
        unsafe {
            let inner = ManuallyDrop::take(&mut self.0);
            dispose(Box::new(inner));
        }
    }
}
