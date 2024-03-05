mod format;
mod future;
use ash::vk;
pub use format::*;
pub use future::*;
use std::{mem::ManuallyDrop, ops::Deref};

#[derive(Debug, Clone)]
pub enum SharingMode<T>
where
    T: Deref<Target = [u32]>,
{
    Exclusive,
    Concurrent { queue_family_indices: T },
}

impl<T: Deref<Target = [u32]>> SharingMode<T> {
    pub fn as_raw(&self) -> vk::SharingMode {
        match self {
            Self::Exclusive => vk::SharingMode::EXCLUSIVE,
            Self::Concurrent { .. } => vk::SharingMode::CONCURRENT,
        }
    }

    pub fn queue_family_indices(&self) -> &[u32] {
        match self {
            Self::Exclusive => &[],
            Self::Concurrent {
                queue_family_indices,
            } => &queue_family_indices.deref(),
        }
    }
}

/// A wrapper for GPU-owned resources that may currently be in use.
/// Dispose-wrapped objects may not be immediately dropped. They must be passed to the `retain` method
/// so that they're kept alive until the frame end.
#[repr(transparent)]
pub struct Dispose<T>(ManuallyDrop<T>);
impl<T> Drop for Dispose<T> {
    fn drop(&mut self) {
        panic!("Dropping Dispose<T> without calling retain is forbidden.");
    }
}
impl<T> Dispose<T> {
    pub fn new(inner: T) -> Self {
        assert!(std::mem::needs_drop::<T>());
        Self(ManuallyDrop::new(inner))
    }
    pub(crate) unsafe fn take(mut self) -> T {
        unsafe {
            let item = ManuallyDrop::take(&mut self.0);
            std::mem::forget(self);
            item
        }
    }
}

#[repr(transparent)]
pub struct SendBox<T>(T);
unsafe impl<T> Send for SendBox<T> {}
unsafe impl<T> Sync for SendBox<T> {}
impl<T> SendBox<T> {
    pub unsafe fn new(inner: T) -> Self {
        Self(inner)
    }
    pub fn into_inner(self) -> T {
        self.0
    }
}
