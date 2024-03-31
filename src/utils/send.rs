use std::ops::{Deref, DerefMut};

/// Hack to allow for a `Send` bound on a trait object. Very unsafe. Mostly used with `vk::Create*Info` types.
/// Pending ash v0.38 release.
/// https://github.com/ash-rs/ash/pull/869
#[repr(transparent)]
pub struct SendBox<T>(pub T);
unsafe impl<T> Send for SendBox<T> {}
unsafe impl<T> Sync for SendBox<T> {}
impl<T> SendBox<T> {
    pub fn into_inner(self) -> T {
        self.0
    }
}
impl<T> Deref for SendBox<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}
impl<T> DerefMut for SendBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}
