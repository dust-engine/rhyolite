mod format;
use ash::vk;
pub use format::*;
use std::ops::Deref;

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
            Self::Concurrent { queue_family_indices } => &queue_family_indices.deref()
        }
    }
}
