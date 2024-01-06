mod format;

pub use format::*;
use std::ops::Deref;

pub enum SharingMode<T>
where
    T: Deref<Target = [u32]>,
{
    Exclusive,
    Concurrent { queue_family_indices: T },
}
