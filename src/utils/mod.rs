mod format;
pub mod resource_pool;
pub use format::*;
use std::ops::Deref;

pub enum SharingMode<T>
where
    T: Deref<Target = [u32]>,
{
    Exclusive,
    Concurrent { queue_family_indices: T },
}
