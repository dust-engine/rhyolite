use std::sync::atomic::AtomicU64;

use ash::vk;

/// A ring buffer for uploads to the GPU.
pub struct StagingBelt {
    chunk_size: vk::DeviceSize,
    head: AtomicU64,
    tail: AtomicU64,
}
