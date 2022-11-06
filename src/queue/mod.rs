pub use crate::semaphore::{SemaphoreOp, StagedSemaphoreOp};
mod router;
use crate::Device;
use ash::{prelude::VkResult, vk};
pub use router::{QueueIndex, QueueType, Queues, QueuesCreateInfo};
use std::sync::Arc;

pub struct Queue {
    pub(super) device: Arc<Device>,
    pub(super) queue: vk::Queue,
    family_index: u32,
}

impl crate::debug::DebugObject for Queue {
    const OBJECT_TYPE: vk::ObjectType = vk::ObjectType::QUEUE;
    fn object_handle(&mut self) -> u64 {
        unsafe { std::mem::transmute(self.queue) }
    }
}

impl crate::HasDevice for Queue {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

/// A thin wrapper for a Vulkan Queue. Most queue operations require host-side syncronization,
/// so these Queue operations take a mutable reference. To perform queue operations safely from multiple
/// threads, either Arc<Mutex<_>> or a threaded dispatcher would be required.
impl Queue {
    pub fn family_index(&self) -> u32 {
        self.family_index
    }

    /// # Safety
    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkQueueSubmit.html>
    pub unsafe fn submit_raw(
        &mut self,
        submits: &[vk::SubmitInfo],
        fence: vk::Fence,
    ) -> VkResult<()> {
        self.device.queue_submit(self.queue, submits, fence)
    }

    /// # Safety
    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkQueueSubmit.html>
    pub unsafe fn submit_raw2(
        &mut self,
        submits: &[vk::SubmitInfo2],
        fence: vk::Fence,
    ) -> VkResult<()> {
        self.device.queue_submit2(self.queue, submits, fence)
    }

    pub unsafe fn bind_sparse(
        &mut self,
        infos: &[vk::BindSparseInfo],
        fence: vk::Fence,
    ) -> VkResult<()> {
        self.device.queue_bind_sparse(self.queue, infos, fence)
    }
}
