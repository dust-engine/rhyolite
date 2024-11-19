use std::{borrow::Borrow, pin::Pin, ptr::NonNull};

use ash::vk;

use crate::semaphore::TimelineSemaphore;


pub struct ResourceState {
    stage: vk::PipelineStageFlags2,
    mask: vk::AccessFlags2,
    queue_family: u32,
    layout: vk::ImageLayout,
}

// Two possible implementations:
// 1. the resource state is inside the resource: directly return.
// 2. the inner product is wrapped in something like an Arc, so we can't safely return its ResoruceState.
// In this case we use the index to index into barriercontext itself.
pub trait TrackedResource {
    fn resource_state(&mut self) -> &mut ResourceState;
}


// Some quesitons to answer:
// 1. swapchain
// 2. queue family ownership
//
// Each command buffer has an "expected states" and "target states"
// To concat two command buffers, we insert barrier from target states of prev command buffer to expected states of next command buffer
// Command buffers that can be executed concurrently are zipped together.
//
// queue family ownership transfers are done at command buffer submission time.
// after all subsequent commandns are in, we submit queue transfer commands to the previous stage.

// queue family ownership transfer IN happens inline.
// queue family ownership transfer OUT happens at the intermediate buffers