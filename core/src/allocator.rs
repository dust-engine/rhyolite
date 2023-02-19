use std::sync::Arc;

use crate::{Device, HasDevice};
use ash::vk;

struct AllocatorInner {
    // This needs to be defined before device, so that it gets dropped hefore device gets dropped.
    inner: vk_mem::Allocator,
    device: Arc<Device>,
}

#[derive(Clone)]
pub struct Allocator(Arc<AllocatorInner>);

impl Allocator {
    pub fn inner(&self) -> &vk_mem::Allocator {
        &self.0.inner
    }
    pub fn new(device: Arc<Device>) -> Self {
        let mut allocator_flags: vk_mem::AllocatorCreateFlags =
            vk_mem::AllocatorCreateFlags::empty();
        if device
            .physical_device()
            .features()
            .v12
            .buffer_device_address
            == vk::TRUE
        {
            allocator_flags |= vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
        }

        let allocator = vk_mem::Allocator::new(
            vk_mem::AllocatorCreateInfo::new(
                device.instance().as_ref(),
                device.as_ref(),
                device.physical_device().raw(),
            )
            .vulkan_api_version(vk::make_api_version(0, 1, 3, 0))
            .flags(allocator_flags),
        )
        .unwrap();
        Self(Arc::new(AllocatorInner {
            inner: allocator,
            device,
        }))
    }
}

impl HasDevice for Allocator {
    fn device(&self) -> &Arc<Device> {
        &self.0.device
    }
}
