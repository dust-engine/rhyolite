use std::sync::atomic::AtomicU64;

use ash::{prelude::VkResult, vk};

use crate::Device;

pub struct TimelineSemaphore {
    device: Device,
    semaphore: vk::Semaphore,
    current_value: AtomicU64,
}

impl TimelineSemaphore {
    pub fn new(device: Device) -> VkResult<Self>{
        let semaphore = unsafe {
            let mut type_info = vk::SemaphoreTypeCreateInfo::builder()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .build();
            let info = vk::SemaphoreCreateInfo::builder()
                .push_next(&mut type_info)
                .build();
            device.create_semaphore(&info, None)
        }?;
        Ok(Self {
            device,
            semaphore,
            current_value: AtomicU64::new(0)
        })
    }
    pub fn raw(&self) -> vk::Semaphore {
        self.semaphore
    }
    pub fn wait_blocked(&self, value: u64, timeout: u64) -> VkResult<()> {
        if self.current_value.load(std::sync::atomic::Ordering::Relaxed) >= value {
            return Ok(());
        }
        unsafe {
            self.device.wait_semaphores(&vk::SemaphoreWaitInfo {
                semaphore_count: 1,
                p_semaphores: &self.semaphore,
                p_values: &value,
                ..Default::default()
            }, timeout)?;
        }
        self.current_value.fetch_max(value, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

impl Drop for TimelineSemaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.semaphore, None);
        }
    }
}
