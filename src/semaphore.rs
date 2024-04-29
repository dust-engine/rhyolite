use std::{fmt::Debug, sync::atomic::AtomicU64};

use ash::{
    prelude::VkResult,
    vk::{self},
};

use crate::{utils::AsVkHandle, Device, HasDevice};

pub struct TimelineSemaphore {
    device: Device,
    semaphore: vk::Semaphore,
    current_value: AtomicU64,
}
impl Debug for TimelineSemaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self
            .current_value
            .load(std::sync::atomic::Ordering::Relaxed);
        f.debug_tuple("TimelineSemaphore")
            .field(&self.semaphore)
            .field(&value)
            .finish()
    }
}

impl TimelineSemaphore {
    pub fn new(device: Device) -> VkResult<Self> {
        let semaphore = unsafe {
            let mut type_info = vk::SemaphoreTypeCreateInfo {
                semaphore_type: vk::SemaphoreType::TIMELINE,
                ..Default::default()
            };
            let info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
            device.create_semaphore(&info, None)
        }?;
        Ok(Self {
            device,
            semaphore,
            current_value: AtomicU64::new(0),
        })
    }
    pub fn raw(&self) -> vk::Semaphore {
        self.semaphore
    }
    pub fn is_signaled(&self, val: u64) -> bool {
        self.current_value
            .load(std::sync::atomic::Ordering::Relaxed)
            >= val
    }
    pub fn wait_blocked(&self, value: u64, timeout: u64) -> VkResult<()> {
        if self
            .current_value
            .load(std::sync::atomic::Ordering::Relaxed)
            >= value
        {
            return Ok(());
        }
        unsafe {
            self.device.wait_semaphores(
                &vk::SemaphoreWaitInfo {
                    semaphore_count: 1,
                    p_semaphores: &self.semaphore,
                    p_values: &value,
                    ..Default::default()
                },
                timeout,
            )?;
        }
        self.current_value
            .fetch_max(value, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
    pub fn wait_all_blocked<'a>(
        semaphores: impl Iterator<Item = (&'a Self, u64)>,
        timeout: u64,
    ) -> VkResult<()> {
        let mut device: Option<&Device> = None;
        let semaphores = semaphores
            .filter(|(s, t)| s.current_value.load(std::sync::atomic::Ordering::Relaxed) < *t)
            .map(|(s, t)| {
                device = Some(&s.device);
                (s, s.semaphore, t)
            });
        let (semaphores, raws, values): (Vec<&TimelineSemaphore>, Vec<vk::Semaphore>, Vec<u64>) =
            itertools::multiunzip(semaphores);
        if semaphores.is_empty() {
            return Ok(());
        }
        unsafe {
            device.unwrap().wait_semaphores(
                &vk::SemaphoreWaitInfo {
                    semaphore_count: semaphores.len() as u32,
                    p_semaphores: raws.as_ptr(),
                    p_values: values.as_ptr(),
                    ..Default::default()
                },
                timeout,
            )?;
        }
        for (&semaphore, &value) in semaphores.iter().zip(values.iter()) {
            semaphore
                .current_value
                .fetch_max(value, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }
}
impl HasDevice for TimelineSemaphore {
    fn device(&self) -> &Device {
        &self.device
    }
}
impl AsVkHandle for TimelineSemaphore {
    type Handle = vk::Semaphore;
    fn vk_handle(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl Drop for TimelineSemaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.semaphore, None);
        }
    }
}
