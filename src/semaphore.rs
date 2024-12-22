use std::{
    fmt::Debug,
    sync::{atomic::AtomicU64, Arc},
};

use ash::{
    prelude::VkResult,
    vk::{self},
};

use crate::{utils::AsVkHandle, Device, HasDevice};

/// A thin wrapper around [Timeline Semaphores](https://www.khronos.org/blog/vulkan-timeline-semaphores).
/// We additionally cache the timeline semaphore value in an [`AtomicU64`] to reduce the number of API
/// calls needed for common operations such as `wait_blocked`.
pub struct TimelineSemaphore {
    device: Device,
    semaphore: vk::Semaphore,
    value: AtomicU64,
}
impl Debug for TimelineSemaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.value();
        f.debug_tuple("TimelineSemaphore")
            .field(&self.semaphore)
            .field(&value)
            .finish()
    }
}

impl TimelineSemaphore {
    pub fn new(device: Device, initial_value: u64) -> VkResult<Self> {
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
            value: AtomicU64::new(initial_value),
        })
    }
    pub fn raw(&self) -> vk::Semaphore {
        self.semaphore
    }
    pub fn value(&self) -> u64 {
        let new_value = unsafe {
            self.device
                .get_semaphore_counter_value(self.semaphore)
                .unwrap()
        };
        let old_value = self
            .value
            .fetch_max(new_value, std::sync::atomic::Ordering::Relaxed);
        debug_assert!(old_value <= new_value);
        new_value
    }
    pub fn is_signaled(&self, val: u64) -> bool {
        self.value() >= val
    }
    pub fn signal(&self, val: u64) {
        let old_value = self.value.load(std::sync::atomic::Ordering::Relaxed);
        if old_value >= val {
            return;
        }
        unsafe {
            self.device
                .signal_semaphore(&vk::SemaphoreSignalInfo {
                    semaphore: self.semaphore,
                    value: val,
                    ..Default::default()
                })
                .unwrap();
            self.value.store(val, std::sync::atomic::Ordering::Relaxed);
        }
    }
    pub fn wait_blocked(&self, value: u64, timeout: u64) -> VkResult<()> {
        if self.value.load(std::sync::atomic::Ordering::Relaxed) >= value {
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
        self.value
            .fetch_max(value, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
    pub fn wait_all_blocked<'a>(
        semaphores: impl Iterator<Item = (&'a Self, u64)>,
        timeout: u64,
    ) -> VkResult<()> {
        let mut device: Option<&Device> = None;
        let semaphores = semaphores
            .filter(|(sem, wait_value)| {
                sem.value.load(std::sync::atomic::Ordering::Relaxed) < *wait_value
            })
            .map(|(s, t)| {
                device = Some(&s.device);
                (s, s.semaphore, t)
            });
        let (semaphores, raws, wait_values): (
            Vec<&TimelineSemaphore>,
            Vec<vk::Semaphore>,
            Vec<u64>,
        ) = itertools::multiunzip(semaphores);
        if semaphores.is_empty() {
            return Ok(());
        }
        unsafe {
            device.unwrap().wait_semaphores(
                &vk::SemaphoreWaitInfo {
                    semaphore_count: semaphores.len() as u32,
                    p_semaphores: raws.as_ptr(),
                    p_values: wait_values.as_ptr(),
                    ..Default::default()
                },
                timeout,
            )?;
        }
        for (sem, wait_value) in semaphores.into_iter().zip(wait_values) {
            sem.value
                .fetch_max(wait_value, std::sync::atomic::Ordering::Relaxed);
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

struct SemaphoreDeferredValueWait {
    semaphore: Arc<TimelineSemaphore>,
    wait_value: u64,
}

impl Drop for SemaphoreDeferredValueWait {
    fn drop(&mut self) {
        self.semaphore.wait_blocked(self.wait_value, !0).unwrap();
    }
}

/// T protected by a timeline semaphore.
pub struct SemaphoreDeferredValue<T> {
    /// This must be defined in this struct before `value`. This ensures that `value` won't be dropped
    /// until we've awaited on the timeline semaphore.
    wait: SemaphoreDeferredValueWait,
    value: T,
}

impl<T> SemaphoreDeferredValue<T> {
    pub fn unwrap_blocked(self) -> T {
        drop(self.wait); // This blocks on the semaphore
        self.value
    }
    pub fn new(semaphore: Arc<TimelineSemaphore>, wait_value: u64, inner_value: T) -> Self {
        Self {
            wait: SemaphoreDeferredValueWait {
                semaphore,
                wait_value,
            },
            value: inner_value,
        }
    }
    pub fn wait_value(&self) -> u64 {
        self.wait.wait_value
    }
    pub fn current_value(&self) -> u64 {
        self.wait.semaphore.value()
    }
}

pub struct Event {
    device: Device,
    raw: vk::Event,
    device_only: bool,
}
impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_event(self.raw, None);
        }
    }
}
impl Event {
    pub fn new(device: Device) -> VkResult<Self> {
        let event = unsafe { device.create_event(&vk::EventCreateInfo::default(), None)? };
        Ok(Self {
            device,
            raw: event,
            device_only: false,
        })
    }

    pub fn new_device_only(device: Device) -> VkResult<Self> {
        let event = unsafe {
            device.create_event(
                &vk::EventCreateInfo {
                    flags: vk::EventCreateFlags::DEVICE_ONLY,
                    ..Default::default()
                },
                None,
            )?
        };
        Ok(Self {
            device,
            raw: event,
            device_only: true,
        })
    }

    pub fn set(&mut self) -> VkResult<()> {
        assert!(!self.device_only);
        unsafe { self.device.set_event(self.raw) }
    }
    pub fn reset(&mut self) -> VkResult<()> {
        assert!(!self.device_only);
        unsafe { self.device.reset_event(self.raw) }
    }
    pub fn get_status(&self) -> VkResult<bool> {
        assert!(!self.device_only);
        unsafe { self.device.get_event_status(self.raw) }
    }
}
