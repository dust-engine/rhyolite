use std::{future::Future, sync::Arc};

use ash::{prelude::VkResult, vk};

use crate::Device;

pub struct Semaphore {
    device: Arc<Device>,
    pub(crate) semaphore: vk::Semaphore,
}

impl crate::HasDevice for Semaphore {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl crate::debug::DebugObject for Semaphore {
    const OBJECT_TYPE: vk::ObjectType = vk::ObjectType::SEMAPHORE;
    fn object_handle(&mut self) -> u64 {
        unsafe { std::mem::transmute(self.semaphore) }
    }
}

impl std::fmt::Debug for Semaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Semaphore({:?})", self.semaphore))
    }
}

impl Semaphore {
    pub fn new(device: Arc<Device>) -> VkResult<Self> {
        let create_info = vk::SemaphoreCreateInfo::default();
        let semaphore = unsafe { device.create_semaphore(&create_info, None)? };
        Ok(Self { device, semaphore })
    }
    pub unsafe fn as_timeline_arc(self: Arc<Self>) -> Arc<TimelineSemaphore> {
        std::mem::transmute(self)
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        tracing::debug!(semaphore = ?self.semaphore, "drop semaphore");
        // Safety: Host access to semaphore must be externally synchronized
        // We have &mut self thus exclusive access to self.semaphore
        unsafe {
            self.device.destroy_semaphore(self.semaphore, None);
        }
    }
}

#[repr(transparent)]
pub struct TimelineSemaphore(Semaphore);

impl std::fmt::Debug for TimelineSemaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("TimelineSemaphore({:?})", self.0.semaphore))
    }
}

impl crate::HasDevice for TimelineSemaphore {
    fn device(&self) -> &Arc<Device> {
        self.0.device()
    }
}

impl TimelineSemaphore {
    pub fn new(device: Arc<Device>, initial_value: u64) -> VkResult<Self> {
        let type_info = vk::SemaphoreTypeCreateInfo::builder()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value)
            .build();
        let create_info = vk::SemaphoreCreateInfo {
            p_next: &type_info as *const _ as *const std::ffi::c_void,
            ..Default::default()
        };
        let semaphore = unsafe { device.create_semaphore(&create_info, None)? };
        Ok(TimelineSemaphore(Semaphore { device, semaphore }))
    }
    pub fn signal(&self, value: u64) -> VkResult<()> {
        unsafe {
            self.0.device.signal_semaphore(&vk::SemaphoreSignalInfo {
                semaphore: self.0.semaphore,
                value,
                ..Default::default()
            })
        }
    }
    pub fn value(&self) -> VkResult<u64> {
        unsafe { self.0.device.get_semaphore_counter_value(self.0.semaphore) }
    }
    /// Block the current thread until the semaphore reaches (>=) the given value
    pub fn block(self: &TimelineSemaphore, value: u64) -> VkResult<()> {
        unsafe {
            self.0.device.wait_semaphores(
                &vk::SemaphoreWaitInfo {
                    semaphore_count: 1,
                    p_semaphores: &self.0.semaphore,
                    p_values: &value,
                    ..Default::default()
                },
                std::u64::MAX,
            )
        }
    }
    /// Return a future that resolves after the semaphore reaches (>=) the given value
    pub fn wait(self: Arc<TimelineSemaphore>, value: u64) -> blocking::Task<VkResult<()>> {
        blocking::unblock(move || {
            self.block(value)?;
            drop(self);
            Ok(())
        })
    }
    /// Downgrade an Arc<TimelineSemaphore> into an Arc<Semaphore>.
    pub fn downgrade_arc(self: Arc<TimelineSemaphore>) -> Arc<Semaphore> {
        unsafe {
            // Safety: This relies on TimelineSemaphore being transmutable to Semaphore.
            std::mem::transmute(self)
        }
    }
}

/// A semaphore signal / wait operation.
/// This could be a binary semaphore or a timeline semaphore.
#[derive(Clone)]
pub struct SemaphoreOp {
    pub semaphore: Arc<Semaphore>,
    pub value: u64,
}

impl SemaphoreOp {
    pub fn staged(self, stage: vk::PipelineStageFlags2) -> StagedSemaphoreOp {
        StagedSemaphoreOp {
            semaphore: self.semaphore,
            stage_mask: stage,
            value: self.value,
        }
    }
    pub fn increment(self) -> Self {
        Self {
            semaphore: self.semaphore,
            value: self.value + 1,
        }
    }
    pub fn is_timeline(&self) -> bool {
        // Because the semaphore value is always >= 0, signaling a semaphore to be 0
        // or waiting a semaphore to turn 0 is meaningless.
        self.value != 0
    }
    pub fn as_timeline(self) -> TimelineSemaphoreOp {
        assert!(self.is_timeline());
        TimelineSemaphoreOp {
            semaphore: unsafe { self.semaphore.as_timeline_arc() },
            value: self.value,
        }
    }
}

/// A semaphore signal / wait operation with a pipeline stage mask.
/// Usually used for GPU-side syncronization. This could be a binary semaphore or a timeline semaphore.
///
/// stage_mask in wait semaphores: Block the execution of these stages until the semaphore was signaled.
/// Stages not specified in wait_stages can proceed before the semaphore signal operation.
///
/// stage_mask in signal semaphores: Block the semaphore signal operation on the completion of these stages.
/// The semaphore will be signaled even if other stages are still running.
#[derive(Clone)]
pub struct StagedSemaphoreOp {
    pub semaphore: Arc<Semaphore>,
    pub stage_mask: vk::PipelineStageFlags2,
    // When value == 0, the `StagedSemaphoreOp` is a binary semaphore.
    pub value: u64,
}
impl std::fmt::Debug for StagedSemaphoreOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = if self.is_timeline() {
            "StagedSemaphoreOp(Timeline)"
        } else {
            "StagedSemaphoreOp(Binary)"
        };
        let mut res = f.debug_tuple(name);
        res.field(&self.semaphore.semaphore).field(&self.stage_mask);
        if self.is_timeline() {
            res.field(&self.value);
        }
        res.finish()
    }
}

impl StagedSemaphoreOp {
    pub fn binary(semaphore: Arc<Semaphore>, stage_mask: vk::PipelineStageFlags2) -> Self {
        StagedSemaphoreOp {
            semaphore,
            stage_mask,
            value: 0,
        }
    }
    pub fn timeline(
        semaphore: Arc<Semaphore>,
        stage_mask: vk::PipelineStageFlags2,
        value: u64,
    ) -> Self {
        StagedSemaphoreOp {
            semaphore,
            stage_mask,
            value,
        }
    }
    pub fn is_timeline(&self) -> bool {
        // Because the semaphore value is always >= 0, signaling a semaphore to be 0
        // or waiting a semaphore to turn 0 is meaningless.
        self.value != 0
    }
    pub fn stageless(self) -> SemaphoreOp {
        SemaphoreOp {
            semaphore: self.semaphore,
            value: self.value,
        }
    }
    pub fn increment(self) -> Self {
        Self {
            semaphore: self.semaphore,
            value: self.value + 1,
            stage_mask: self.stage_mask,
        }
    }
}

/// A timeline semaphore capable of host-side syncronization.
#[derive(Clone)]
pub struct TimelineSemaphoreOp {
    pub semaphore: Arc<TimelineSemaphore>,
    pub value: u64,
}

impl std::fmt::Debug for TimelineSemaphoreOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "TimelineSemaphore({:?})[{}]",
            self.semaphore.0.semaphore, self.value
        ))
    }
}

impl TimelineSemaphoreOp {
    pub fn block(self) -> VkResult<()> {
        self.semaphore.block(self.value)
    }
    pub fn wait(self) -> impl Future<Output = VkResult<()>> + Unpin {
        self.semaphore.wait(self.value)
    }

    pub fn block_n<const N: usize>(semaphores: [&TimelineSemaphoreOp; N]) -> VkResult<()> {
        let device = semaphores[0].semaphore.0.device.clone();
        // Ensure that all semaphores have the same device
        for op in semaphores.iter().skip(1) {
            assert_eq!(op.semaphore.0.device.handle(), device.handle());
        }
        let semaphore_values = semaphores.map(|s| s.semaphore.0.semaphore);
        let values = semaphores.map(|s| s.value);
        unsafe {
            device.wait_semaphores(
                &vk::SemaphoreWaitInfo {
                    semaphore_count: N as u32,
                    p_semaphores: semaphore_values.as_ptr(),
                    p_values: values.as_ptr(),
                    ..Default::default()
                },
                std::u64::MAX,
            )
        }
    }
    pub fn wait_n<const N: usize>(
        semaphores: [TimelineSemaphoreOp; N],
    ) -> impl Future<Output = VkResult<()>> {
        blocking::unblock(move || {
            Self::block_n(semaphores.each_ref())?;
            drop(semaphores);
            Ok(())
        })
    }
    pub fn block_many(semaphores: &[&TimelineSemaphoreOp]) -> VkResult<()> {
        if semaphores.len() == 0 {
            return Ok(());
        }
        let device = semaphores[0].semaphore.0.device.clone();
        // Ensure that all semaphores have the same device
        for op in semaphores.iter().skip(1) {
            assert_eq!(op.semaphore.0.device.handle(), device.handle());
        }
        let semaphore_values: Vec<_> = semaphores.iter().map(|s| s.semaphore.0.semaphore).collect();
        let values: Vec<_> = semaphores.iter().map(|s| s.value).collect();
        unsafe {
            device.wait_semaphores(
                &vk::SemaphoreWaitInfo {
                    semaphore_count: semaphore_values.len() as u32,
                    p_semaphores: semaphore_values.as_ptr(),
                    p_values: values.as_ptr(),
                    ..Default::default()
                },
                std::u64::MAX,
            )
        }
    }

    pub fn wait_many(semaphores: Vec<TimelineSemaphoreOp>) -> impl Future<Output = VkResult<()>> {
        // Ensure that all semaphores have the same device
        blocking::unblock(move || {
            let refs: Vec<_> = semaphores.iter().collect();
            Self::block_many(&refs)?;
            drop(semaphores);
            Ok(())
        })
    }
    pub fn signal(&self) -> VkResult<()> {
        self.semaphore.signal(self.value)
    }
    pub fn finished(&self) -> VkResult<bool> {
        let val = self.semaphore.value()?;
        Ok(val >= self.value)
    }
    pub fn downgrade_arc(self) -> SemaphoreOp {
        SemaphoreOp {
            value: self.value,
            semaphore: self.semaphore.downgrade_arc(),
        }
    }
    pub fn increment(self) -> Self {
        Self {
            semaphore: self.semaphore,
            value: self.value + 1,
        }
    }
}
