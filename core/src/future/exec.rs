use crate::{
    commands::{CommandBufferLike, SharedCommandPool},
    HasDevice, ImageLike, QueueRef,
};
use ash::vk;
use std::{
    cell::RefCell, collections::BTreeMap, marker::PhantomData, mem::ManuallyDrop, ops::Range,
    pin::Pin, task::Poll,
};

use super::{Disposable, GPUCommandFuture};

pub struct ResTrackingInfo {
    pub prev_stage_access: Access,
    pub current_stage_access: Access,
    pub last_accessed_stage_index: u32,

    pub queue_family: u32,
    pub queue_index: QueueRef,
    pub prev_queue_family: u32,
    pub prev_queue_index: QueueRef,
    pub last_accessed_timeline: u32,

    pub untracked_semaphore: Option<vk::Semaphore>,
}
impl Default for ResTrackingInfo {
    fn default() -> Self {
        Self {
            prev_stage_access: Access::default(),
            current_stage_access: Access::default(),
            last_accessed_stage_index: 0,

            queue_family: vk::QUEUE_FAMILY_IGNORED,
            queue_index: QueueRef::null(),
            prev_queue_index: QueueRef::null(),
            prev_queue_family: vk::QUEUE_FAMILY_IGNORED,
            last_accessed_timeline: 0,

            untracked_semaphore: None,
        }
    }
}

pub struct Res<T> {
    pub tracking_info: ManuallyDrop<RefCell<ResTrackingInfo>>,
    pub inner: ManuallyDrop<T>,
}
impl<T> Disposable for Res<T> {
    fn dispose(mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.tracking_info);
            ManuallyDrop::drop(&mut self.inner);
            std::mem::forget(self)
        }
    }
}
impl<T> Drop for Res<T> {
    fn drop(&mut self) {
        if std::mem::needs_drop::<T>() {
            panic!("Res<{}> must be disposed!", std::any::type_name::<T>());
        }
    }
}
// TODO: Make Res error out when not retained / disposed properly.
impl<T> Res<T> {
    pub fn new(inner: T) -> Self {
        Self {
            tracking_info: Default::default(),
            inner: ManuallyDrop::new(inner),
        }
    }
    pub fn inner(&self) -> &T {
        &self.inner
    }
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    pub fn map<RET>(mut self, mapper: impl FnOnce(T) -> RET) -> Res<RET> {
        let (inner, tracking) = unsafe {
            let inner = ManuallyDrop::take(&mut self.inner);
            let tracking = ManuallyDrop::take(&mut self.tracking_info);
            std::mem::forget(self);
            (inner, tracking)
        };
        Res {
            inner: ManuallyDrop::new((mapper)(inner)),
            tracking_info: ManuallyDrop::new(tracking),
        }
    }
}

pub struct ResImage<T> {
    pub res: Res<T>,
    old_layout: vk::ImageLayout,
    layout: vk::ImageLayout,
}
impl<T> Disposable for ResImage<T> {
    fn dispose(self) {
        self.res.dispose()
    }
}
impl<T> ResImage<T> {
    pub fn new(inner: T, initial_layout: vk::ImageLayout) -> Self {
        Self {
            res: Res::new(inner),
            layout: initial_layout,
            old_layout: initial_layout,
        }
    }
    pub fn inner(&self) -> &T {
        &self.res.inner
    }
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.res.inner
    }

    pub fn map<RET>(self, mapper: impl FnOnce(T) -> RET) -> ResImage<RET> {
        let res = self.res.map(mapper);
        ResImage {
            res,
            old_layout: self.old_layout,
            layout: self.layout,
        }
    }
}

/// One per command buffer record call. If multiple command buffers were merged together on the queue level,
/// this would be the same.
pub struct CommandBufferRecordContext<'a> {
    // perhaps also a reference to the command buffer allocator
    pub stage_index: u32,
    pub timeline_index: u32,
    pub queue: QueueRef,
    pub command_buffers: &'a mut Vec<vk::CommandBuffer>,
    pub recording_command_buffer: &'a mut Option<vk::CommandBuffer>,
    pub command_pool: &'a mut SharedCommandPool,
}
impl<'a> HasDevice for CommandBufferRecordContext<'a> {
    fn device(&self) -> &std::sync::Arc<crate::Device> {
        self.command_pool.device()
    }
}
impl<'a> CommandBufferRecordContext<'a> {
    pub fn queue_family_index(&self) -> u32 {
        self.command_pool.queue_family_index()
    }
    /// Immediatly record a command buffer, allocated from the shared command pool.
    pub fn record(&mut self, callback: impl FnOnce(&Self, vk::CommandBuffer)) {
        let command_buffer = if let Some(command_buffer) = self.recording_command_buffer.take() {
            command_buffer
        } else {
            let buffer = self.command_pool.allocate_one();
            unsafe {
                self.device()
                    .begin_command_buffer(
                        buffer,
                        &vk::CommandBufferBeginInfo {
                            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                            ..Default::default()
                        },
                    )
                    .unwrap();
            }
            buffer
        };
        callback(self, command_buffer);
        *self.recording_command_buffer = Some(command_buffer);
    }
    pub fn add_command_buffer<T: CommandBufferLike + 'a>(&mut self, buffer: T) {
        if let Some(command_buffer) = self.recording_command_buffer.take() {
            unsafe {
                self.device().end_command_buffer(command_buffer).unwrap();
            }
            self.command_buffers.push(command_buffer);
        }
        self.command_buffers.push(buffer.raw_command_buffer());
    }
}

pub struct CommandBufferRecordContextInner<'host, 'retain> {
    pub ctx: &'host mut CommandBufferRecordContext<'host>,
    pub _marker: PhantomData<&'retain ()>,
}

impl<'host> CommandBufferRecordContext<'host> {
    pub fn current_stage_index(&self) -> u32 {
        self.stage_index
    }
}
impl<'host, 'retain> CommandBufferRecordContextInner<'host, 'retain> {
    pub fn current_stage_index(&self) -> u32 {
        self.ctx.stage_index
    }
    pub unsafe fn new(ptr: *mut ()) -> Self {
        Self {
            ctx: &mut *(ptr as *mut _),
            _marker: PhantomData,
        }
    }
    /// Perserve the lifetimes
    pub unsafe fn update(old: Self, ptr: *mut ()) -> Self {
        Self {
            ctx: &mut *(ptr as *mut _),
            _marker: old._marker,
        }
    }
    pub unsafe fn add_res<T>(&mut self, res: T) -> Res<T> {
        // Extend the lifetime of res so that it lives as long as 'retain
        Res::new(res)
    }
    pub unsafe fn add_image<T: ImageLike>(
        &mut self,
        res: T,
        initial_layout: vk::ImageLayout,
    ) -> ResImage<T> {
        ResImage::new(res, initial_layout)
    }
}

struct StageContextImage {
    image: vk::Image,
    subresource_range: vk::ImageSubresourceRange,
    extent: vk::Extent3D,
}
impl ImageLike for StageContextImage {
    fn raw_image(&self) -> vk::Image {
        self.image
    }

    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        self.subresource_range
    }

    fn extent(&self) -> vk::Extent3D {
        self.extent
    }
}

impl PartialEq for StageContextImage {
    fn eq(&self, other: &Self) -> bool {
        self.image == other.image
            && self.subresource_range.aspect_mask == other.subresource_range.aspect_mask
            && self.subresource_range.base_array_layer == other.subresource_range.base_array_layer
            && self.subresource_range.base_mip_level == other.subresource_range.base_mip_level
            && self.subresource_range.level_count == other.subresource_range.level_count
            && self.subresource_range.layer_count == other.subresource_range.layer_count
    }
}
impl Eq for StageContextImage {}

impl PartialOrd for StageContextImage {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for StageContextImage {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.image.cmp(&other.image) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        match self
            .subresource_range
            .base_array_layer
            .cmp(&other.subresource_range.base_array_layer)
        {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        match self
            .subresource_range
            .layer_count
            .cmp(&other.subresource_range.layer_count)
        {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        match self
            .subresource_range
            .base_mip_level
            .cmp(&other.subresource_range.base_mip_level)
        {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        match self
            .subresource_range
            .level_count
            .cmp(&other.subresource_range.level_count)
        {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        self.subresource_range
            .aspect_mask
            .cmp(&other.subresource_range.aspect_mask)
    }
}

struct StageContextBuffer {
    buffer: vk::Buffer,
    range: Range<vk::DeviceSize>,
}
impl PartialEq for StageContextBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.buffer == other.buffer && self.range == other.range
    }
}
impl Eq for StageContextBuffer {}
impl PartialOrd for StageContextBuffer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for StageContextBuffer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.range.start.cmp(&other.range.start) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        match self.range.end.cmp(&other.range.start) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        self.buffer.cmp(&other.buffer)
    }
}

pub enum StageContextSemaphoreTransition {
    Managed {
        src_queue: QueueRef,
        dst_queue: QueueRef,
        src_stages: vk::PipelineStageFlags2,
        dst_stages: vk::PipelineStageFlags2,
    },
    Untracked {
        semaphore: vk::Semaphore,
        dst_queue: QueueRef,
        dst_stages: vk::PipelineStageFlags2,
    },
}

pub struct StageContext {
    stage_index: u32,
    timeline_index: u32,
    queue_family_index: u32,
    queue_index: QueueRef,
    global_access: vk::MemoryBarrier2,
    image_accesses:
        BTreeMap<StageContextImage, (vk::MemoryBarrier2, vk::ImageLayout, vk::ImageLayout)>,

    // Queue, srcQueue, dstQueue, srcStages, dstStages
    pub semaphore_transitions: Vec<StageContextSemaphoreTransition>,
}

impl StageContext {
    pub fn new(
        stage_index: u32,
        timeline_index: u32,
        queue_family_index: u32,
        queue_index: QueueRef,
    ) -> Self {
        Self {
            stage_index,
            queue_family_index,
            queue_index,
            timeline_index,
            global_access: vk::MemoryBarrier2::default(),
            image_accesses: BTreeMap::new(),
            semaphore_transitions: Vec::new(),
        }
    }
    fn add_barrier_tracking(&mut self, tracking: &mut ResTrackingInfo, access: &Access) {
        if tracking.last_accessed_timeline < self.timeline_index {
            tracking.prev_queue_family =
                std::mem::replace(&mut tracking.queue_family, self.queue_family_index);
            tracking.prev_queue_index =
                std::mem::replace(&mut tracking.queue_index, self.queue_index);
            // Need semaphore sync
            // queue, timeline: signal at pipeline barrier.
            // If an earlier stage was already signaled we need to make another signal.
            // If a later stage was already signaled, we can
            // ok this is very problematic.

            // In the binary semaphore model, each semaphore can only be waited on once.
            // This is great for our purpose.
            // We can say unconditionally: This queue, signal on this stage. (what is "this stage?")
            tracking.last_accessed_timeline = self.timeline_index;

            if let Some(untracked_semaphore) = tracking.untracked_semaphore.as_mut() {
                if *untracked_semaphore == vk::Semaphore::null() {
                    panic!("Attempts to wait on vk::AcquireNextImageKHR twice.");
                }

                let mut barrier = vk::MemoryBarrier2::default();
                get_memory_access(&mut barrier, &tracking.prev_stage_access, access);

                self.semaphore_transitions
                    .push(StageContextSemaphoreTransition::Untracked {
                        semaphore: *untracked_semaphore,
                        dst_queue: tracking.queue_index,
                        dst_stages: barrier.dst_stage_mask,
                    });

                *untracked_semaphore = vk::Semaphore::null();
            } else if tracking.prev_queue_family != vk::QUEUE_FAMILY_IGNORED {
                let mut barrier = vk::MemoryBarrier2::default();
                get_memory_access(&mut barrier, &tracking.prev_stage_access, access);
                self.semaphore_transitions
                    .push(StageContextSemaphoreTransition::Managed {
                        src_queue: tracking.prev_queue_index,
                        dst_queue: tracking.queue_index,
                        src_stages: barrier.src_stage_mask,
                        dst_stages: barrier.dst_stage_mask,
                    });
            }
        }
        if tracking.prev_queue_family != self.queue_family_index {
            // Need queue family transfer.
        }
    }
    /// Declare a global memory write
    #[inline]
    pub fn write<T>(
        &mut self,
        res: &mut Res<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
    ) {
        let access = Access {
            write_access: accesses,
            write_stages: stages,
            ..Default::default()
        };

        let mut tracking = res.tracking_info.borrow_mut();
        if tracking.last_accessed_stage_index < self.stage_index
            || tracking.last_accessed_timeline < self.timeline_index
        {
            tracking.prev_stage_access = std::mem::take(&mut tracking.current_stage_access);
        }
        self.add_barrier_tracking(&mut tracking, &access);
        get_memory_access(
            &mut self.global_access,
            &tracking.prev_stage_access,
            &access,
        );
        tracking.current_stage_access.write_access |= accesses;
        tracking.current_stage_access.write_stages |= stages;
        tracking.last_accessed_stage_index = self.stage_index;
    }
    /// Declare a global memory read
    #[inline]
    pub fn read<T>(
        &mut self,
        res: &Res<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
    ) {
        let access = Access {
            read_access: accesses,
            read_stages: stages,
            ..Default::default()
        };
        let mut tracking = res.tracking_info.borrow_mut();
        if tracking.last_accessed_stage_index < self.stage_index
            || tracking.last_accessed_timeline < self.timeline_index
        {
            tracking.prev_stage_access = std::mem::take(&mut tracking.current_stage_access);
        }
        self.add_barrier_tracking(&mut tracking, &access);
        get_memory_access(
            &mut self.global_access,
            &tracking.prev_stage_access,
            &access,
        );
        tracking.current_stage_access.read_access |= accesses;
        tracking.current_stage_access.read_stages |= stages;
        tracking.last_accessed_stage_index = self.stage_index;
    }
    #[inline]
    pub fn write_image<T>(
        &mut self,
        res: &mut ResImage<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
        layout: vk::ImageLayout,
    ) where
        T: ImageLike,
    {
        let access = Access {
            write_access: accesses,
            write_stages: stages,
            ..Default::default()
        };

        let mut tracking = res.res.tracking_info.borrow_mut();
        if tracking.last_accessed_stage_index < self.stage_index
            || tracking.last_accessed_timeline < self.timeline_index
        {
            tracking.prev_stage_access = std::mem::take(&mut tracking.current_stage_access);
            res.old_layout = std::mem::replace(&mut res.layout, layout);
        } else {
            assert_eq!(
                tracking.queue_family, self.queue_family_index,
                "Layout mismatch."
            );
            assert_eq!(res.layout, layout, "Layout mismatch.");
        }
        self.add_barrier_tracking(&mut tracking, &access);

        let (barrier, _old_layout, _new_layout) = self
            .image_accesses
            .entry(StageContextImage {
                image: res.inner().raw_image(),
                subresource_range: res.inner().subresource_range(),
                extent: res.inner().extent(),
            })
            .or_insert((Default::default(), res.old_layout, layout));
        get_memory_access(barrier, &tracking.prev_stage_access, &access);

        tracking.current_stage_access.write_access |= accesses;
        tracking.current_stage_access.write_stages |= stages;
        tracking.last_accessed_stage_index = self.stage_index;
    }
    /// Declare a global memory read
    #[inline]
    pub fn read_image<T>(
        &mut self,
        res: &mut ResImage<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
        layout: vk::ImageLayout,
    ) where
        T: ImageLike,
    {
        let access = Access {
            read_access: accesses,
            read_stages: stages,
            ..Default::default()
        };
        let mut tracking = res.res.tracking_info.borrow_mut();
        if tracking.last_accessed_stage_index < self.stage_index
            || tracking.last_accessed_timeline < self.timeline_index
        {
            tracking.prev_queue_family =
                std::mem::replace(&mut tracking.queue_family, self.queue_family_index);
            tracking.prev_stage_access = std::mem::take(&mut tracking.current_stage_access);
            res.old_layout = std::mem::replace(&mut res.layout, layout);
        } else {
            assert_eq!(
                tracking.queue_family, self.queue_family_index,
                "Queue family mismatch."
            );
            assert_eq!(res.layout, layout, "Layout mismatch.");
        }
        self.add_barrier_tracking(&mut tracking, &access);

        let (barrier, _old_layout, _new_layout) = self
            .image_accesses
            .entry(StageContextImage {
                image: res.inner().raw_image(),
                subresource_range: res.inner().subresource_range(),
                extent: res.inner().extent(),
            })
            .or_insert((Default::default(), res.old_layout, layout));
        get_memory_access(barrier, &tracking.prev_stage_access, &access);

        tracking.current_stage_access.read_access |= accesses;
        tracking.current_stage_access.read_stages |= stages;
        tracking.last_accessed_stage_index = self.stage_index;
    }
}

impl<'a> CommandBufferRecordContext<'a> {
    pub fn record_first_step<T: GPUCommandFuture>(
        &mut self,
        mut fut: Pin<&'a mut T>,
        recycled_state: &mut T::RecycledState,
        context_handler: impl FnOnce(&StageContext),
    ) -> Poll<(T::Output, T::RetainedState)> {
        let mut next_stage = StageContext::new(
            self.stage_index,
            self.timeline_index,
            self.command_pool.queue_family_index(),
            self.queue,
        );
        fut.as_mut().context(&mut next_stage);
        (context_handler)(&next_stage);

        Self::add_barrier(&next_stage, |dependency_info| {
            self.record(|ctx, command_buffer| unsafe {
                ctx.device()
                    .cmd_pipeline_barrier2(command_buffer, dependency_info);
            });
        });
        let ret = fut.as_mut().record(self, recycled_state);
        ret
    }

    pub fn record_one_step<T: GPUCommandFuture>(
        &mut self,
        mut fut: Pin<&'a mut T>,
        recycled_state: &mut T::RecycledState,
    ) -> Poll<(T::Output, T::RetainedState)> {
        let mut next_stage = StageContext::new(
            self.stage_index,
            self.timeline_index,
            self.command_pool.queue_family_index(),
            self.queue,
        );
        fut.as_mut().context(&mut next_stage);
        assert_eq!(next_stage.semaphore_transitions.len(), 0);
        Self::add_barrier(&next_stage, |dependency_info| {
            self.record(|ctx, command_buffer| unsafe {
                ctx.device()
                    .cmd_pipeline_barrier2(command_buffer, dependency_info);
            });
        });

        let ret = fut.as_mut().record(self, recycled_state);
        ret
    }
    fn add_barrier(
        next_stage: &StageContext,
        cmd_pipeline_barrier: impl FnOnce(&vk::DependencyInfo),
    ) {
        let mut global_memory_barrier = next_stage.global_access;
        let mut image_barrier: Vec<vk::ImageMemoryBarrier2> = Vec::new();

        // Set the global memory barrier.

        for (image, (barrier, old_layout, new_layout)) in next_stage.image_accesses.iter() {
            if old_layout == new_layout {
                global_memory_barrier.dst_access_mask |= barrier.dst_access_mask;
                global_memory_barrier.src_access_mask |= barrier.src_access_mask;
                global_memory_barrier.dst_stage_mask |= barrier.dst_stage_mask;
                global_memory_barrier.src_stage_mask |= barrier.src_stage_mask;
            } else {
                println!("Needs image layout transfers");
                // Needs image layout transfer.
                let mut o = vk::ImageMemoryBarrier2 {
                    src_access_mask: barrier.src_access_mask,
                    src_stage_mask: barrier.src_stage_mask,
                    dst_access_mask: barrier.dst_access_mask,
                    dst_stage_mask: barrier.dst_stage_mask,
                    image: image.image,
                    subresource_range: image.subresource_range,
                    old_layout: *old_layout,
                    new_layout: *new_layout,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    ..Default::default()
                };
                if barrier.src_access_mask.is_empty() {
                    o.old_layout = vk::ImageLayout::UNDEFINED;
                }
                image_barrier.push(o);
            }
        }
        let mut dep = vk::DependencyInfo {
            dependency_flags: vk::DependencyFlags::BY_REGION, // TODO
            ..Default::default()
        };
        if !global_memory_barrier.dst_access_mask.is_empty()
            || !global_memory_barrier.src_access_mask.is_empty()
            || !global_memory_barrier.dst_stage_mask.is_empty()
            || !global_memory_barrier.src_stage_mask.is_empty()
        {
            dep.memory_barrier_count = 1;
            dep.p_memory_barriers = &global_memory_barrier;
        }
        if !image_barrier.is_empty() {
            dep.image_memory_barrier_count = image_barrier.len() as u32;
            dep.p_image_memory_barriers = image_barrier.as_ptr();
        }

        if dep.memory_barrier_count > 0
            || dep.buffer_memory_barrier_count > 0
            || dep.image_memory_barrier_count > 0
        {
            cmd_pipeline_barrier(&dep);
        }
    }
}

#[derive(Clone, Default, Debug)]
pub struct Access {
    pub read_stages: vk::PipelineStageFlags2,
    pub read_access: vk::AccessFlags2,
    pub write_stages: vk::PipelineStageFlags2,
    pub write_access: vk::AccessFlags2,
}

impl Access {
    pub fn has_read(&self) -> bool {
        !self.read_stages.is_empty()
    }
    pub fn has_write(&self) -> bool {
        !self.write_stages.is_empty()
    }
}

fn get_memory_access(
    memory_barrier: &mut vk::MemoryBarrier2,
    before_access: &Access,
    after_access: &Access,
) {
    if before_access.has_write() && after_access.has_write() {
        // Write after write
        memory_barrier.src_stage_mask |= before_access.write_stages;
        memory_barrier.dst_stage_mask |= after_access.write_stages;
        memory_barrier.src_access_mask |= before_access.write_access;
        memory_barrier.dst_access_mask |= after_access.write_access;
    }
    if before_access.has_read() && after_access.has_write() {
        // Write after read
        memory_barrier.src_stage_mask |= before_access.read_stages;
        memory_barrier.dst_stage_mask |= after_access.write_stages;
        // No need for memory barrier
    }
    if before_access.has_write() && after_access.has_read() {
        // Read after write
        memory_barrier.src_stage_mask |= before_access.write_stages;
        memory_barrier.dst_stage_mask |= after_access.read_stages;
        memory_barrier.src_access_mask |= before_access.write_access;
        memory_barrier.dst_access_mask |= after_access.read_access;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn assert_global(
        dep: &vk::DependencyInfo,
        src_stage_mask: vk::PipelineStageFlags2,
        src_access_mask: vk::AccessFlags2,
        dst_stage_mask: vk::PipelineStageFlags2,
        dst_access_mask: vk::AccessFlags2,
    ) {
        assert_eq!(dep.memory_barrier_count, 1);
        assert_eq!(dep.buffer_memory_barrier_count, 0);
        assert_eq!(dep.image_memory_barrier_count, 0);
        let memory_barrier = unsafe { &*dep.p_memory_barriers };
        assert_eq!(memory_barrier.src_stage_mask, src_stage_mask);
        assert_eq!(memory_barrier.src_access_mask, src_access_mask);
        assert_eq!(memory_barrier.dst_stage_mask, dst_stage_mask);
        assert_eq!(memory_barrier.dst_access_mask, dst_access_mask);
        assert_ne!(memory_barrier.src_stage_mask, vk::PipelineStageFlags2::NONE);
        assert_ne!(memory_barrier.dst_stage_mask, vk::PipelineStageFlags2::NONE);
    }

    use vk::AccessFlags2 as A;
    use vk::PipelineStageFlags2 as S;
    enum ReadWrite {
        Read(vk::PipelineStageFlags2, vk::AccessFlags2),
        Write(vk::PipelineStageFlags2, vk::AccessFlags2),
        ReadWrite(Access),
    }
    impl ReadWrite {
        fn stage<T>(&self, stage_ctx: &mut StageContext, res: &mut Res<T>) {
            match &self {
                ReadWrite::Read(stage, access) => stage_ctx.read(res, *stage, *access),
                ReadWrite::Write(stage, access) => stage_ctx.write(res, *stage, *access),
                ReadWrite::ReadWrite(access) => {
                    stage_ctx.read(res, access.read_stages, access.read_access);
                    stage_ctx.write(res, access.write_stages, access.write_access);
                }
            }
        }
    }

    fn make_stage(stage_index: u32) -> StageContext {
        StageContext::new(stage_index, 0, vk::QUEUE_FAMILY_IGNORED, QueueRef(0))
    }

    #[test]
    fn c2c_global_tests() {
        let test_cases = [
            (
                [
                    ReadWrite::Read(S::COMPUTE_SHADER, A::SHADER_STORAGE_READ),
                    ReadWrite::Read(S::COMPUTE_SHADER, A::SHADER_READ),
                ],
                (S::NONE, A::NONE, S::NONE, A::NONE),
            ), // RaR
            (
                [
                    ReadWrite::Read(S::COMPUTE_SHADER, A::SHADER_STORAGE_READ),
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                ],
                (S::COMPUTE_SHADER, A::NONE, S::COMPUTE_SHADER, A::NONE),
            ), // WaR
            (
                [
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_READ),
                    ReadWrite::Read(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                ],
                (
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_READ,
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                ),
            ), // RaW
            (
                [
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                ],
                (
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                ),
            ), // WaW
            (
                [
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                    ReadWrite::Read(S::INDEX_INPUT, A::INDEX_READ),
                ],
                (
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                    S::INDEX_INPUT,
                    A::INDEX_READ,
                ),
            ), // Dispatch writes into a storage buffer. Draw consumes that buffer as an index buffer.
        ];

        let mut buffer: vk::Buffer = unsafe { std::mem::transmute(123_u64) };
        for test_case in test_cases.into_iter() {
            let mut buffer = Res::new(&mut buffer);
            let mut stage1 = make_stage(0);
            test_case.0[0].stage(&mut stage1, &mut buffer);

            let mut stage2 = make_stage(1);
            test_case.0[1].stage(&mut stage2, &mut buffer);

            let mut called = false;
            CommandBufferRecordContext::add_barrier(&stage2, |dep| {
                called = true;
                assert_global(
                    dep,
                    test_case.1 .0,
                    test_case.1 .1,
                    test_case.1 .2,
                    test_case.1 .3,
                );
            });
            if test_case.1 .0 != vk::PipelineStageFlags2::NONE
                && test_case.1 .2 != vk::PipelineStageFlags2::NONE
            {
                assert!(called);
            } else {
                assert!(!called);
            }
        }
    }

    #[test]
    fn c2c_global_carryover_tests() {
        let test_cases = [
            (
                [
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                    ReadWrite::ReadWrite(Access {
                        read_stages: S::COMPUTE_SHADER,
                        read_access: A::SHADER_STORAGE_READ,
                        write_stages: S::COMPUTE_SHADER,
                        write_access: A::SHADER_STORAGE_WRITE,
                    }),
                    ReadWrite::Read(S::COMPUTE_SHADER, A::SHADER_STORAGE_READ),
                ],
                [
                    (
                        S::COMPUTE_SHADER,
                        A::SHADER_STORAGE_WRITE,
                        S::COMPUTE_SHADER,
                        A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
                    ),
                    (
                        S::COMPUTE_SHADER,
                        A::SHADER_STORAGE_WRITE,
                        S::COMPUTE_SHADER,
                        A::SHADER_STORAGE_READ,
                    ),
                ],
            ),
            (
                [
                    ReadWrite::Read(S::COMPUTE_SHADER, A::SHADER_STORAGE_READ),
                    ReadWrite::ReadWrite(Access {
                        read_stages: S::COMPUTE_SHADER,
                        read_access: A::SHADER_STORAGE_READ,
                        write_stages: S::COMPUTE_SHADER,
                        write_access: A::SHADER_STORAGE_WRITE,
                    }),
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                ],
                [
                    (S::COMPUTE_SHADER, A::empty(), S::COMPUTE_SHADER, A::empty()),
                    (
                        S::COMPUTE_SHADER,
                        A::SHADER_STORAGE_WRITE,
                        S::COMPUTE_SHADER,
                        A::SHADER_STORAGE_WRITE,
                    ),
                ],
            ),
        ];
        let mut buffer: vk::Buffer = unsafe { std::mem::transmute(123_u64) };
        for test_case in test_cases.into_iter() {
            let mut buffer = Res::new(&mut buffer);

            let mut stage1 = make_stage(0);
            test_case.0[0].stage(&mut stage1, &mut buffer);

            let mut stage2 = make_stage(1);
            test_case.0[1].stage(&mut stage2, &mut buffer);

            let mut called = false;
            CommandBufferRecordContext::add_barrier(&stage2, |dep| {
                called = true;
                assert_global(
                    dep,
                    test_case.1[0].0,
                    test_case.1[0].1,
                    test_case.1[0].2,
                    test_case.1[0].3,
                );
            });

            if test_case.1[0].0 != vk::PipelineStageFlags2::NONE
                && test_case.1[0].2 != vk::PipelineStageFlags2::NONE
            {
                assert!(called);
            }
            called = false;

            let mut stage3 = make_stage(2);
            test_case.0[2].stage(&mut stage3, &mut buffer);

            CommandBufferRecordContext::add_barrier(&stage3, |dep| {
                called = true;
                assert_global(
                    dep,
                    test_case.1[1].0,
                    test_case.1[1].1,
                    test_case.1[1].2,
                    test_case.1[1].3,
                );
            });
            if test_case.1[1].0 != vk::PipelineStageFlags2::NONE
                && test_case.1[1].2 != vk::PipelineStageFlags2::NONE
            {
                assert!(called);
            }
        }
    }

    #[test]
    fn c2c_image_tests() {
        let image: vk::Image = unsafe { std::mem::transmute(123_usize) };
        let mut stage_image = StageContextImage {
            image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            },
            extent: vk::Extent3D::default(),
        };
        let image2: vk::Image = unsafe { std::mem::transmute(456_usize) };
        let mut stage_image2 = StageContextImage {
            image: image2,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            },
            extent: vk::Extent3D::default(),
        };

        let mut buffer1: vk::Buffer = unsafe { std::mem::transmute(4562_usize) };
        let mut buffer2: vk::Buffer = unsafe { std::mem::transmute(578_usize) };

        {
            let mut stage_image_res = ResImage::new(&mut stage_image, vk::ImageLayout::GENERAL);
            // First dispatch writes to a storage image, second dispatch reads from that storage image.
            let mut stage1 = make_stage(0);
            stage1.write_image(
                &mut stage_image_res,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::ImageLayout::GENERAL,
            );

            let mut stage2 = make_stage(1);
            stage2.read_image(
                &mut stage_image_res,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::ImageLayout::GENERAL,
            );

            let mut called = false;
            CommandBufferRecordContext::add_barrier(&stage2, |dep| {
                called = true;
                assert_global(
                    dep,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                );
            });
            assert!(called);
        }
        {
            let mut stage_image_res = ResImage::new(&mut stage_image, vk::ImageLayout::GENERAL);
            // Dispatch writes into a storage image. Draw samples that image in a fragment shader.
            let mut stage1 = make_stage(0);
            stage1.write_image(
                &mut stage_image_res,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::ImageLayout::GENERAL,
            );

            let mut stage2 = make_stage(1);
            stage2.read_image(
                &mut stage_image_res,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::ImageLayout::READ_ONLY_OPTIMAL,
            );

            let mut called = false;
            CommandBufferRecordContext::add_barrier(&stage2, |dep| {
                called = true;
                assert_eq!(dep.memory_barrier_count, 0);
                assert_eq!(dep.buffer_memory_barrier_count, 0);
                assert_eq!(dep.image_memory_barrier_count, 1);
                let image_memory_barrier = unsafe { &*dep.p_image_memory_barriers };
                assert_eq!(image_memory_barrier.old_layout, vk::ImageLayout::GENERAL);
                assert_eq!(
                    image_memory_barrier.new_layout,
                    vk::ImageLayout::READ_ONLY_OPTIMAL
                );
                assert_eq!(
                    image_memory_barrier.src_stage_mask,
                    vk::PipelineStageFlags2::COMPUTE_SHADER
                );
                assert_eq!(
                    image_memory_barrier.src_access_mask,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE
                );
                assert_eq!(
                    image_memory_barrier.dst_stage_mask,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER
                );
                assert_eq!(
                    image_memory_barrier.dst_access_mask,
                    vk::AccessFlags2::SHADER_SAMPLED_READ
                );
            });
            assert!(called);
        }

        {
            // Tests that image access info are retained across stages.
            let mut stage_image_res = ResImage::new(&mut stage_image, vk::ImageLayout::GENERAL);
            let mut stage_image_res2 =
                ResImage::new(&mut stage_image2, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            let mut buffer_res1 = Res::new(&mut buffer1);
            let mut buffer_res2 = Res::new(&mut buffer2);
            // Stage 1 is a compute shader which writes into buffer1 and an image.
            // Stage 2 is a graphics pass which reads buffer1 as the vertex input and writes to another image.
            // Stage 3 is a compute shader, reads both images, and writes into buffer2.
            // Stage 4 is a compute shader that reads buffer2.
            let mut stage1 = make_stage(0);
            stage1.write(
                &mut buffer_res1,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );
            stage1.write_image(
                &mut stage_image_res,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::ImageLayout::GENERAL,
            );

            let mut stage2 = make_stage(1);
            stage2.read(
                &mut buffer_res1,
                vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
                vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
            );
            stage2.write_image(
                &mut stage_image_res2,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            let mut called = false;
            CommandBufferRecordContext::add_barrier(&stage2, |dep| {
                called = true;
                assert_global(
                    dep,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
                    vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
                );
            });
            assert!(called);

            let mut stage3 = make_stage(2);
            stage3.read_image(
                &mut stage_image_res,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            stage3.read_image(
                &mut stage_image_res2,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            stage3.write(
                &mut buffer_res2,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );

            called = false;
            CommandBufferRecordContext::add_barrier(&stage3, |dep| {
                called = true;
                assert_eq!(dep.memory_barrier_count, 0);
                assert_eq!(dep.buffer_memory_barrier_count, 0);
                assert_eq!(dep.image_memory_barrier_count, 2);
                let image_memory_barriers = unsafe {
                    std::slice::from_raw_parts(
                        dep.p_image_memory_barriers,
                        dep.image_memory_barrier_count as usize,
                    )
                };
                {
                    let image_memory_barrier = image_memory_barriers
                        .iter()
                        .find(|a| a.image == image2)
                        .unwrap();
                    assert_eq!(
                        image_memory_barrier.old_layout,
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                    );
                    assert_eq!(
                        image_memory_barrier.new_layout,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    );
                    assert_eq!(
                        image_memory_barrier.src_stage_mask,
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                    );
                    assert_eq!(
                        image_memory_barrier.src_access_mask,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                    );
                    assert_eq!(
                        image_memory_barrier.dst_stage_mask,
                        vk::PipelineStageFlags2::COMPUTE_SHADER
                    );
                    assert_eq!(
                        image_memory_barrier.dst_access_mask,
                        vk::AccessFlags2::SHADER_STORAGE_READ
                    );
                }
                {
                    let image_memory_barrier = image_memory_barriers
                        .iter()
                        .find(|a| a.image == image)
                        .unwrap();
                    assert_eq!(image_memory_barrier.old_layout, vk::ImageLayout::GENERAL);
                    assert_eq!(
                        image_memory_barrier.new_layout,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    );
                    assert_eq!(
                        image_memory_barrier.src_stage_mask,
                        vk::PipelineStageFlags2::COMPUTE_SHADER
                    );
                    assert_eq!(
                        image_memory_barrier.src_access_mask,
                        vk::AccessFlags2::SHADER_STORAGE_WRITE
                    );
                    assert_eq!(
                        image_memory_barrier.dst_stage_mask,
                        vk::PipelineStageFlags2::COMPUTE_SHADER
                    );
                    assert_eq!(
                        image_memory_barrier.dst_access_mask,
                        vk::AccessFlags2::SHADER_STORAGE_READ
                    );
                }
            });
            assert!(called);

            let mut stage4 = make_stage(3);
            stage4.read(
                &mut buffer_res2,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            );
            called = false;
            CommandBufferRecordContext::add_barrier(&stage4, |dep| {
                called = true;
                assert_global(
                    dep,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                );
            });
            assert!(called);
        }
    }
}
