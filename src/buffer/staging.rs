use std::{
    collections::VecDeque,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::Arc,
};

use ash::{prelude::VkResult, vk};
use bevy::{
    app::Plugin,
    ecs::{
        system::{Local, ResMut, Resource},
        world::FromWorld,
    },
};

use crate::{
    commands::SemaphoreSignalCommands, semaphore::TimelineSemaphore, BufferLike, Device, HasDevice,
};

impl FromWorld for StagingBelt {
    fn from_world(world: &mut bevy::ecs::world::World) -> Self {
        let device = world.resource::<Device>().clone();
        // 64MB page size
        StagingBelt::new(device, 64 * 1024 * 1024).unwrap()
    }
}

#[derive(Resource)]
pub struct StagingBelt {
    device: Device,
    chunk_size: vk::DeviceSize,
    head: u64,
    tail: u64,
    used_chunks: VecDeque<StagingBeltChunk>,
    memory_type_index: u32,

    lifetime_marker: Vec<(Arc<TimelineSemaphore>, u64)>,
}
impl StagingBelt {
    pub fn new(device: Device, chunk_size: vk::DeviceSize) -> VkResult<Self> {
        let Some((memory_type_index, _)) = device
            .physical_device()
            .properties()
            .memory_types()
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, memory_type)| {
                memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            })
            .max_by_key(|(_, memory_type)| {
                let mut priority: i32 = 0;
                if memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                {
                    priority -= 10;
                }
                if memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_CACHED)
                {
                    priority -= 1;
                }
                if memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD)
                {
                    priority -= 100;
                }
                if memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD)
                {
                    priority -= 1000;
                }
                priority
            })
        else {
            return ash::prelude::VkResult::Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        };

        Ok(StagingBelt {
            device,
            chunk_size,
            head: 0,
            tail: 0,
            used_chunks: VecDeque::new(),
            memory_type_index: memory_type_index as u32,
            lifetime_marker: Vec::new(),
        })
    }
    #[must_use]
    pub fn start(&mut self, commands: &mut impl SemaphoreSignalCommands) -> StagingBeltBatchJob {
        let (sem, val) = commands.signal_semaphore(vk::PipelineStageFlags2KHR::TRANSFER);

        for (curr_sem, curr_val) in self.lifetime_marker.iter_mut() {
            if Arc::ptr_eq(curr_sem, &sem) {
                *curr_val = val.max(*curr_val);
                return StagingBeltBatchJob {
                    belt: self,
                    dirty: false,
                };
            }
        }
        self.lifetime_marker.push((sem, val));
        StagingBeltBatchJob {
            belt: self,
            dirty: false,
        }
    }
}
struct StagingBeltChunk {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: NonNull<u8>,
    start_index: u64,
    end_index: u64,
}
unsafe impl Send for StagingBeltChunk {}
unsafe impl Sync for StagingBeltChunk {}

pub struct StagingBeltBatchJob<'a> {
    belt: &'a mut StagingBelt,
    dirty: bool,
}

impl StagingBeltBatchJob<'_> {
    pub fn allocate_item<T>(&mut self) -> StagingBeltSuballocationItem<T> {
        let allocation = self.allocate_buffer(
            std::mem::size_of::<T>() as u64,
            std::mem::align_of::<T>() as u64,
        );
        StagingBeltSuballocationItem {
            allocation,
            item: std::marker::PhantomData,
        }
    }
    pub fn allocate_buffer(
        &mut self,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
    ) -> StagingBeltSuballocation {
        self.dirty = true;
        if size > self.belt.chunk_size {
            unimplemented!()
        }
        let start = self.belt.tail;
        let aligned_start = self.belt.tail.next_multiple_of(alignment);
        let end = aligned_start + size;

        let mut current_chunk_end_index = 0;
        if let Some(current_chunk) = &mut self.belt.used_chunks.back() {
            if end <= current_chunk.end_index {
                // there's enough space
                let offset = aligned_start - current_chunk.start_index;
                self.belt.tail = end;
                return StagingBeltSuballocation {
                    buffer: current_chunk.buffer,
                    start,
                    offset,
                    size,
                    ptr: unsafe { current_chunk.ptr.add(offset as usize) },
                };
            } else {
                current_chunk_end_index = current_chunk.end_index;
            }
        }
        // not enough space at the back of the belt. try reuse heads of the belt.
        if let Some(peek) = self.belt.used_chunks.front() {
            if self.belt.head >= peek.end_index {
                // has already been freed
                let mut chunk = self.belt.used_chunks.pop_front().unwrap();
                chunk.end_index = current_chunk_end_index + self.belt.chunk_size;
                chunk.start_index = current_chunk_end_index;
                let ptr = chunk.ptr;
                let buffer = chunk.buffer;
                self.belt.used_chunks.push_back(chunk);
                self.belt.tail = current_chunk_end_index + size;
                return StagingBeltSuballocation {
                    buffer,
                    start: current_chunk_end_index,
                    offset: 0,
                    size,
                    ptr,
                };
            }
        }
        // Can't reuse any old chunks, so we need to allocate a new one
        unsafe {
            let buffer = self
                .belt
                .device
                .create_buffer(
                    &vk::BufferCreateInfo {
                        usage: vk::BufferUsageFlags::TRANSFER_SRC,
                        size: self.belt.chunk_size,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            let memory = self
                .belt
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo {
                        allocation_size: self.belt.chunk_size,
                        memory_type_index: self.belt.memory_type_index,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            self.belt
                .device
                .bind_buffer_memory(buffer, memory, 0)
                .unwrap();
            let ptr = self
                .belt
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .unwrap() as *mut u8;
            let ptr = NonNull::new(ptr).unwrap();
            let chunk = StagingBeltChunk {
                buffer,
                memory,
                ptr,
                end_index: self.belt.chunk_size + current_chunk_end_index,
                start_index: current_chunk_end_index,
            };
            self.belt.used_chunks.push_back(chunk);
            self.belt.tail = current_chunk_end_index + size;
            return StagingBeltSuballocation {
                buffer,
                start: current_chunk_end_index,
                offset: 0,
                size,
                ptr,
            };
        }
    }
}
impl Drop for StagingBeltBatchJob<'_> {
    fn drop(&mut self) {
        if !self.dirty {
            self.belt.lifetime_marker.pop();
        }
    }
}

pub struct StagingBeltSuballocation {
    pub buffer: vk::Buffer,
    // The start of the suballocation block, including the alignment padding
    start: u64,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
    ptr: NonNull<u8>,
}
impl Deref for StagingBeltSuballocation {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size as usize) }
    }
}
impl DerefMut for StagingBeltSuballocation {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size as usize) }
    }
}

pub struct StagingBeltSuballocationItem<T> {
    allocation: StagingBeltSuballocation,
    item: std::marker::PhantomData<T>,
}

impl<T> BufferLike for StagingBeltSuballocationItem<T> {
    fn size(&self) -> vk::DeviceSize {
        self.allocation.size
    }
    fn raw_buffer(&self) -> vk::Buffer {
        self.allocation.buffer
    }
    fn offset(&self) -> vk::DeviceSize {
        self.allocation.offset
    }
}
impl<T> Deref for StagingBeltSuballocationItem<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self.allocation.ptr.as_ptr() as *const T) }
    }
}
impl<T> DerefMut for StagingBeltSuballocationItem<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self.allocation.ptr.as_ptr() as *mut T) }
    }
}

struct StagingBufferCleanupJob {
    lifetime_marker: Vec<(Arc<TimelineSemaphore>, u64)>,
    tail: u64,
}
fn staging_buffer_cleanup_system(
    mut staging_belt: ResMut<StagingBelt>,
    mut jobs: Local<VecDeque<StagingBufferCleanupJob>>,
) {
    'pop_ready_jobs: while let Some(job) = jobs.front() {
        for (sem, val) in job.lifetime_marker.iter() {
            if !sem.is_signaled(*val) {
                break 'pop_ready_jobs;
            }
        }
        // All semaphores in this job are now marked as ready.
        let job = jobs.pop_front().unwrap();
        assert!(staging_belt.head < job.tail);
        staging_belt.head = job.tail;
    }

    if !staging_belt.lifetime_marker.is_empty() {
        let job = StagingBufferCleanupJob {
            lifetime_marker: std::mem::take(&mut staging_belt.lifetime_marker),
            tail: staging_belt.tail,
        };
        jobs.push_back(job);
    }
}

pub(crate) struct StagingBeltPlugin;
impl Plugin for StagingBeltPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(bevy::app::First, staging_buffer_cleanup_system);
    }
    fn finish(&self, app: &mut bevy::prelude::App) {
        app.init_resource::<StagingBelt>();
    }
}

// TODO: check wrap around behavior.
// TODO: create more tests
