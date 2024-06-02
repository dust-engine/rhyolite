use std::{
    collections::VecDeque,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::Arc,
};

use ash::vk;
use bevy::{
    app::Plugin,
    ecs::{
        system::{Local, ResMut, Resource},
        world::FromWorld,
    },
};
use bytemuck::{AnyBitPattern, NoUninit};

use crate::{
    commands::SemaphoreSignalCommands, semaphore::TimelineSemaphore, BufferLike, Device, HasDevice,
};

impl FromWorld for StagingBelt {
    fn from_world(world: &mut bevy::ecs::world::World) -> Self {
        let device = world.resource::<Device>().clone();
        let chunk_size = 64 * 1024 * 1024;

        let mut requirements = vk::MemoryRequirements2::default();
        unsafe {
            device.get_device_buffer_memory_requirements(
                &vk::DeviceBufferMemoryRequirements::default().create_info(&vk::BufferCreateInfo {
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    size: chunk_size,
                    ..Default::default()
                }),
                &mut requirements,
            );
        }
        assert_eq!(requirements.memory_requirements.size, chunk_size);

        let Some((memory_type_index, _)) = device
            .physical_device()
            .properties()
            .memory_types()
            .iter()
            .enumerate()
            .rev()
            .filter(|(index, memory_type)| {
                memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                    && requirements.memory_requirements.memory_type_bits & (1 << index) != 0
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
            panic!()
        };

        // 64MB page size
        StagingBelt::new_with_memory_type_index(
            device,
            chunk_size,
            memory_type_index as u32,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )
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
    usage: vk::BufferUsageFlags,

    lifetime_marker: Vec<(Arc<TimelineSemaphore>, u64)>,
}

/// Ring allocator for staging buffers.
/// Good for occasional updates of device-local buffers.
impl StagingBelt {
    pub(crate) fn new_with_memory_type_index(
        device: Device,
        chunk_size: vk::DeviceSize,
        memory_type_index: u32,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        StagingBelt {
            device,
            chunk_size,
            head: 0,
            tail: 0,
            used_chunks: VecDeque::new(),
            memory_type_index: memory_type_index as u32,
            lifetime_marker: Vec::new(),
            usage,
        }
    }
    fn cleanup(&mut self, jobs: &mut VecDeque<StagingBufferCleanupJob>) {
        'pop_ready_jobs: while let Some(job) = jobs.front() {
            for (sem, val) in job.lifetime_marker.iter() {
                if !sem.is_signaled(*val) {
                    break 'pop_ready_jobs;
                }
            }
            // All semaphores in this job are now marked as ready.
            let job = jobs.pop_front().unwrap();
            assert!(self.head < job.tail);
            self.head = job.tail;
        }

        if !self.lifetime_marker.is_empty() {
            let job = StagingBufferCleanupJob {
                lifetime_marker: std::mem::take(&mut self.lifetime_marker),
                tail: self.tail,
            };
            jobs.push_back(job);
        }
    }
    #[must_use]
    pub fn start(&mut self, commands: &mut impl SemaphoreSignalCommands) -> StagingBeltBatchJob {
        self.start_aligned(commands, 1)
    }
    #[must_use]
    pub fn start_aligned(
        &mut self,
        commands: &mut impl SemaphoreSignalCommands,
        alignment: u32,
    ) -> StagingBeltBatchJob {
        let (sem, val) = commands.signal_semaphore(vk::PipelineStageFlags2KHR::TRANSFER);

        for (curr_sem, curr_val) in self.lifetime_marker.iter_mut() {
            if Arc::ptr_eq(curr_sem, &sem) {
                *curr_val = val.max(*curr_val);
                return StagingBeltBatchJob {
                    belt: self,
                    dirty: false,
                    alignment,
                };
            }
        }
        self.lifetime_marker.push((sem, val));
        StagingBeltBatchJob {
            belt: self,
            dirty: false,
            alignment,
        }
    }
}
struct StagingBeltChunk {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: NonNull<u8>,
    device_address: vk::DeviceAddress,
    start_index: u64,
    end_index: u64,
}
unsafe impl Send for StagingBeltChunk {}
unsafe impl Sync for StagingBeltChunk {}

pub struct StagingBeltBatchJob<'a> {
    belt: &'a mut StagingBelt,
    alignment: u32,
    dirty: bool,
}

impl StagingBeltBatchJob<'_> {
    pub fn push_item<T>(&mut self, item: &T) -> StagingBeltSuballocation<T> {
        let mut x = self.allocate_item::<T>();
        x.write(item);
        x
    }
    pub fn allocate_item<T>(&mut self) -> StagingBeltSuballocation<T> {
        let allocation = self.allocate_buffer(std::mem::size_of::<T>() as u64);
        StagingBeltSuballocation {
            _marker: std::marker::PhantomData,
            ..allocation
        }
    }
    pub fn allocate_buffer(&mut self, size: vk::DeviceSize) -> StagingBeltSuballocation<[u8]> {
        self.dirty = true;
        if size > self.belt.chunk_size {
            unimplemented!()
        }
        let start = self.belt.tail;
        let aligned_start = self.belt.tail.next_multiple_of(self.alignment as u64);
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
                    device_address: if current_chunk.device_address == 0 {
                        0
                    } else {
                        current_chunk.device_address + offset
                    },
                    ptr: unsafe { current_chunk.ptr.add(offset as usize) },
                    _marker: PhantomData,
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
                let device_address = chunk.device_address;
                self.belt.used_chunks.push_back(chunk);
                self.belt.tail = current_chunk_end_index + size;
                return StagingBeltSuballocation {
                    buffer,
                    start: current_chunk_end_index,
                    offset: 0,
                    size,
                    ptr,
                    device_address,
                    _marker: PhantomData,
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
                        usage: self.belt.usage,
                        size: self.belt.chunk_size,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            let mut memory_allocate_flags = vk::MemoryAllocateFlags::default();
            if self
                .belt
                .usage
                .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            {
                memory_allocate_flags |= vk::MemoryAllocateFlags::DEVICE_ADDRESS;
            }
            let memory = self
                .belt
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo {
                        allocation_size: self.belt.chunk_size,
                        memory_type_index: self.belt.memory_type_index,
                        ..Default::default()
                    }
                    .push_next(&mut vk::MemoryAllocateFlagsInfo {
                        flags: memory_allocate_flags,
                        ..Default::default()
                    }),
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
            let device_address = if self
                .belt
                .usage
                .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            {
                self.belt
                    .device
                    .get_buffer_device_address(&vk::BufferDeviceAddressInfo {
                        buffer,
                        ..Default::default()
                    })
            } else {
                0
            };
            let chunk = StagingBeltChunk {
                buffer,
                memory,
                ptr,
                end_index: self.belt.chunk_size + current_chunk_end_index,
                start_index: current_chunk_end_index,
                device_address,
            };
            self.belt.used_chunks.push_back(chunk);
            self.belt.tail = current_chunk_end_index + size;
            return StagingBeltSuballocation {
                buffer,
                start: current_chunk_end_index,
                offset: 0,
                size,
                ptr,
                device_address,
                _marker: PhantomData,
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

pub struct StagingBeltSuballocation<T: ?Sized> {
    pub buffer: vk::Buffer,
    // The start of the suballocation block, including the alignment padding
    start: u64,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
    ptr: NonNull<u8>,
    device_address: vk::DeviceAddress,
    _marker: std::marker::PhantomData<T>,
}
impl<T: ?Sized> BufferLike for StagingBeltSuballocation<T> {
    fn raw_buffer(&self) -> vk::Buffer {
        self.buffer
    }
    fn offset(&self) -> vk::DeviceSize {
        self.offset
    }
    fn size(&self) -> vk::DeviceSize {
        self.size
    }
    fn device_address(&self) -> vk::DeviceAddress {
        if self.device_address == 0 {
            panic!("Device address not available for staging buffer");
        }
        self.device_address
    }
}
impl<T: ?Sized> StagingBeltSuballocation<T> {
    pub fn write(&mut self, item: &T) {
        assert_eq!(std::mem::size_of_val(item), self.size as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(
                item as *const T as *const u8,
                self.ptr.as_ptr(),
                self.size as usize,
            );
        }
    }
    pub fn uninit_mut(&mut self) -> &mut MaybeUninit<T>
    where
        T: Sized,
    {
        assert_eq!(self.size, std::mem::size_of::<T>() as u64);
        unsafe { self.ptr.cast::<T>().as_uninit_mut() }
    }
}
impl<T: AnyBitPattern> Deref for StagingBeltSuballocation<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        let slice = unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size as usize) };
        bytemuck::from_bytes(slice)
    }
}
impl<T: AnyBitPattern + NoUninit> DerefMut for StagingBeltSuballocation<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let slice =
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size as usize) };
        bytemuck::from_bytes_mut(slice)
    }
}
impl Deref for StagingBeltSuballocation<[u8]> {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size as usize) }
    }
}
impl DerefMut for StagingBeltSuballocation<[u8]> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size as usize) }
    }
}

struct StagingBufferCleanupJob {
    lifetime_marker: Vec<(Arc<TimelineSemaphore>, u64)>,
    tail: u64,
}
fn staging_buffer_cleanup_system(
    mut staging_belt: ResMut<StagingBelt>,
    mut uniform_belt: ResMut<UniformBelt>,
    mut staging_cleanup_jobs: Local<VecDeque<StagingBufferCleanupJob>>,
    mut uniform_cleanup_jobs: Local<VecDeque<StagingBufferCleanupJob>>,
) {
    staging_belt.cleanup(&mut staging_cleanup_jobs);
    uniform_belt.0.cleanup(&mut uniform_cleanup_jobs);
}

/// Ring allocator for uniform buffers
/// This will be created on host-visible and preferably device-local memory.
#[derive(Resource)]
pub struct UniformBelt(StagingBelt);
impl UniformBelt {
    #[must_use]
    pub fn start(&mut self, commands: &mut impl SemaphoreSignalCommands) -> StagingBeltBatchJob {
        let alignment = self
            .0
            .device
            .physical_device()
            .properties()
            .limits
            .min_uniform_buffer_offset_alignment;
        self.0.start_aligned(commands, alignment as u32)
    }

    #[must_use]
    pub fn start_aligned(
        &mut self,
        commands: &mut impl SemaphoreSignalCommands,
        alignment: u32,
    ) -> StagingBeltBatchJob {
        self.0.start_aligned(commands, alignment)
    }
}
impl FromWorld for UniformBelt {
    fn from_world(world: &mut bevy::prelude::World) -> Self {
        let device = world.resource::<Device>().clone();

        let mut usages = vk::BufferUsageFlags::UNIFORM_BUFFER;
        if device
            .feature::<vk::PhysicalDeviceBufferDeviceAddressFeatures>()
            .map(|f| f.buffer_device_address == vk::TRUE)
            .unwrap_or(false)
        {
            usages |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        }
        if device
            .feature::<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>()
            .map(|f| f.ray_tracing_pipeline == vk::TRUE)
            .unwrap_or(false)
        {
            usages |= vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR;
        }
        let chunk_size = 1 * 1024 * 1024;

        let mut requirements = vk::MemoryRequirements2::default();
        unsafe {
            device.get_device_buffer_memory_requirements(
                &vk::DeviceBufferMemoryRequirements::default().create_info(&vk::BufferCreateInfo {
                    usage: usages,
                    size: chunk_size,
                    ..Default::default()
                }),
                &mut requirements,
            );
        }
        assert_eq!(requirements.memory_requirements.size, chunk_size);
        let Some((memory_type_index, flags)) = device
            .physical_device()
            .properties()
            .memory_types()
            .iter()
            .enumerate()
            .rev()
            .filter(|(index, memory_type)| {
                memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                    && requirements.memory_requirements.memory_type_bits & (1 << index) != 0
            })
            .max_by_key(|(_, memory_type)| {
                let mut priority: i32 = 0;
                if memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                {
                    priority += 10;
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
            panic!()
        };
        if !flags
            .property_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        {
            tracing::warn!("Uniform buffers will be created on non-device-local memory");
        }
        Self(StagingBelt::new_with_memory_type_index(
            device,
            chunk_size,
            memory_type_index as u32,
            usages,
        ))
    }
}

pub(crate) struct StagingBeltPlugin;
impl Plugin for StagingBeltPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(bevy::app::First, staging_buffer_cleanup_system);
    }
    fn finish(&self, app: &mut bevy::prelude::App) {
        app.init_resource::<StagingBelt>();
        app.init_resource::<UniformBelt>();
    }
}

// TODO: check wrap around behavior.
// TODO: create more tests
