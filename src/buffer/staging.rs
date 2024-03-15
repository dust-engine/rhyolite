use std::{
    collections::VecDeque,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use ash::{prelude::VkResult, vk};
use bevy::ecs::{system::Resource, world::FromWorld};

use crate::{utils::Dispose, Device};

/// A ring buffer for allocations of transient data.
/// TODO: Can we make this thread safe?
#[derive(Resource)]
pub struct StagingBelt(Arc<StagingBeltInner>);
impl FromWorld for StagingBelt {
    fn from_world(world: &mut bevy::ecs::world::World) -> Self {
        let device = world.resource::<Device>().clone();
        // 64MB page size
        StagingBelt::new(device, 64 * 1024 * 1024).unwrap()
    }
}
struct StagingBeltInner {
    device: Device,
    chunk_size: vk::DeviceSize,
    head: AtomicU64,
    tail: u64,
    used_chunks: VecDeque<StagingBeltChunk>,
    memory_type_index: u32,
}
impl StagingBelt {
    pub fn new(device: Device, chunk_size: vk::DeviceSize) -> VkResult<Self> {
        let Some((memory_type_index, _)) = device
            .physical_device()
            .properties()
            .memory_types()
            .iter()
            .enumerate()
            .find(|(_, memory_type)| {
                memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                    && !memory_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    && !memory_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_CACHED)
            })
        else {
            return ash::prelude::VkResult::Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        };

        Ok(StagingBelt(Arc::new(StagingBeltInner {
            device,
            chunk_size,
            head: AtomicU64::new(0),
            tail: 0,
            used_chunks: VecDeque::new(),
            memory_type_index: memory_type_index as u32,
        })))
    }
    #[must_use]
    pub fn start(&mut self) -> StagingBeltBatchJob {
        StagingBeltBatchJob { belt: self }
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
}
impl Drop for StagingBeltBatchJob<'_> {
    fn drop(&mut self) {
        panic!("Calling the `finish` method is required.");
    }
}
pub struct StagingBeltRecallToken {
    belt: Arc<StagingBeltInner>,
    tail: u64,
}
impl Drop for StagingBeltRecallToken {
    fn drop(&mut self) {
        let old = self.belt.head.swap(self.tail, Ordering::Relaxed);
        assert!(
            old <= self.tail,
            "Staging belt recall token used out of order"
        );
    }
}
impl StagingBeltBatchJob<'_> {
    pub fn allocate_buffer(
        &mut self,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
    ) -> StagingBeltSuballocation {
        // This should be ok. All other copies of this are only gonna access the atomic value contained within.
        let belt = unsafe { Arc::get_mut_unchecked(&mut self.belt.0) };
        if size > belt.chunk_size {
            unimplemented!()
        }
        let start = belt.tail;
        let aligned_start = belt.tail.next_multiple_of(alignment);
        let end = aligned_start + size;

        let mut current_chunk_end_index = 0;
        if let Some(current_chunk) = &mut belt.used_chunks.back() {
            if end <= current_chunk.end_index {
                // there's enough space
                let offset = aligned_start - current_chunk.start_index;
                belt.tail = end;
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
        if let Some(peek) = belt.used_chunks.front() {
            if belt.head.load(Ordering::Relaxed) >= peek.end_index {
                // has already been freed
                let mut chunk = belt.used_chunks.pop_front().unwrap();
                chunk.end_index = current_chunk_end_index + belt.chunk_size;
                chunk.start_index = current_chunk_end_index;
                let ptr = chunk.ptr;
                let buffer = chunk.buffer;
                belt.used_chunks.push_back(chunk);
                belt.tail = current_chunk_end_index + size;
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
            let buffer = belt
                .device
                .create_buffer(
                    &vk::BufferCreateInfo {
                        usage: vk::BufferUsageFlags::TRANSFER_SRC,
                        size: belt.chunk_size,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            let memory = belt
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo {
                        allocation_size: belt.chunk_size,
                        memory_type_index: belt.memory_type_index,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            belt.device.bind_buffer_memory(buffer, memory, 0).unwrap();
            let ptr = belt
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .unwrap() as *mut u8;
            let ptr = NonNull::new(ptr).unwrap();
            let chunk = StagingBeltChunk {
                buffer,
                memory,
                ptr,
                end_index: belt.chunk_size + current_chunk_end_index,
                start_index: current_chunk_end_index,
            };
            belt.used_chunks.push_back(chunk);
            belt.tail = current_chunk_end_index + size;
            return StagingBeltSuballocation {
                buffer,
                start: current_chunk_end_index,
                offset: 0,
                size,
                ptr,
            };
        }
    }
    pub fn finish(self) -> Dispose<StagingBeltRecallToken> {
        let belt = unsafe { Arc::get_mut_unchecked(&mut self.belt.0) };
        let tail = belt.tail;
        let belt = self.belt.0.clone();
        std::mem::forget(self);
        Dispose::new(StagingBeltRecallToken { tail: tail, belt })
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

#[cfg(test)]
mod tests {
    use ash::vk;

    use super::StagingBelt;

    fn create_device_for_test() -> crate::Device {
        use crate::{Device, Instance};
        use cstr::cstr;
        use std::sync::Arc;
        let instance = Instance::create(
            Arc::new(unsafe { ash::Entry::load().unwrap() }),
            crate::InstanceCreateInfo {
                flags: vk::InstanceCreateFlags::empty(),
                application_name: cstr!("Rhyolite Tests"),
                application_version: Default::default(),
                engine_name: cstr!("rhyolite"),
                engine_version: Default::default(),
                api_version: crate::Version::new(1, 3, 0, 0),
                enabled_layer_names: &[],
                enabled_extension_names: &[],
                meta_builders: Vec::new(),
            },
        )
        .unwrap();
        let pdevice = instance
            .enumerate_physical_devices()
            .unwrap()
            .next()
            .unwrap();
        let priority: f32 = 1.0;
        let device = Device::create(
            pdevice,
            &[vk::DeviceQueueCreateInfo {
                queue_family_index: 0,
                queue_count: 1,
                p_queue_priorities: &priority,
                ..Default::default()
            }],
            &[],
            &vk::PhysicalDeviceFeatures2::default(),
            Vec::new(),
        )
        .unwrap();
        device
    }

    #[test]
    fn test_allocate() {
        let device = create_device_for_test();
        let mut belt = StagingBelt::new(device, 128).unwrap();
        let mut allocator = belt.start();
        let buf1 = allocator.allocate_buffer(64, 1);
        assert_eq!(buf1.start, 0);
        assert_eq!(buf1.offset, 0);
        assert_eq!(buf1.size, 64);
        let buffer = buf1.buffer;

        allocator.allocate_buffer(4, 1);
        let buf2 = allocator.allocate_buffer(32, 16);
        assert_eq!(buf2.start, 68);
        assert_eq!(buf2.offset, 80);
        assert_eq!(buf2.size, 32);
        assert_eq!(buf2.buffer, buffer);

        let buf3 = allocator.allocate_buffer(64, 1);
        assert_eq!(buf3.start, 128);
        assert_eq!(buf3.offset, 0);
        assert_eq!(buf3.size, 64);
        assert_ne!(buf3.buffer, buffer);
        let buffer = buf3.buffer;

        allocator.allocate_buffer(4, 1);
        let buf4 = allocator.allocate_buffer(64, 1);
        // didn't have enough space, should create a new page now
        assert_eq!(buf4.start, 256);
        assert_eq!(buf4.offset, 0);
        assert_eq!(buf4.size, 64);
        assert_ne!(buf4.buffer, buffer);

        unsafe {
            allocator.finish().take();
        }
    }

    #[test]
    fn test_reusing() {
        let device = create_device_for_test();
        let mut belt = StagingBelt::new(device, 64).unwrap();
        let mut allocator = belt.start();
        let buf = allocator.allocate_buffer(4, 1);
        assert_eq!(buf.start, 0);
        assert_eq!(buf.offset, 0);
        assert_eq!(buf.size, 4);
        let initial_buffer = buf.buffer;
        let buf = allocator.allocate_buffer(56, 1);
        assert_eq!(buf.start, 4);
        assert_eq!(buf.offset, 4);
        assert_eq!(buf.size, 56);
        let buf = allocator.allocate_buffer(56, 1);
        assert_eq!(buf.start, 64);
        assert_eq!(buf.offset, 0);
        assert_eq!(buf.size, 56);

        unsafe {
            allocator.finish().take();
        }

        let mut allocator = belt.start();
        let buf = allocator.allocate_buffer(16, 1);
        assert_eq!(buf.start, 128);
        assert_eq!(buf.offset, 0);
        assert_eq!(buf.size, 16);
        assert_eq!(buf.buffer, initial_buffer);
        unsafe {
            allocator.finish().take();
        }
    }
}
