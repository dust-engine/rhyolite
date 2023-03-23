use core::num;
use std::collections::{BTreeMap, BTreeSet};

use ash::vk;
use rhyolite::macros::commands;
use crate::{future::{PerFrameState, PerFrameContainer, use_shared_state, SharedDeviceStateHostContainer, use_shared_state_with_old, RenderRes, GPUCommandFuture}, ResidentBuffer, Allocator, BufferLike, utils::merge_ranges::MergeRangeIteratorExt, commands, copy_buffer, copy_buffer_regions, HasDevice, device};

pub struct ManagedBuffer<T> {
    allocator: Allocator,
    buffer_usage_flags: vk::BufferUsageFlags,
    strategy: ManagedBufferStrategy<T>
}
impl<T> HasDevice for ManagedBuffer<T>  {
    fn device(&self) -> &std::sync::Arc<crate::Device> {
        self.allocator.device()
    }
}

enum ManagedBufferStrategy<T> {
    DirectWrite {
        buffers: PerFrameState<ResidentBuffer>,
        objects: Vec<T>,
        changeset: BTreeMap<vk::Buffer, BTreeSet<usize>>,
    },
    StagingBuffer {
        device_buffer: Option<SharedDeviceStateHostContainer<ResidentBuffer>>,
        staging_buffer: PerFrameState<ResidentBuffer>,
        changes: BTreeMap<usize, T>,
        num_items: usize
    }
}

impl<T> ManagedBuffer<T> {
    pub fn new(allocator: Allocator, buffer_usage_flags: vk::BufferUsageFlags) -> Self {
        use crate::PhysicalDeviceMemoryModel::*;
        let strategy = match allocator.physical_device().memory_model() {
            Discrete | Bar => ManagedBufferStrategy::StagingBuffer {
                device_buffer: None, staging_buffer: Default::default(), changes: Default::default(), num_items: 0 },
            ResizableBar | UMA => ManagedBufferStrategy::DirectWrite {
                buffers: Default::default(), objects: Vec::new(), changeset: Default::default() }
        };
        Self {
            allocator,
            buffer_usage_flags,
            strategy
        }
    }
    pub fn len(&mut self) -> usize {
        match &self.strategy {
            ManagedBufferStrategy::DirectWrite { objects, .. }=> {
                objects.len()
            },
            ManagedBufferStrategy::StagingBuffer { num_items, .. } => {
                *num_items
            }
        }
    }
    pub fn allocator(&self) -> &Allocator {
        &self.allocator
    }
    pub fn push(&mut self, item: T) {
        match &mut self.strategy {
            ManagedBufferStrategy::DirectWrite { objects, changeset, .. }=> {
                let index = objects.len();
                objects.push(item);
                for changes in changeset.values_mut() {
                    changes.insert(index);
                }
            },
            ManagedBufferStrategy::StagingBuffer { changes, num_items, .. } => {
                let index = *num_items;
                *num_items += 1;
                changes.insert(index, item);
            }
        }
    }
    pub fn set(&mut self, index: usize, item: T) {
        match &mut self.strategy {
            ManagedBufferStrategy::DirectWrite { objects, changeset, .. }=> {
                objects[index] = item;
                for changes in changeset.values_mut() {
                    changes.insert(index);
                }
            },
            ManagedBufferStrategy::StagingBuffer { changes, .. } => {
                changes.insert(index, item);
            }
        }
    }

    fn buffer(&mut self) -> impl GPUCommandFuture<Output = RenderRes<Box<dyn BufferLike>>> {
        let item_size = std::alloc::Layout::new::<T>().pad_to_align().size();
        let (device_buffer_direct, device_buffer_staged, old_device_buffer) = match &mut self.strategy {
            ManagedBufferStrategy::DirectWrite { buffers, objects, changeset }=> {
                let create_buffer = || {
                    let create_buffer = self.allocator.create_write_buffer_uninit(
                        (objects.capacity() * item_size) as u64,
                    self.buffer_usage_flags).unwrap();
                    create_buffer.contents_mut().unwrap().copy_from_slice(unsafe {
                        std::slice::from_raw_parts(objects.as_ptr() as *const u8, objects.len() * item_size)
                    });
                    create_buffer
                };
                
                let buf = buffers.use_state(|| {
                    let new_buffer = create_buffer();
                    changeset.insert(new_buffer.raw_buffer(), Default::default());
                    new_buffer
                }).reuse(|buffer| {
                    let changes = std::mem::take(changeset.get_mut(&buffer.raw_buffer()).unwrap());
                    if buffer.size() < (objects.len() * item_size) as u64 {
                        // need to recreate the buffer
                        changeset.remove(&buffer.raw_buffer()).unwrap();
                        let new_buffer = create_buffer();
                        // Create empty entry for the buffer just created.
                        // From now on, changes need to be recorded for this new buffer.
                        changeset.insert(new_buffer.raw_buffer(), Default::default());
                        *buffer = new_buffer;
                    } else {
                        for (changed_index_start, num_changes) in changes.into_iter().merge_ranges() {
                            let start = changed_index_start * item_size;
                            let end = (changed_index_start + num_changes) * item_size;
                            buffer.contents_mut().unwrap()[start..end].copy_from_slice(unsafe {
                                std::slice::from_raw_parts((objects.as_ptr() as *const u8).add(start), num_changes * item_size)
                            })
                        }
                    }
                });
                (Some(buf), None, None)
            },
            ManagedBufferStrategy::StagingBuffer { num_items, device_buffer, staging_buffer, changes } => {
                let expected_staging_size = (item_size * changes.len()) as u64;
                let staging_buffer = staging_buffer.use_state(|| {
                    self.allocator.create_staging_buffer(expected_staging_size).unwrap()
                }).reuse(|old| {
                    if old.size() < expected_staging_size {
                        // Too small. Enlarge.
                        *old = self.allocator.create_staging_buffer((old.size() as u64 * 2).max(expected_staging_size)).unwrap();
                    }
                });

                let changes = std::mem::take(changes);
                let (changed_indices, changed_items): (Vec<usize>, Vec<T>) = changes.into_iter().unzip();
                staging_buffer.contents_mut().unwrap().copy_from_slice(unsafe {
                    std::slice::from_raw_parts(changed_items.as_ptr() as *const u8, std::mem::size_of_val(changed_items.as_slice()))
                });

                let staging_current_index = 0;
                let buffer_copy = changed_indices.into_iter().merge_ranges().map(|(start, len)| {
                    vk::BufferCopy {
                        src_offset: staging_current_index * item_size as u64,
                        dst_offset: start as u64 * item_size as u64,
                        size: len as u64 * item_size as u64
                    }
                }).collect::<Vec<_>>();

                let expected_whole_buffer_size = *num_items as u64 * item_size as u64;
                let (device_buffer, old_device_buffer) = use_shared_state_with_old(device_buffer, |_| {
                    self.allocator.create_device_buffer_uninit(expected_whole_buffer_size, self.buffer_usage_flags).unwrap()
                }, |buf| {
                    buf.size() < expected_whole_buffer_size
                });
                (None, Some((device_buffer, staging_buffer, buffer_copy)), old_device_buffer)
            }
        };
        
        commands! {
            if let Some((device_buffer, staging_buffer, buffer_copy)) = device_buffer_staged {
                let mut device_buffer = RenderRes::new(device_buffer);
                
                if let Some(old_buffer) = old_device_buffer {
                    let old_buffer = RenderRes::new(old_buffer);
                    copy_buffer(&old_buffer, &mut device_buffer).await;
                    retain!(old_buffer);
                }
                
                let staging_buffer = RenderRes::new(staging_buffer);
                copy_buffer_regions(&staging_buffer, &mut device_buffer, buffer_copy).await;
                retain!(staging_buffer);
                
                
                device_buffer.map(|a| Box::new(a) as Box<dyn BufferLike>)
            } else {
                let buffer = Box::new(device_buffer_direct.unwrap()) as Box<dyn BufferLike>;
                RenderRes::new(buffer)
            }
        }
    }
}
