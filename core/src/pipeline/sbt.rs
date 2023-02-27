// Gives out SbtEntryHandle.
// User gives you the T. On duplicates, return exisitng handle.
// When unique, create new entry.

use std::{
    alloc::Layout,
    collections::{BTreeSet, HashMap, VecDeque},
    hash::Hash,
    sync::{Arc, Weak},
};

use ash::{prelude::VkResult, vk};
use macros::{commands, gpu};

use crate::{
    commands::SharedCommandPool,
    copy_buffer,
    future::{
        use_per_frame_state, GPUCommandFuture, GPUCommandFutureExt, PerFrameState, RenderRes,
    },
    Allocator, FencePool, QueueFuture, QueueRef, QueueSubmitFuture, Queues, QueuesRouter,
    RayTracingPipeline, ResidentBuffer, SbtHandles, TimelineSemaphorePool,
};

pub trait HitgroupSbtEntry: Hash + Eq + Clone {
    type ShaderParameter: Copy;
    fn parameter(&self, raytype_index: u32) -> Self::ShaderParameter;
    fn hitgroup_index(&self, raytype_index: u32) -> usize;
}

struct HitgroupSbtHandleInner {
    id: usize,
    sender: std::sync::mpsc::Sender<usize>,
}
impl Drop for HitgroupSbtHandleInner {
    fn drop(&mut self) {
        self.sender.send(self.id).unwrap();
    }
}
pub struct HitgroupSbtHandle(Arc<HitgroupSbtHandleInner>);

pub struct HitgroupSbtBuffer {
    buffer: ResidentBuffer,
}

pub struct HitgroupSbtVec<T: HitgroupSbtEntry> {
    allocator: Allocator,
    total_raytype: u32,
    shader_group_handles: SbtHandles,
    /// A map of (item -> (id, generation, handle))
    handles: HashMap<T, (usize, Weak<HitgroupSbtHandleInner>)>,

    /// A list of (generation, item) indexed by id
    entries: Vec<T>,

    sender: std::sync::mpsc::Sender<usize>,
    available_indices: std::sync::mpsc::Receiver<usize>,

    changeset: BTreeSet<usize>,
    frames: Option<Arc<ResidentBuffer>>,
}

struct QueueSubmitInfo<'a> {
    queue: &'a mut Queues,
    shared_command_pools: &'a mut [Option<SharedCommandPool>],

    semaphore_pool: &'a mut TimelineSemaphorePool,
    fence_pool: &'a mut FencePool,
    apply_final_signal: bool,
}
impl<'a> QueueSubmitInfo<'a> {
    fn submit<F: QueueFuture<RecycledState = ((),)>>(
        &mut self,
        future: F,
    ) -> QueueSubmitFuture<F::RetainedState, F::Output> {
        self.queue.submit(
            future,
            self.shared_command_pools,
            self.semaphore_pool,
            self.fence_pool,
            &mut ((),),
            self.apply_final_signal,
        )
    }
}

impl<T: HitgroupSbtEntry> HitgroupSbtVec<T> {
    pub fn new(pipeline: &RayTracingPipeline, allocator: Allocator) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        Self {
            allocator,
            total_raytype: 1,
            handles: HashMap::new(),
            entries: Vec::new(),
            sender,
            available_indices: receiver,
            changeset: BTreeSet::new(),
            frames: None,
            shader_group_handles: pipeline.get_shader_group_handles(),
        }
    }
    pub fn get(&self, handle: &HitgroupSbtHandle) -> &T {
        &self.entries[handle.0.id]
    }
    pub fn add(&mut self, item: T) -> HitgroupSbtHandle {
        if let Some((id, retained_handle)) = self.handles.get_mut(&item) {
            // This item was previously requested
            if let Some(handle) = retained_handle.upgrade() {
                // And the handle is still valid now
                assert_eq!(*id, handle.id);
                return HitgroupSbtHandle(handle);
            } else if self.entries[*id] == item {
                // But the handle is no longer valid. Fortunately no one has overwritten the entry yet.
                // Let's reuse that entry.
                let handle = Arc::new(HitgroupSbtHandleInner {
                    id: *id,
                    sender: self.sender.clone(),
                });
                *retained_handle = Arc::downgrade(&handle);
                // This way, no need to call self.record_location_update
                return HitgroupSbtHandle(handle);
            }
            unreachable!();
        }

        loop {
            let candidate = if let Some(candidate) = self.available_indices.try_recv().ok() {
                candidate
            } else {
                // No more available indices. Need to create a new entry.
                break;
            };
            let prev_item = &self.entries[candidate];
            if self.handles[prev_item].1.strong_count() == 0 {
                // This slot is safe to reuse.
                let handle = Arc::new(HitgroupSbtHandleInner {
                    id: candidate,
                    sender: self.sender.clone(),
                });
                self.handles.remove(prev_item);
                self.entries[candidate] = item.clone();
                assert!(self
                    .handles
                    .insert(item, (candidate, Arc::downgrade(&handle)))
                    .is_none());
                self.record_location_update(candidate);
                return HitgroupSbtHandle(handle);
            } else {
                // This slot was already reused. Try again.
                continue;
            }
        }

        let candidate = self.entries.len();
        self.entries.push(item.clone());
        let handle = Arc::new(HitgroupSbtHandleInner {
            id: candidate,
            sender: self.sender.clone(),
        });
        assert!(self
            .handles
            .insert(item, (candidate, Arc::downgrade(&handle)))
            .is_none());
        self.record_location_update(candidate);
        return HitgroupSbtHandle(handle);
    }

    fn record_location_update(&mut self, location: usize) {
        self.changeset.insert(location);
    }
    fn get_sbt_buffer(
        &mut self,
        mut submit_commands: QueueSubmitInfo,
        transfer_queue: QueueRef,
    ) -> VkResult<()> {
        if let Some(old_buffer) = self.frames.as_ref() {
            // copy everything over.
            // apply changeset updates.
        } else {
            let new_sbt = self.create_full_sbt()?;
            let sbt_buffer = submit_commands.submit(new_sbt.schedule_on_queue(transfer_queue));
        }
        todo!()
    }

    fn create_full_sbt(
        &self,
    ) -> VkResult<impl GPUCommandFuture<RecycledState = ((),), Output = RenderRes<ResidentBuffer>>>
    {
        let handle_size = self
            .shader_group_handles
            .handle_layout()
            .pad_to_align()
            .size();
        let entry_layout_one_raytype = self
            .shader_group_handles
            .handle_layout()
            .extend(Layout::new::<T::ShaderParameter>())
            .unwrap()
            .0;

        let entry_layout_one = entry_layout_one_raytype
            .repeat(self.total_raytype as usize)
            .unwrap()
            .0;

        let entry_layout_all = entry_layout_one.repeat(self.entries.len()).unwrap().0;
        let buffer_size = entry_layout_all.pad_to_align().size();
        let buffer = self.allocator.create_upload_buffer_uninit(
            buffer_size as u64,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
        )?;

        let mut staging_buffer = None;
        let write_target = if let Some(buffer_content) = buffer.contents_mut() {
            buffer_content
        } else {
            // Need to create staging buffer.
            staging_buffer = Some(self.allocator.create_staging_buffer(buffer_size as u64)?);
            staging_buffer.as_ref().unwrap().contents_mut().unwrap()
        };

        for (i, entry) in self.entries.iter().enumerate() {
            let size = entry_layout_one.pad_to_align().size();
            let entry_write_target = &mut write_target[size * i..size * (i + 1)];
            // For each raytype
            for i in 0..self.total_raytype {
                let size = entry_layout_one_raytype.pad_to_align().size();
                let raytype_write_target =
                    &mut entry_write_target[size * i as usize..size * (i as usize + 1)];

                let hitgroup_index = entry.hitgroup_index(i);
                let shader_data = self.shader_group_handles.hitgroup(hitgroup_index);
                raytype_write_target[..shader_data.len()].copy_from_slice(shader_data);

                let parameters = entry.parameter(i);
                let parameters_slice = unsafe {
                    std::slice::from_raw_parts(
                        &parameters as *const _ as *const u8,
                        std::mem::size_of_val(&parameters),
                    )
                };
                raytype_write_target[handle_size..handle_size + parameters_slice.len()]
                    .copy_from_slice(parameters_slice);
            }
        }
        Ok(commands! {
            let mut dst_buffer = RenderRes::new(buffer);
            if let Some(staging_buffer) = staging_buffer {
                let staging_buffer = RenderRes::new(staging_buffer);
                copy_buffer(&staging_buffer, &mut dst_buffer).await;
                retain!(staging_buffer);
            }
            dst_buffer
        })
    }
}
