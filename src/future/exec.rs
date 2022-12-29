use crate::{
    utils::merge_iter::btree_map_union, Device, ImageLike, SubmissionResource,
    SubmissionResourceType,
};
use ash::vk;
use std::{
    alloc::Allocator,
    collections::{BTreeMap, HashMap},
    pin::Pin,
    task::Poll,
};

use super::GPUCommandFuture;

pub struct Res<'a, T> {
    id: u32,
    inner: &'a mut T,
}
impl<'a, T> Res<'a, T> {
    pub fn new(id: u32, inner: &'a mut T) -> Self {
        Self { id, inner }
    }
    pub fn id(&self) -> u32 {
        self.id
    }
    pub fn inner(&self) -> &T {
        self.inner
    }
    pub fn inner_mut(&mut self) -> &mut T {
        self.inner
    }
}

pub struct ResImage<T: ImageLike> {
    id: u32,
    inner: T,
}
impl<T: ImageLike> ResImage<T> {
    pub fn new(id: u32, inner: T) -> Self {
        Self { id, inner }
    }
    pub fn id(&self) -> u32 {
        self.id
    }
}

#[derive(Clone)]
pub struct Access {
    ty: AccessType,

    read_stages: vk::PipelineStageFlags2,
    read_access: vk::AccessFlags2,
    write_stages: vk::PipelineStageFlags2,
    write_access: vk::AccessFlags2,
}

impl Access {
    pub fn expected_layout(&self) -> vk::ImageLayout {
        match self.ty {
            AccessType::Memory => panic!(),
            AccessType::Image {
                expected_layout, ..
            } => expected_layout,
        }
    }
    pub fn produced_layout(&self) -> vk::ImageLayout {
        match self.ty {
            AccessType::Memory => panic!(),
            AccessType::Image {
                produced_layout, ..
            } => produced_layout,
        }
    }

    pub fn expected_layout_mut(&mut self) -> &mut vk::ImageLayout {
        match &mut self.ty {
            AccessType::Memory => panic!(),
            AccessType::Image {
                expected_layout, ..
            } => expected_layout,
        }
    }
    pub fn produced_layout_mut(&mut self) -> &mut vk::ImageLayout {
        match &mut self.ty {
            AccessType::Memory => panic!(),
            AccessType::Image {
                produced_layout, ..
            } => produced_layout,
        }
    }
}
#[derive(Clone)]
pub enum AccessType {
    Memory,
    Image {
        /// Image layout when the command buffer starts
        expected_layout: vk::ImageLayout,

        /// Layout of the image produced by the command buffer
        produced_layout: vk::ImageLayout,
    },
}
impl Access {
    pub fn has_read(&self) -> bool {
        !self.read_stages.is_empty()
    }
    pub fn has_write(&self) -> bool {
        !self.write_stages.is_empty()
    }
}

enum GlobalContextResource {
    Memory,
    Image {
        initial_layout: vk::ImageLayout,
        image: vk::Image,
        subresource_range: vk::ImageSubresourceRange,
    },
}

/// One per command buffer record call. If multiple command buffers were merged together on the queue level,
/// this would be the same.
pub struct CommandBufferRecordContext<'a> {
    pub(crate) resources: &'a mut Vec<SubmissionResource>,
    // perhaps also a reference to the command buffer allocator
    pub(crate) command_buffer: vk::CommandBuffer,
    pub(crate) stage_index: &'a mut u32,
    pub(crate) last_stage: &'a mut Option<StageContext>,
}

impl<'b> CommandBufferRecordContext<'b> {
    pub fn current_stage_index(&self) -> u32 {
        *self.stage_index
    }
    pub fn add_res<'a, T>(&mut self, res: &'a mut T) -> Res<'a, T> {
        let id = self.resources.len() as u32;
        self.resources.push(SubmissionResource {
            ty: SubmissionResourceType::Memory,
            access: None,
        });
        Res { id, inner: res }
    }
    pub fn add_image<'a, T: ImageLike>(
        &mut self,
        res: &'a mut T,
        initial_layout: vk::ImageLayout,
    ) -> Res<'a, T> {
        let id = self.resources.len() as u32;
        self.resources.push(SubmissionResource {
            ty: SubmissionResourceType::Image {
                initial_layout,
                image: res.raw_image(),
                subresource_range: res.subresource_range(),
            },
            access: None,
        });
        Res { id, inner: res }
    }
}

#[derive(Default)]
pub struct StageContext {
    accesses: BTreeMap<u32, Access>,
}

impl StageContext {
    /// Declare a global memory write
    #[inline]
    pub fn write<T>(
        &mut self,
        res: &Res<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
    ) {
        let entry = self.accesses.entry(res.id).or_insert(Access {
            read_stages: vk::PipelineStageFlags2::NONE,
            read_access: vk::AccessFlags2::NONE,
            write_stages: vk::PipelineStageFlags2::NONE,
            write_access: vk::AccessFlags2::NONE,
            ty: AccessType::Memory,
        });
        entry.write_stages |= stages;
        entry.write_access |= accesses;
    }
    /// Declare a global memory read
    #[inline]
    pub fn read<T>(
        &mut self,
        res: &Res<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
    ) {
        let entry = self.accesses.entry(res.id).or_insert(Access {
            read_stages: vk::PipelineStageFlags2::NONE,
            read_access: vk::AccessFlags2::NONE,
            write_stages: vk::PipelineStageFlags2::NONE,
            write_access: vk::AccessFlags2::NONE,
            ty: AccessType::Memory,
        });
        entry.read_stages |= stages;
        entry.read_access |= accesses;
    }
    #[inline]
    pub fn write_image<T>(
        &mut self,
        res: &Res<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
        layout: vk::ImageLayout,
    ) {
        let entry = self.accesses.entry(res.id).or_insert(Access {
            read_stages: vk::PipelineStageFlags2::NONE,
            read_access: vk::AccessFlags2::NONE,
            write_stages: vk::PipelineStageFlags2::NONE,
            write_access: vk::AccessFlags2::NONE,
            ty: AccessType::Image {
                expected_layout: vk::ImageLayout::UNDEFINED,
                produced_layout: vk::ImageLayout::UNDEFINED,
            },
        });
        entry.write_stages |= stages;
        entry.write_access |= accesses;
        let produced_layout = entry.produced_layout_mut();
        *produced_layout = layout;
    }
    /// Declare a global memory read
    #[inline]
    pub fn read_image<T>(
        &mut self,
        res: &Res<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
        layout: vk::ImageLayout,
    ) {
        let entry = self.accesses.entry(res.id).or_insert(Access {
            read_stages: vk::PipelineStageFlags2::NONE,
            read_access: vk::AccessFlags2::NONE,
            write_stages: vk::PipelineStageFlags2::NONE,
            write_access: vk::AccessFlags2::NONE,
            ty: AccessType::Image {
                expected_layout: vk::ImageLayout::UNDEFINED,
                produced_layout: vk::ImageLayout::UNDEFINED,
            },
        });
        entry.read_stages |= stages;
        entry.read_access |= accesses;
        let expected_layout = entry.expected_layout_mut();
        *expected_layout = layout;

        let produced_layout = entry.produced_layout_mut();
        if *produced_layout == vk::ImageLayout::UNDEFINED {
            // Setting expected_layout = layout is going to automatically transition the image to the specified layout.
            // Unless the application writes to that image again in a different layout, the image will
            // retain its original layout.
            *produced_layout = layout;
        }
    }
    pub fn merge(&mut self, mut other: Self) {
        // TODO: merge accesses. Do we actually need to merge access within each resource?
        self.accesses.append(&mut other.accesses);
    }
}

impl<'b> CommandBufferRecordContext<'b> {
    fn add_barrier(
        prev: &StageContext,
        next: &StageContext,
        resources: &mut Vec<SubmissionResource>,
    ) {
        let mut memory_barrier = vk::MemoryBarrier2::default();
        let mut image_barrier: Vec<vk::ImageMemoryBarrier2> = Vec::new();
        for (id, before_access, after_access) in btree_map_union(&prev.accesses, &next.accesses) {
            let res = &resources[*id as usize];

            // If before_access is none, it means that the resource wasn't accessed in the prior stage.
            // Take the value from the stages before that. Otherwise, take the access from the prior stage.
            let before_access =
                before_access.or(resources.get(*id as usize).and_then(|a| a.access.as_ref()));
            match (&res.ty, before_access, after_access) {
                (SubmissionResourceType::Memory, Some(before_access), Some(after_access)) => {
                    get_memory_access(&mut memory_barrier, before_access, after_access)
                }
                (
                    SubmissionResourceType::Image {
                        initial_layout: _,
                        image,
                        subresource_range,
                    },
                    Some(before_access),
                    Some(after_access),
                ) if before_access.produced_layout() != after_access.expected_layout()
                    && after_access.expected_layout() != vk::ImageLayout::UNDEFINED =>
                {
                    // Image layout transition. Prev image is in VALID or UNDEFINED. Next image is VALID to be read.
                    let mut image_memory_barrier = vk::MemoryBarrier2::default();
                    get_memory_access(&mut image_memory_barrier, before_access, after_access);
                    image_barrier.push(vk::ImageMemoryBarrier2 {
                        src_stage_mask: image_memory_barrier.src_stage_mask,
                        src_access_mask: image_memory_barrier.src_access_mask,
                        dst_stage_mask: image_memory_barrier.dst_stage_mask,
                        dst_access_mask: image_memory_barrier.dst_access_mask,
                        old_layout: before_access.produced_layout(),
                        new_layout: after_access.expected_layout(),
                        image: *image,
                        subresource_range: subresource_range.clone(),
                        ..Default::default()
                    });
                }
                (
                    SubmissionResourceType::Image {
                        initial_layout: _,
                        image,
                        subresource_range,
                    },
                    Some(before_access),
                    Some(after_access),
                ) if after_access.expected_layout() == vk::ImageLayout::UNDEFINED
                    && after_access.produced_layout() != vk::ImageLayout::UNDEFINED =>
                {
                    // Image layout transition. Prev image is in VALID or UNDEFINED. Next image is VALID to be read.
                    let mut image_memory_barrier = vk::MemoryBarrier2::default();
                    get_memory_access(&mut image_memory_barrier, before_access, after_access);
                    image_barrier.push(vk::ImageMemoryBarrier2 {
                        src_stage_mask: image_memory_barrier.src_stage_mask,
                        src_access_mask: image_memory_barrier.src_access_mask,
                        dst_stage_mask: image_memory_barrier.dst_stage_mask,
                        dst_access_mask: image_memory_barrier.dst_access_mask,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: after_access.produced_layout(),
                        image: *image,
                        subresource_range: subresource_range.clone(),
                        ..Default::default()
                    });
                }
                (
                    SubmissionResourceType::Image {
                        initial_layout: _,
                        image: _,
                        subresource_range: _,
                    },
                    Some(before_access),
                    Some(after_access),
                ) => {
                    // Image without layout transition
                    get_memory_access(&mut memory_barrier, before_access, after_access)
                }
                (
                    SubmissionResourceType::Image {
                        initial_layout,
                        image,
                        subresource_range,
                    },
                    None,
                    Some(after_access),
                ) => {
                    let mut new_layout = after_access.expected_layout();
                    if new_layout == vk::ImageLayout::UNDEFINED {
                        new_layout = after_access.produced_layout();
                    }
                    if *initial_layout != new_layout && new_layout != vk::ImageLayout::UNDEFINED {
                        image_barrier.push(vk::ImageMemoryBarrier2 {
                            src_stage_mask: vk::PipelineStageFlags2::NONE,
                            src_access_mask: vk::AccessFlags2::NONE,
                            dst_stage_mask: after_access.read_stages | after_access.write_stages,
                            dst_access_mask: after_access.read_access | after_access.write_access,
                            old_layout: *initial_layout,
                            new_layout,
                            image: *image,
                            subresource_range: subresource_range.clone(),
                            ..Default::default()
                        });
                    }
                }
                _ => {}
            }

            let id = *id as usize;

            // If this stage has None access, this will write inherited access.
            // Otherwise, write the access for the current stage.
            resources[id].access = before_access.map(|a| a.clone());
        }

        if memory_barrier.src_stage_mask.is_empty()
            && memory_barrier.dst_stage_mask.is_empty()
            && image_barrier.is_empty()
        {
            // No need for pipeline barrier
            println!("-----pipeline barrier: no op-------");
        } else {
            println!(
                "-----pipeline barrier: {:?} => {:?} -------",
                memory_barrier.src_access_mask, memory_barrier.dst_access_mask
            );
            /*
            unsafe {
                device.cmd_pipeline_barrier2(
                    command_buffer,
                    &vk::DependencyInfo {
                        dependency_flags: vk::DependencyFlags::BY_REGION, // TODO
                        memory_barrier_count: 1,
                        p_memory_barriers: &memory_barrier,
                        ..Default::default()
                    },
                );
            }
            */
        }
    }
    pub(crate) fn record_one_step<T: GPUCommandFuture>(
        &mut self,
        mut fut: Pin<&mut T>,
    ) -> Poll<(T::Output, T::RetainedState)> {
        let mut next_stage = StageContext::default();
        fut.as_mut().context(&mut next_stage);
        if let Some(last_stage) = &self.last_stage {
            Self::add_barrier(last_stage, &next_stage, &mut self.resources);
        }

        let ret = fut.as_mut().record(self);
        *self.last_stage = Some(next_stage);
        *self.stage_index += 1;
        ret
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
