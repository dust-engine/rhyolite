use crate::{utils::merge_iter::btree_map_union, Device, ImageLike};
use ash::vk;
use std::{
    alloc::Allocator,
    collections::{BTreeMap, HashMap},
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
pub struct GlobalContext {
    resources: Vec<GlobalContextResource>,
    pub command_buffer: vk::CommandBuffer,
    stage_index: u32,
}
impl GlobalContext {
    pub fn current_stage_index(&self) -> u32 {
        self.stage_index
    }
    pub fn add_res<'a, T>(&mut self, res: &'a mut T) -> Res<'a, T> {
        let id = self.resources.len() as u32;
        self.resources.push(GlobalContextResource::Memory);
        Res { id, inner: res }
    }
    pub fn add_image<'a, T: ImageLike>(
        &mut self,
        res: &'a mut T,
        initial_layout: vk::ImageLayout,
    ) -> Res<'a, T> {
        let id = self.resources.len() as u32;
        self.resources.push(GlobalContextResource::Image {
            initial_layout,
            image: res.raw_image(),
            subresource_range: res.subresource_range(),
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

pub trait GPUCommandFutureRecordAll: GPUCommandFuture + Sized {
    #[inline]
    fn record_all(
        mut self,
        //device: &crate::Device,
        command_buffer: vk::CommandBuffer,
    ) -> Self::Output {
        let mut ctx = GlobalContext {
            resources: Vec::new(),
            command_buffer,
            stage_index: 1,
        };

        let mut this = unsafe { std::pin::Pin::new_unchecked(&mut self) };
        this.as_mut().init(&mut ctx);

        let mut current_context = StageContext::default();
        this.as_mut().context(&mut current_context);

        let mut resource_accesses: Vec<Option<Access>> = vec![];
        let mut memory_barrier = vk::MemoryBarrier2::default();
        let mut image_barrier: Vec<vk::ImageMemoryBarrier2> = Vec::new();

        let (result, retained_state) = loop {
            if let Poll::Ready(result) = this.as_mut().record(&mut ctx) {
                break result;
            }

            let mut next_context = StageContext::default();
            this.as_mut().context(&mut next_context);

            for (id, before_access, after_access) in
                btree_map_union(&current_context.accesses, &next_context.accesses)
            {
                let res = &ctx.resources[*id as usize];
                let before_access =
                    before_access.or(resource_accesses.get(*id as usize).and_then(|a| a.as_ref()));
                match (res, before_access, after_access) {
                    (GlobalContextResource::Memory, Some(before_access), Some(after_access)) => {
                        get_memory_access(&mut memory_barrier, before_access, after_access)
                    }
                    (
                        GlobalContextResource::Image {
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
                        GlobalContextResource::Image {
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
                        GlobalContextResource::Image {
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
                        GlobalContextResource::Image {
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
                        if *initial_layout != new_layout && new_layout != vk::ImageLayout::UNDEFINED
                        {
                            image_barrier.push(vk::ImageMemoryBarrier2 {
                                src_stage_mask: vk::PipelineStageFlags2::NONE,
                                src_access_mask: vk::AccessFlags2::NONE,
                                dst_stage_mask: after_access.read_stages
                                    | after_access.write_stages,
                                dst_access_mask: after_access.read_access
                                    | after_access.write_access,
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
                drop(before_access);

                let id = *id as usize;
                let before_access = before_access.map(|a| a.clone());
                resource_accesses.resize(resource_accesses.len().max(id + 1), None);
                resource_accesses[id] = before_access.map(|a| a.clone());
            }
            if memory_barrier.src_stage_mask.is_empty()
                && memory_barrier.dst_stage_mask.is_empty()
                && image_barrier.is_empty()
            {
                // No need for pipeline barrier
                println!("-----pipeline barrier: no op-------");
            } else {
                
                println!("-----pipeline barrier: {:?} => {:?} -------", memory_barrier.src_access_mask, memory_barrier.dst_access_mask);
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
            memory_barrier = vk::MemoryBarrier2::default();
            image_barrier.clear();
            current_context = next_context;
            ctx.stage_index += 1;
        };
        println!("End, starting dropping state");
        drop(retained_state);
        println!("End, ending dropping state");
        result
    }
}
impl<T> GPUCommandFutureRecordAll for T where T: GPUCommandFuture {}

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
