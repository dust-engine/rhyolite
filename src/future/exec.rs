use crate::{
    utils::merge_iter::btree_map_union, Device, ImageLike, SubmissionResource,
    SubmissionResourceType,
};
use ash::vk;
use bevy::reflect::erased_serde::__private::serde::__private::de;
use std::{
    alloc::Allocator,
    collections::{BTreeMap, HashMap},
    pin::Pin,
    task::Poll,
};

use super::GPUCommandFuture;

pub struct Res<'a, T> {
    access: Access,
    inner: &'a mut T,
}
impl<'a, T> Res<'a, T> {
    pub fn new(inner: &'a mut T) -> Self {
        Self {
            access: Access::default(),
            inner
        }
    }
    pub fn inner(&self) -> &T {
        self.inner
    }
    pub fn inner_mut(&mut self) -> &mut T {
        self.inner
    }
}

pub struct ResImage<'a, T> {
    res: Res<'a, T>,
    layout: vk::ImageLayout,
}
impl<'a, T> ResImage<'a, T> {
    pub fn new(inner: &'a mut T, initial_layout: vk::ImageLayout) -> Self {
        Self {
            res: Res::new(inner),
            layout: initial_layout
        }
    }
    pub fn inner(&self) -> &T {
        self.res.inner
    }
    pub fn inner_mut(&mut self) -> &mut T {
        self.res.inner
    }
}

#[derive(Clone, Default)]
pub struct Access {
    read_stages: vk::PipelineStageFlags2,
    read_access: vk::AccessFlags2,
    write_stages: vk::PipelineStageFlags2,
    write_access: vk::AccessFlags2,
}

#[derive(Clone, Default)]
pub struct ImageAccess {
    access: Access,
    prev_access: Access,
    prev_layout: vk::ImageLayout,
    layout: vk::ImageLayout,
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
    pub(crate) state: &'a mut CommandBufferRecordState,
    // perhaps also a reference to the command buffer allocator
    pub(crate) stage_index: &'a mut u32,
    pub(crate) last_stage: &'a mut Option<StageContext>,
}

impl<'b> CommandBufferRecordContext<'b> {
    pub fn current_stage_index(&self) -> u32 {
        *self.stage_index
    }
    pub fn add_res<'a, T>(&mut self, res: &'a mut T) -> Res<'a, T> {
        Res::new(res)
    }
    pub fn add_image<'a, T: ImageLike>(
        &mut self,
        res: &'a mut T,
        initial_layout: vk::ImageLayout,
    ) -> Res<'a, T> {
        Res::new(res)
    }
}

struct StageContextImage {
    image: vk::Image,
    subresource_range: vk::ImageSubresourceRange,
}
impl ImageLike for StageContextImage {
    fn raw_image(&self) -> vk::Image {
        self.image
    }

    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        self.subresource_range.clone()
    }
}
 
impl PartialEq for StageContextImage {
    fn eq(&self, other: &Self) -> bool {
        self.image == other.image &&
        self.subresource_range.aspect_mask == other.subresource_range.aspect_mask &&
        self.subresource_range.base_array_layer == other.subresource_range.base_array_layer &&
        self.subresource_range.base_mip_level == other.subresource_range.base_mip_level &&
        self.subresource_range.level_count == other.subresource_range.level_count &&
        self.subresource_range.layer_count == other.subresource_range.layer_count
    }
}
impl Eq for StageContextImage{}

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
        match self.subresource_range.base_array_layer.cmp(&other.subresource_range.base_array_layer) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        match self.subresource_range.layer_count.cmp(&other.subresource_range.layer_count) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        match self.subresource_range.base_mip_level.cmp(&other.subresource_range.base_mip_level) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        match self.subresource_range.level_count.cmp(&other.subresource_range.level_count) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        self.subresource_range.aspect_mask.cmp(&other.subresource_range.aspect_mask)
    }
}


#[derive(Default)]
pub struct StageContext {
    global_access: Access,
    image_accesses: BTreeMap<StageContextImage, ImageAccess>,
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
        self.global_access.write_stages |= stages;
        self.global_access.write_access |= accesses;
    }
    /// Declare a global memory read
    #[inline]
    pub fn read<T>(
        &mut self,
        res: &Res<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
    ) {
        self.global_access.read_stages |= stages;
        self.global_access.read_access |= accesses;
    }
    #[inline]
    pub fn write_image<T>(
        &mut self,
        res: &mut ResImage<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
        layout: vk::ImageLayout,
    ) where T: ImageLike {
        let entry = self.image_accesses.entry(StageContextImage {
            image: res.inner().raw_image(),
            subresource_range: res.inner().subresource_range(),
        }).or_insert(ImageAccess::default());

        entry.prev_access = res.res.access.clone();
        entry.prev_layout = res.layout;
        entry.layout = layout;

        entry.access.write_stages |= stages;
        entry.access.write_access |= accesses;

        res.layout = layout;
        res.res.access = Access {
            write_access: accesses,
            write_stages: stages,
            ..Default::default()
        };
    }
    /// Declare a global memory read
    #[inline]
    pub fn read_image<T>(
        &mut self,
        res: &mut ResImage<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
        layout: vk::ImageLayout,
    ) where T: ImageLike {
        let entry = self.image_accesses.entry(StageContextImage {
            image: res.inner().raw_image(),
            subresource_range: res.inner().subresource_range(),
        }).or_insert(ImageAccess::default());
        
        entry.prev_access = res.res.access.clone();
        entry.prev_layout = res.layout;
        entry.access.read_stages |= stages;
        entry.access.read_access |= accesses;
        entry.layout = layout;

        res.layout = layout;
        res.res.access = Access {
            read_access: accesses,
            read_stages: stages,
            ..Default::default()
        };

    }
    pub fn merge(&mut self, mut other: Self) {
        self.global_access.read_access |= other.global_access.read_access;
        self.global_access.read_stages |= other.global_access.read_stages;
        self.global_access.write_access |= other.global_access.write_access;
        self.global_access.write_stages |= other.global_access.write_stages;
        // TODO: Merge image accesses.
    }
}

#[derive(Default)]
pub struct CommandBufferRecordState {
    
}

impl CommandBufferRecordState {
    fn add_barrier(
        &mut self,
        prev: &StageContext,
        next: &StageContext,
        cmd_pipeline_barrier: impl FnOnce(&vk::DependencyInfo) -> ()
    ) {
        let mut global_memory_barrier = vk::MemoryBarrier2::default();
        let mut image_barrier: Vec<vk::ImageMemoryBarrier2> = Vec::new();

        // Set the global memory barrier.
        get_memory_access(&mut global_memory_barrier, &prev.global_access, &next.global_access);



        for (image, image_access) in next.image_accesses.iter() {
            if image_access.prev_layout == image_access.layout {
                get_memory_access(&mut global_memory_barrier, &image_access.prev_access, &image_access.access);
            } else {
                // Needs image layout transfer.
                let mut b  = vk::MemoryBarrier2::default();
                get_memory_access(&mut b, &image_access.prev_access, &image_access.access);
                let o = vk::ImageMemoryBarrier2 {
                    src_access_mask: b.src_access_mask,
                    src_stage_mask: b.src_stage_mask,
                    dst_access_mask: b.dst_access_mask,
                    dst_stage_mask: b.dst_stage_mask,
                    image: image.image,
                    subresource_range: image.subresource_range,
                    old_layout: image_access.prev_layout,
                    new_layout: image_access.layout,
                    ..Default::default()
                };
                if b.src_access_mask.is_empty() {
                    continue;
                }
                image_barrier.push(o);
            }
        }
        let mut dep = vk::DependencyInfo {
            dependency_flags: vk::DependencyFlags::BY_REGION, // TODO
            ..Default::default()
        };
        if !global_memory_barrier.dst_access_mask.is_empty() ||
        !global_memory_barrier.src_access_mask.is_empty() ||
        !global_memory_barrier.dst_stage_mask.is_empty() ||
        !global_memory_barrier.src_stage_mask.is_empty() {
            dep.memory_barrier_count = 1;
            dep.p_memory_barriers = &global_memory_barrier;
        }
        if !image_barrier.is_empty() {
            dep.image_memory_barrier_count = image_barrier.len() as u32;
            dep.p_image_memory_barriers = image_barrier.as_ptr();
        }

        if dep.memory_barrier_count > 0 || dep.buffer_memory_barrier_count > 0 || dep.image_memory_barrier_count > 0 {
            cmd_pipeline_barrier(&dep);
        }
    }
}
impl<'b> CommandBufferRecordContext<'b> {
    pub(crate) fn record_one_step<T: GPUCommandFuture>(
        &mut self,
        mut fut: Pin<&mut T>,
    ) -> Poll<(T::Output, T::RetainedState)> {
        let mut next_stage = StageContext::default();
        fut.as_mut().context(&mut next_stage);
        if let Some(last_stage) = &self.last_stage {
            self.state.add_barrier(last_stage, &next_stage, |_| {
                // TODO: noop for now
            });
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
        let memory_barrier = unsafe {
            &*dep.p_memory_barriers
        };
        assert_eq!(memory_barrier.src_stage_mask, src_stage_mask);
        assert_eq!(memory_barrier.src_access_mask, src_access_mask);
        assert_eq!(memory_barrier.dst_stage_mask, dst_stage_mask);
        assert_eq!(memory_barrier.dst_access_mask, dst_access_mask);
        assert_ne!(memory_barrier.src_stage_mask, vk::PipelineStageFlags2::NONE);
        assert_ne!(memory_barrier.dst_stage_mask, vk::PipelineStageFlags2::NONE);
    }

    
    use vk::PipelineStageFlags2 as S;
    use vk::AccessFlags2 as A;
    enum ReadWrite {
        Read(vk::PipelineStageFlags2, vk::AccessFlags2),
        Write(vk::PipelineStageFlags2, vk::AccessFlags2),
        ReadWrite(Access),
        None,
    }
    impl ReadWrite {
        fn to_access(&self) -> Access {
            match &self {
                ReadWrite::Read(read_stages, read_access) => Access { read_stages: *read_stages, read_access: *read_access, ..Default::default() },
                ReadWrite::Write(write_stages, write_access) => Access { write_stages: *write_stages, write_access: *write_access, ..Default::default() },
                ReadWrite::ReadWrite(a) => a.clone(),
                ReadWrite::None => Default::default()
            }
        }
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
                (S::COMPUTE_SHADER, A::SHADER_STORAGE_READ, S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
            ), // RaW
            (
                [
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                ],
                (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
            ), // WaW
            (
                [
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                    ReadWrite::Read(S::INDEX_INPUT, A::INDEX_READ),
                ],
                (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, S::INDEX_INPUT, A::INDEX_READ),
            ), // Dispatch writes into a storage buffer. Draw consumes that buffer as an index buffer.
        ];

        for test_case in test_cases.into_iter() {
            let mut state = CommandBufferRecordState::default();

            let mut called = false;
            state.add_barrier(&StageContext {
                global_access: test_case.0[0].to_access(),
                ..Default::default()
            }, &StageContext {
                global_access: test_case.0[1].to_access(),
                ..Default::default()
            }, |dep| {
                called = true;
                assert_global(
                    dep,
                    test_case.1.0,
                    test_case.1.1,
                    test_case.1.2,
                    test_case.1.3,
                );
            });
            if test_case.1.0 != vk::PipelineStageFlags2::NONE && test_case.1.2 != vk::PipelineStageFlags2::NONE {
                assert!(called);
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
                        write_access: A::SHADER_STORAGE_WRITE
                    }),
                    ReadWrite::Read(S::COMPUTE_SHADER, A::SHADER_STORAGE_READ),
                ],
                [
                    (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, S::COMPUTE_SHADER, A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE),
                    (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, S::COMPUTE_SHADER, A::SHADER_STORAGE_READ),
                ]
            ),
            (
                [
                    ReadWrite::Read(S::COMPUTE_SHADER, A::SHADER_STORAGE_READ),
                    ReadWrite::ReadWrite(Access {
                        read_stages: S::COMPUTE_SHADER,
                        read_access: A::SHADER_STORAGE_READ,
                        write_stages: S::COMPUTE_SHADER,
                        write_access: A::SHADER_STORAGE_WRITE
                    }),
                    ReadWrite::Write(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                ],
                [
                    (S::COMPUTE_SHADER, A::empty(), S::COMPUTE_SHADER, A::empty()),
                    (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                ]
            ),
        ];
        for test_case in test_cases.into_iter() {
            let mut state = CommandBufferRecordState::default();

            let mut called = false;
            state.add_barrier(&StageContext {
                global_access: test_case.0[0].to_access(),
                ..Default::default()
            }, &StageContext {
                global_access: test_case.0[1].to_access(),
                ..Default::default()
            }, |dep| {
                called = true;
                assert_global(
                    dep,
                    test_case.1[0].0,
                    test_case.1[0].1,
                    test_case.1[0].2,
                    test_case.1[0].3,
                );
            });
            
            if test_case.1[0].0 != vk::PipelineStageFlags2::NONE && test_case.1[0].2 != vk::PipelineStageFlags2::NONE {
                assert!(called);
            }
            called = false;
            state.add_barrier(&StageContext {
                global_access: test_case.0[1].to_access(),
                ..Default::default()
            }, &StageContext {
                global_access: test_case.0[2].to_access(),
                ..Default::default()
            }, |dep| {
                called = true;
                assert_global(
                    dep,
                    test_case.1[1].0,
                    test_case.1[1].1,
                    test_case.1[1].2,
                    test_case.1[1].3,
                );
            });
            if test_case.1[1].0 != vk::PipelineStageFlags2::NONE && test_case.1[1].2 != vk::PipelineStageFlags2::NONE {
                assert!(called);
            }
        }
    }

    #[test]
    fn c2c_image_tests() {
        let mut state = CommandBufferRecordState::default();
        let image: vk::Image = unsafe { std::mem::transmute(123_usize)};
        let mut stage_image = StageContextImage {
            image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            }
        };
        let image2: vk::Image = unsafe { std::mem::transmute(456_usize)};
        let mut stage_image2 = StageContextImage {
            image: image2,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            }
        };


        {
            let mut stage_image_res = ResImage::new(&mut stage_image, vk::ImageLayout::UNDEFINED);
            // First dispatch writes to a storage image, second dispatch reads from that storage image.
            let mut stage1 = StageContext::default();
            stage1.write_image(&mut stage_image_res, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::ImageLayout::GENERAL);
    
    
            let mut stage2 = StageContext::default();
            stage2.read_image(&mut stage_image_res, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_READ, vk::ImageLayout::GENERAL);
    
            let mut called = false;
            state.add_barrier(&stage1, &stage2, |dep| {
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
            let mut stage_image_res = ResImage::new(&mut stage_image, vk::ImageLayout::UNDEFINED);
            // Dispatch writes into a storage image. Draw samples that image in a fragment shader.
            let mut stage1 = StageContext::default();
            stage1.write_image(&mut stage_image_res, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::ImageLayout::GENERAL);
    
    
            let mut stage2 = StageContext::default();
            stage2.read_image(&mut stage_image_res, vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ, vk::ImageLayout::READ_ONLY_OPTIMAL);
    
            let mut called = false;
            state.add_barrier(&stage1, &stage2, |dep| {
                called = true;
                assert_eq!(dep.memory_barrier_count, 0);
                assert_eq!(dep.buffer_memory_barrier_count, 0);
                assert_eq!(dep.image_memory_barrier_count, 1);
                let image_memory_barrier = unsafe { &*dep.p_image_memory_barriers};
                assert_eq!(image_memory_barrier.old_layout, vk::ImageLayout::GENERAL);
                assert_eq!(image_memory_barrier.new_layout, vk::ImageLayout::READ_ONLY_OPTIMAL);
                assert_eq!(image_memory_barrier.src_stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
                assert_eq!(image_memory_barrier.src_access_mask, vk::AccessFlags2::SHADER_STORAGE_WRITE);
                assert_eq!(image_memory_barrier.dst_stage_mask, vk::PipelineStageFlags2::FRAGMENT_SHADER);
                assert_eq!(image_memory_barrier.dst_access_mask, vk::AccessFlags2::SHADER_SAMPLED_READ);
            });
            assert!(called);
        }

        {
            // Tests that image access info are retained across stages.
            let mut stage_image_res = ResImage::new(&mut stage_image, vk::ImageLayout::UNDEFINED);
            let mut stage_image_res2 = ResImage::new(&mut stage_image2, vk::ImageLayout::UNDEFINED);
            // Stage 1 is a compute shader which writes into a buffer and an image.
            // Stage 2 is a graphics pass which reads the buffer as the vertex input and writes to another image.
            // Stage 3 is a compute shader, reads both images, and writes into a buffer.
            // Stage 4 is a compute shader that reads the buffer.
            let mut stage1 = StageContext::default();
            stage1.global_access.write_access |= vk::AccessFlags2::SHADER_STORAGE_WRITE;
            stage1.global_access.write_stages |= vk::PipelineStageFlags2::COMPUTE_SHADER;
            stage1.write_image(&mut stage_image_res, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::ImageLayout::GENERAL);
    
    
            let mut stage2 = StageContext::default();
            stage2.global_access.read_access |= vk::AccessFlags2::VERTEX_ATTRIBUTE_READ;
            stage2.global_access.read_stages |= vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT;
            stage2.write_image(&mut stage_image_res2, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            

            let mut called = false;
            state.add_barrier(&stage1, &stage2, |dep| {
                called = true;
                assert_global(dep, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,vk::AccessFlags2::VERTEX_ATTRIBUTE_READ);
            });
            assert!(called);

            let mut stage3 = StageContext::default();
            stage3.read_image(&mut stage_image_res, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_READ, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            stage3.read_image(&mut stage_image_res2, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_READ, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            stage3.global_access.write_access |= vk::AccessFlags2::SHADER_STORAGE_WRITE;
            stage3.global_access.write_stages |= vk::PipelineStageFlags2::COMPUTE_SHADER;
    
            called = false;
            state.add_barrier(&stage2, &stage3, |dep| {
                called = true;
                assert_eq!(dep.memory_barrier_count, 0);
                assert_eq!(dep.buffer_memory_barrier_count, 0);
                assert_eq!(dep.image_memory_barrier_count, 2);
                let image_memory_barriers = unsafe {
                    std::slice::from_raw_parts(dep.p_image_memory_barriers, dep.image_memory_barrier_count as usize)
                };
                {
                    let image_memory_barrier = image_memory_barriers.iter().find(|a| a.image == image2).unwrap();
                    assert_eq!(image_memory_barrier.old_layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
                    assert_eq!(image_memory_barrier.new_layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                    assert_eq!(image_memory_barrier.src_stage_mask, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
                    assert_eq!(image_memory_barrier.src_access_mask, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE);
                    assert_eq!(image_memory_barrier.dst_stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
                    assert_eq!(image_memory_barrier.dst_access_mask, vk::AccessFlags2::SHADER_STORAGE_READ);
                }
                {
                    let image_memory_barrier = image_memory_barriers.iter().find(|a| a.image == image).unwrap();
                    assert_eq!(image_memory_barrier.old_layout, vk::ImageLayout::GENERAL);
                    assert_eq!(image_memory_barrier.new_layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                    assert_eq!(image_memory_barrier.src_stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
                    assert_eq!(image_memory_barrier.src_access_mask, vk::AccessFlags2::SHADER_STORAGE_WRITE);
                    assert_eq!(image_memory_barrier.dst_stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
                    assert_eq!(image_memory_barrier.dst_access_mask, vk::AccessFlags2::SHADER_STORAGE_READ);
                }
            });
            assert!(called);
        }
    }
}
