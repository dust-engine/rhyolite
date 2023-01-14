use crate::{
    utils::merge_iter::btree_map_union, Device, ImageLike,
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
    last_accessed_stage_index: u32,
    prev_stage_access: Access,
    current_stage_access: Access,
    inner: &'a mut T,
}
impl<'a, T> Res<'a, T> {
    pub fn new(inner: &'a mut T) -> Self {
        Self {
            last_accessed_stage_index: 0,
            prev_stage_access: Access::default(),
            current_stage_access: Access::default(),
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
    old_layout: vk::ImageLayout,
    layout: vk::ImageLayout,
}
impl<'a, T> ResImage<'a, T> {
    pub fn new(inner: &'a mut T, initial_layout: vk::ImageLayout) -> Self {
        Self {
            res: Res::new(inner),
            layout: initial_layout,
            old_layout: initial_layout,
        }
    }
    pub fn inner(&self) -> &T {
        self.res.inner
    }
    pub fn inner_mut(&mut self) -> &mut T {
        self.res.inner
    }
}

/// One per command buffer record call. If multiple command buffers were merged together on the queue level,
/// this would be the same.
pub struct CommandBufferRecordContext {
    // perhaps also a reference to the command buffer allocator
    pub(crate) stage_index: u32,
}

impl CommandBufferRecordContext {
    pub fn new(stage_index: u32) -> Self {
        Self {
            stage_index
        }
    }
    pub fn current_stage_index(&self) -> u32 {
        self.stage_index
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


pub struct StageContext {
    stage_index: u32,
    global_access: vk::MemoryBarrier2,
    image_accesses: BTreeMap<StageContextImage, (vk::MemoryBarrier2, vk::ImageLayout, vk::ImageLayout)>,
}

impl StageContext {
    pub fn new(index: u32) -> Self {
        Self {
            stage_index: index,
            global_access: vk::MemoryBarrier2::default(),
            image_accesses: BTreeMap::new()
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
        if res.last_accessed_stage_index < self.stage_index {
            res.prev_stage_access = std::mem::replace(&mut res.current_stage_access, Default::default());
        }
        get_memory_access(&mut self.global_access, &res.prev_stage_access, &Access {
            write_access: accesses,
            write_stages: stages,
            ..Default::default()
        });
        res.current_stage_access.write_access |= accesses;
        res.current_stage_access.write_stages |= stages;
        res.last_accessed_stage_index = self.stage_index;
    }
    /// Declare a global memory read
    #[inline]
    pub fn read<T>(
        &mut self,
        res: &mut Res<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
    ) {
        if res.last_accessed_stage_index < self.stage_index {
            res.prev_stage_access = std::mem::replace(&mut res.current_stage_access, Default::default());
        }
        get_memory_access(&mut self.global_access, &res.prev_stage_access, &Access {
            read_access: accesses,
            read_stages: stages,
            ..Default::default()
        });
        res.current_stage_access.read_access |= accesses;
        res.current_stage_access.read_stages |= stages;
        res.last_accessed_stage_index = self.stage_index;
    }
    #[inline]
    pub fn write_image<T>(
        &mut self,
        res: &mut ResImage<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
        layout: vk::ImageLayout,
    ) where T: ImageLike {
        let (barrier, old_layout, new_layout) = self.image_accesses.entry(StageContextImage {
            image: res.inner().raw_image(),
            subresource_range: res.inner().subresource_range(),
        }).or_insert((Default::default(), res.old_layout, layout));
        
        if res.res.last_accessed_stage_index < self.stage_index {
            res.res.prev_stage_access = std::mem::replace(&mut res.res.current_stage_access, Default::default());
            res.old_layout = res.layout;
        } else {
            assert_eq!(res.layout, layout, "Layout mismatch.");
        }
        get_memory_access(barrier, &res.res.prev_stage_access, &Access {
            write_access: accesses,
            write_stages: stages,
            ..Default::default()
        });

        res.res.current_stage_access.write_access |= accesses;
        res.res.current_stage_access.write_stages |= stages;
        res.res.last_accessed_stage_index = self.stage_index;
        res.layout = layout;
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
        let (barrier, old_layout, new_layout) = self.image_accesses.entry(StageContextImage {
            image: res.inner().raw_image(),
            subresource_range: res.inner().subresource_range(),
        }).or_insert((Default::default(), res.old_layout, layout));
        
        if res.res.last_accessed_stage_index < self.stage_index {
            res.res.prev_stage_access = std::mem::replace(&mut res.res.current_stage_access, Default::default());
            res.old_layout = res.layout;
        } else {
            assert_eq!(res.layout, layout, "Layout mismatch.");
        }
        get_memory_access(barrier, &res.res.prev_stage_access, &Access {
            read_access: accesses,
            read_stages: stages,
            ..Default::default()
        });

        res.res.current_stage_access.read_access |= accesses;
        res.res.current_stage_access.read_stages |= stages;
        res.res.last_accessed_stage_index = self.stage_index;
        res.layout = layout;
    }
    pub fn merge(&mut self, mut other: Self) {
        todo!()
        // TODO: Merge image accesses.
    }
}

impl CommandBufferRecordContext {
    pub fn record_one_step<T: GPUCommandFuture>(
        &mut self,
        mut fut: Pin<&mut T>,
    ) -> Poll<(T::Output, T::RetainedState)> {
        let mut next_stage = StageContext::new(self.stage_index);
        fut.as_mut().context(&mut next_stage);
        Self::add_barrier(&next_stage, |_| {
            // TODO: noop for now
        });

        let ret = fut.as_mut().record(self);
        self.stage_index += 1;
        ret
    }
    fn add_barrier(
        next_stage: &StageContext,
        cmd_pipeline_barrier: impl FnOnce(&vk::DependencyInfo) -> ()
    ) {
        let mut global_memory_barrier = next_stage.global_access.clone();
        let mut image_barrier: Vec<vk::ImageMemoryBarrier2> = Vec::new();

        // Set the global memory barrier.



        for (image, (barrier, old_layout, new_layout)) in next_stage.image_accesses.iter() {
            if old_layout == new_layout {
                global_memory_barrier.dst_access_mask |= barrier.dst_access_mask;
                global_memory_barrier.src_access_mask |= barrier.src_access_mask;
                global_memory_barrier.dst_stage_mask |= barrier.dst_stage_mask;
                global_memory_barrier.src_stage_mask |= barrier.src_stage_mask;
            } else {
                // Needs image layout transfer.
                let o = vk::ImageMemoryBarrier2 {
                    src_access_mask: barrier.src_access_mask,
                    src_stage_mask: barrier.src_stage_mask,
                    dst_access_mask: barrier.dst_access_mask,
                    dst_stage_mask: barrier.dst_stage_mask,
                    image: image.image,
                    subresource_range: image.subresource_range,
                    old_layout: *old_layout,
                    new_layout: *new_layout,
                    ..Default::default()
                };
                if barrier.src_access_mask.is_empty() {
                    // TODO: This is a bit questionable
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



#[derive(Clone, Default)]
struct Access {
    read_stages: vk::PipelineStageFlags2,
    read_access: vk::AccessFlags2,
    write_stages: vk::PipelineStageFlags2,
    write_access: vk::AccessFlags2,
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
        fn stage<T>(&self, stage_ctx: &mut StageContext, res: &mut Res<T>) {
            match &self {
                ReadWrite::Read(stage, access) => stage_ctx.read(res, *stage, *access),
                ReadWrite::Write(stage, access) => stage_ctx.write(res, *stage, *access),
                ReadWrite::ReadWrite(access) => {
                    stage_ctx.read(res, access.read_stages, access.read_access);
                    stage_ctx.write(res, access.write_stages, access.write_access);
                },
                ReadWrite::None => (),
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

        let mut buffer: vk::Buffer = unsafe { std::mem::transmute(123_u64)};
        for test_case in test_cases.into_iter() {
            let mut buffer = Res::new(&mut buffer);
            let mut stage1 = StageContext::new(0);
            test_case.0[0].stage(&mut stage1, &mut buffer);

            let mut stage2 = StageContext::new(1);
            test_case.0[1].stage(&mut stage2, &mut buffer);

            let mut called = false;
            CommandBufferRecordContext::add_barrier(&stage2, |dep| {
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
        let mut buffer: vk::Buffer = unsafe { std::mem::transmute(123_u64)};
        for test_case in test_cases.into_iter() {
            let mut buffer = Res::new(&mut buffer);

            let mut stage1 = StageContext::new(0);
            test_case.0[0].stage(&mut stage1, &mut buffer);

            let mut stage2 = StageContext::new(1);
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
            
            if test_case.1[0].0 != vk::PipelineStageFlags2::NONE && test_case.1[0].2 != vk::PipelineStageFlags2::NONE {
                assert!(called);
            }
            called = false;

            
            let mut stage3 = StageContext::new(2);
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
            if test_case.1[1].0 != vk::PipelineStageFlags2::NONE && test_case.1[1].2 != vk::PipelineStageFlags2::NONE {
                assert!(called);
            }
        }
    }

    #[test]
    fn c2c_image_tests() {
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
        
        let mut buffer1: vk::Buffer = unsafe { std::mem::transmute(4562_usize)};
        let mut buffer2: vk::Buffer = unsafe { std::mem::transmute(578_usize)};


        {
            let mut stage_image_res = ResImage::new(&mut stage_image, vk::ImageLayout::GENERAL);
            // First dispatch writes to a storage image, second dispatch reads from that storage image.
            let mut stage1 = StageContext::new(0);
            stage1.write_image(&mut stage_image_res, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::ImageLayout::GENERAL);
    
            let mut stage2 = StageContext::new(1);
            stage2.read_image(&mut stage_image_res, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_READ, vk::ImageLayout::GENERAL);
    
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
            let mut stage1 = StageContext::new(0);
            stage1.write_image(&mut stage_image_res, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::ImageLayout::GENERAL);
    
    
            let mut stage2 = StageContext::new(1);
            stage2.read_image(&mut stage_image_res, vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ, vk::ImageLayout::READ_ONLY_OPTIMAL);
    
            let mut called = false;
            CommandBufferRecordContext::add_barrier(&stage2, |dep| {
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
            let mut stage_image_res = ResImage::new(&mut stage_image, vk::ImageLayout::GENERAL);
            let mut stage_image_res2 = ResImage::new(&mut stage_image2, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            let mut buffer_res1 = Res::new(&mut buffer1);
            let mut buffer_res2 = Res::new(&mut buffer2);
            // Stage 1 is a compute shader which writes into buffer1 and an image.
            // Stage 2 is a graphics pass which reads buffer1 as the vertex input and writes to another image.
            // Stage 3 is a compute shader, reads both images, and writes into buffer2.
            // Stage 4 is a compute shader that reads buffer2.
            let mut stage1 = StageContext::new(0);
            stage1.write(&mut buffer_res1, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE);
            stage1.write_image(&mut stage_image_res, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::ImageLayout::GENERAL);
    
    
            let mut stage2 = StageContext::new(1);
            stage2.read(&mut buffer_res1, vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT, vk::AccessFlags2::VERTEX_ATTRIBUTE_READ);
            stage2.write_image(&mut stage_image_res2, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            

            let mut called = false;
            CommandBufferRecordContext::add_barrier(&stage2, |dep| {
                called = true;
                assert_global(dep, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,vk::AccessFlags2::VERTEX_ATTRIBUTE_READ);
            });
            assert!(called);

            let mut stage3 = StageContext::new(2);
            stage3.read_image(&mut stage_image_res, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_READ, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            stage3.read_image(&mut stage_image_res2, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_READ, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            stage3.write(&mut buffer_res2, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE);
    
            called = false;
            CommandBufferRecordContext::add_barrier(&stage3, |dep| {
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


            let mut stage4 = StageContext::new(3);
            stage4.read(&mut buffer_res2, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_READ);
            called = false;
            CommandBufferRecordContext::add_barrier(&stage4, |dep| {
                called = true;
                assert_global(dep, vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::PipelineStageFlags2::COMPUTE_SHADER,vk::AccessFlags2::SHADER_STORAGE_READ);
            });
            assert!(called);
        }
    }
}
