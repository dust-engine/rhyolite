use ash::vk;
use rhyolite::sync::GPUBorrowed;
use std::ops::Deref;

#[derive(Clone, Debug)]
pub struct ResourceState {
    pub read: Access,
    pub write: Access,
    pub queue_family: u32,
    pub layout: vk::ImageLayout,
}
impl Default for ResourceState {
    fn default() -> Self {
        Self {
            read: Default::default(),
            write: Default::default(),
            queue_family: u32::MAX,
            layout: vk::ImageLayout::default(),
        }
    }
}
pub type ResourceStateTable = ();

pub unsafe trait GPUResource {
    fn get_resource_state(&self, state_table: &ResourceStateTable) -> ResourceState;

    fn set_resource_state(&mut self, state_table: &mut ResourceStateTable, state: ResourceState);
}

// This is returned by the retain! macro.
pub struct GPUOwned<'a, T> {
    inner: &'a mut T,
}
impl<'a, T> GPUOwned<'a, T> {
    pub fn __retain(item: &'a mut T) -> Self {
        Self { inner: item }
    }
}
pub struct GPUOwnedResource<'a, T> {
    item: GPUOwned<'a, T>,
    state: ResourceState,
}
unsafe impl<T> GPUResource for GPUOwnedResource<'_, T> {
    fn get_resource_state(&self, _state_table: &ResourceStateTable) -> ResourceState {
        self.state.clone()
    }
    fn set_resource_state(&mut self, _state_table: &mut ResourceStateTable, state: ResourceState) {
        self.state = state;
    }
}
impl<T> Deref for GPUOwnedResource<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.item.inner
    }
}

impl<'a, T> GPUOwnedResource<'a, T> {
    pub fn new(item: GPUOwned<'a, T>) -> Self {
        Self {
            item,
            state: Default::default(),
        }
    }
}

/// GPU borrowed objects bundled with its associated resource states.
///
/// Wrapped objects will have its resource states tracked by the runtime such that pipeline barriers
/// or events will be automatically injected into the command stream to ensure hazard-free usage on
/// the GPU. The objects will also stay alive until the referencing operation has finished.
pub struct GPUBorrowedResource<T> {
    item: GPUBorrowed<T>,
    state: ResourceState,
}
unsafe impl<T> GPUResource for GPUBorrowedResource<T> {
    fn get_resource_state(&self, _state_table: &ResourceStateTable) -> ResourceState {
        self.state.clone()
    }
    fn set_resource_state(&mut self, _state_table: &mut ResourceStateTable, state: ResourceState) {
        self.state = state;
    }
}
impl<T> Deref for GPUBorrowedResource<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.item
    }
}

impl<T> GPUBorrowedResource<T> {
    pub fn new(item: T) -> Self {
        Self {
            item: GPUBorrowed::new(item),
            state: Default::default(),
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct Access {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}
impl Access {
    pub fn is_readonly(&self) -> bool {
        if self.access == vk::AccessFlags2::empty() {
            return false;
        }
        (self.access & !utils::ALL_READ_BITS).is_empty()
    }
}

impl ResourceState {
    pub fn get_barrier(
        &self,
        next: Access,
        with_layout_transition: bool,
    ) -> vk::MemoryBarrier2<'static> {
        let mut barrier = vk::MemoryBarrier2 {
            src_stage_mask: self.write.stage,
            src_access_mask: self.write.access,
            dst_stage_mask: next.stage,
            dst_access_mask: next.access,
            ..Default::default()
        };
        if self.write.access == vk::AccessFlags2::empty()
            && self.write.stage == vk::PipelineStageFlags2::empty()
        {
            // Resource was never accessed before.
            if with_layout_transition {
                // ... but we might still be syncing with it using a semaphore.
                // block the image layout transition so that it happens after the semaphore wait.
                barrier.src_access_mask = vk::AccessFlags2::empty();
                barrier.src_stage_mask = next.stage;
            } else {
                barrier = vk::MemoryBarrier2::default();
            }
        } else if next.is_readonly() {
            if let Some(ordering) = utils::compare_pipeline_stages(self.read.stage, next.stage) {
                if ordering.is_gt() {
                    barrier.src_stage_mask = self.read.stage;
                    barrier.src_access_mask = vk::AccessFlags2::empty();
                    barrier.dst_access_mask = vk::AccessFlags2::empty();
                } else {
                    // it has already been made visible at the desired stage
                    barrier = vk::MemoryBarrier2::default();
                }
            }
        } else {
            // The next stage will be writing.
            if self.read.stage != vk::PipelineStageFlags2::empty() {
                // This is a WAR hazard, which you would usually only need an execution dependency for.
                // meaning you wouldn't need to supply any memory barriers.
                barrier.src_stage_mask = self.read.stage;
                barrier.src_access_mask = vk::AccessFlags2::empty();
                if !with_layout_transition {
                    // When we do have a layout transition, you still need a memory barrier,
                    // but you don't need any access types in the src access mask. The layout transition
                    // itself is considered a write operation though, so you do need the destination
                    // access mask to be correct - or there would be a WAW hazard between the layout
                    // transition and the color attachment write.
                    barrier.dst_access_mask = vk::AccessFlags2::empty();
                }
            }
        }
        barrier
    }
    pub fn transition(&mut self, next: Access) {
        // the info is now available for all stages after next.stage, but not for stages before next.stage.
        if next.is_readonly() {
            self.read.stage = utils::earlier_stage(self.read.stage, next.stage);
        } else {
            self.write = next.clone();
            self.read = Access::default();
        }
    }
}

pub mod utils {
    use ash::vk;
    use std::cmp::Ordering;
    pub const ALL_WRITE_BITS: vk::AccessFlags2 = vk::AccessFlags2::from_raw(
        vk::AccessFlags2::SHADER_WRITE.as_raw()
            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw()
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
            | vk::AccessFlags2::TRANSFER_WRITE.as_raw()
            | vk::AccessFlags2::HOST_WRITE.as_raw()
            | vk::AccessFlags2::MEMORY_WRITE.as_raw()
            | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw()
            | vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR.as_raw()
            | vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR.as_raw()
            | vk::AccessFlags2::TRANSFORM_FEEDBACK_WRITE_EXT.as_raw()
            | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT.as_raw()
            | vk::AccessFlags2::COMMAND_PREPROCESS_WRITE_NV.as_raw()
            | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR.as_raw()
            | vk::AccessFlags2::MICROMAP_WRITE_EXT.as_raw()
            | vk::AccessFlags2::OPTICAL_FLOW_WRITE_NV.as_raw(),
    );
    pub const ALL_READ_BITS: vk::AccessFlags2 = vk::AccessFlags2::from_raw(
        vk::AccessFlags2::INDIRECT_COMMAND_READ.as_raw()
            | vk::AccessFlags2::INDEX_READ.as_raw()
            | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ.as_raw()
            | vk::AccessFlags2::UNIFORM_READ.as_raw()
            | vk::AccessFlags2::INPUT_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags2::SHADER_READ.as_raw()
            | vk::AccessFlags2::COLOR_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags2::TRANSFER_READ.as_raw()
            | vk::AccessFlags2::HOST_READ.as_raw()
            | vk::AccessFlags2::MEMORY_READ.as_raw()
            | vk::AccessFlags2::SHADER_SAMPLED_READ.as_raw()
            | vk::AccessFlags2::SHADER_STORAGE_READ.as_raw()
            | vk::AccessFlags2::VIDEO_DECODE_READ_KHR.as_raw()
            | vk::AccessFlags2::VIDEO_ENCODE_READ_KHR.as_raw()
            | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_READ_EXT.as_raw()
            | vk::AccessFlags2::CONDITIONAL_RENDERING_READ_EXT.as_raw()
            | vk::AccessFlags2::COMMAND_PREPROCESS_READ_NV.as_raw()
            | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR.as_raw()
            | vk::AccessFlags2::FRAGMENT_DENSITY_MAP_READ_EXT.as_raw()
            | vk::AccessFlags2::COLOR_ATTACHMENT_READ_NONCOHERENT_EXT.as_raw()
            | vk::AccessFlags2::DESCRIPTOR_BUFFER_READ_EXT.as_raw()
            | vk::AccessFlags2::INVOCATION_MASK_READ_HUAWEI.as_raw()
            | vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR.as_raw()
            | vk::AccessFlags2::MICROMAP_READ_EXT.as_raw()
            | vk::AccessFlags2::OPTICAL_FLOW_READ_NV.as_raw(),
    );
    const GRAPHICS_PIPELINE_ORDER: [vk::PipelineStageFlags2; 13] = [
        vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::PipelineStageFlags2::INDEX_INPUT,
        vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
        vk::PipelineStageFlags2::VERTEX_SHADER,
        vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
        vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
        vk::PipelineStageFlags2::GEOMETRY_SHADER,
        vk::PipelineStageFlags2::TRANSFORM_FEEDBACK_EXT,
        vk::PipelineStageFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR,
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
    ];
    const GRAPHICS_MESH_PIPELINE_ORDER: [vk::PipelineStageFlags2; 8] = [
        vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::PipelineStageFlags2::TASK_SHADER_EXT,
        vk::PipelineStageFlags2::MESH_SHADER_EXT,
        vk::PipelineStageFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR,
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
    ];
    const COMPUTE_PIPELINE_ORDER: [vk::PipelineStageFlags2; 2] = [
        vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::PipelineStageFlags2::COMPUTE_SHADER,
    ];
    const FRAGMENT_DENSITY_ORDER: [vk::PipelineStageFlags2; 2] = [
        vk::PipelineStageFlags2::FRAGMENT_DENSITY_PROCESS_EXT,
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
    ];
    const RAYTRACING_PIPELINE_ORDER: [vk::PipelineStageFlags2; 2] = [
        vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
    ];
    const ALL_ORDERS: [&[vk::PipelineStageFlags2]; 5] = [
        &GRAPHICS_PIPELINE_ORDER,
        &GRAPHICS_MESH_PIPELINE_ORDER,
        &COMPUTE_PIPELINE_ORDER,
        &FRAGMENT_DENSITY_ORDER,
        &RAYTRACING_PIPELINE_ORDER,
    ];
    /// Compare two pipeline stages. Returns [`Some(Ordering::Less)`] if `a` is earlier than `b`,
    /// [`Ordering::Equal`] if they are the same, [`Ordering::Greater`] if `a` is later than `b`,
    /// and [`None`] if they are not in the same time and are not mutually ordered.
    pub fn compare_pipeline_stages(
        a: vk::PipelineStageFlags2,
        b: vk::PipelineStageFlags2,
    ) -> Option<Ordering> {
        if a == b {
            return Some(std::cmp::Ordering::Equal);
        }
        for order in ALL_ORDERS.iter() {
            let first_index: Option<usize> = order.iter().position(|&x| a.contains(x));
            let second_index: Option<usize> = order.iter().position(|&x| b.contains(x));
            if let Some(first_index) = first_index {
                if let Some(second_index) = second_index {
                    return first_index.partial_cmp(&second_index);
                }
            }
        }
        None
    }
    pub fn earlier_stage(
        a: vk::PipelineStageFlags2,
        b: vk::PipelineStageFlags2,
    ) -> vk::PipelineStageFlags2 {
        if let Some(ordering) = compare_pipeline_stages(a, b) {
            if ordering.is_le() {
                a
            } else {
                b
            }
        } else {
            a | b
        }
    }
    pub fn later_stage(
        a: vk::PipelineStageFlags2,
        b: vk::PipelineStageFlags2,
    ) -> vk::PipelineStageFlags2 {
        if let Some(ordering) = compare_pipeline_stages(a, b) {
            if ordering.is_ge() {
                a
            } else {
                b
            }
        } else {
            a | b
        }
    }
}
