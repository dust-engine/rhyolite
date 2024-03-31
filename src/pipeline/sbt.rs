use std::alloc::Layout;

use crate::shader::SpecializedShader;

use super::{
    ray_tracing::{HitgroupHandle, RayTracingPipelineBuildInfoCommon, RayTracingPipelineManager},
    PipelineCache,
};

pub struct HitgroupSbtLayout {
    /// The layout for one raytype.
    /// | Raytype 1                                    |
    /// | shader_handles | inline_parameters | padding |
    /// | <--              size           -> | align   |
    one_raytype: Layout,

    // The layout for one entry with all its raytypes
    /// | Raytype 1                                    | Raytype 2                                    |
    /// | shader_handles | inline_parameters | padding | shader_handles | inline_parameters | padding |
    /// | <---                                      size                               ---> |  align  |
    one_entry: Layout,

    /// The size of the shader group handles, padded.
    /// | Raytype 1                                    |
    /// | shader_handles | inline_parameters | padding |
    /// | <--- size ---> |
    handle_size: usize,

    /// The number of hitgroup raytypes.
    num_raytypes: u32,
}

pub struct SbtManager {
    pipeline_manager: RayTracingPipelineManager,
    hitgroup_layout: HitgroupSbtLayout,
}

pub struct SbtHandle(u32);

impl SbtManager {
    pub fn new(
        pipeline_manager: RayTracingPipelineManager,
        hitgroup_layout: HitgroupSbtLayout,
    ) -> Self {
        Self {
            pipeline_manager,
            hitgroup_layout,
        }
    }
}
