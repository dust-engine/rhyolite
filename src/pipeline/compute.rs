use std::sync::Arc;

use ash::vk;
use bevy::asset::Assets;

use crate::{
    device::HasDevice,
    shader::{ShaderModule, SpecializedShader},
    Device,
};

use super::{PipelineInner, PipelineLayout};

pub struct ComputePipelineCreateInfo {
    pub device: Device,
    pub shader: SpecializedShader,
    pub layout: Arc<PipelineLayout>,
    pub flags: vk::PipelineCreateFlags,
}

pub struct ComputePipeline(Arc<PipelineInner>);

impl super::Pipeline for ComputePipeline {
    type BuildInfo = ComputePipelineCreateInfo;
    const TYPE: vk::PipelineBindPoint = vk::PipelineBindPoint::COMPUTE;
    fn from_built(
        _info: &mut ComputePipelineCreateInfo,
        item: <Self::BuildInfo as super::PipelineBuildInfo>::Pipeline,
    ) -> Self {
        ComputePipeline(Arc::new(item))
    }

    fn as_raw(&self) -> vk::Pipeline {
        self.0.pipeline
    }
}

impl super::PipelineBuildInfo for ComputePipelineCreateInfo {
    type Pipeline = PipelineInner;
    fn build(
        &mut self,
        pool: &crate::DeferredOperationTaskPool,
        assets: &Assets<ShaderModule>,
        cache: vk::PipelineCache,
    ) -> Option<crate::Task<Self::Pipeline>> {
        let shader = self.shader.clone();
        let layout = self.layout.clone();
        let flags = self.flags;
        let module = assets.get(&shader.shader)?.raw();
        Some(pool.schedule(move || {
            let specialization_info = vk::SpecializationInfo::default()
                .data(&shader.specialization_info.data)
                .map_entries(&shader.specialization_info.entries);
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .flags(shader.flags)
                .stage(shader.stage)
                .module(module) // This will remain valid as long as we still have Handle<ShaderModule>
                .name(&shader.entry_point)
                .specialization_info(&specialization_info);
            let info = vk::ComputePipelineCreateInfo {
                flags,
                stage,
                layout: layout.raw(),
                ..Default::default()
            };
            let (result, pipeline) = unsafe {
                let mut pipeline = vk::Pipeline::null();
                let result = (layout.device().fp_v1_0().create_compute_pipelines)(
                    layout.device().handle(),
                    cache,
                    1,
                    &info,
                    std::ptr::null(),
                    &mut pipeline,
                );
                (result, pipeline)
            };
            drop(shader);
            result.result_with_success(PipelineInner {
                device: layout.device().clone(),
                pipeline,
            })
        }))
    }

    fn all_shaders(
        &self,
    ) -> impl Iterator<Item = bevy::prelude::AssetId<crate::shader::ShaderModule>> {
        std::iter::once(self.shader.shader.id())
    }
}
