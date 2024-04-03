use std::sync::Arc;

use ash::vk;
use bevy::asset::Assets;

use crate::{
    device::HasDevice,
    shader::{ShaderModule, SpecializedShader},
    utils::SendBox,
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
    fn from_built(
        _info: &mut ComputePipelineCreateInfo,
        item: <Self::BuildInfo as super::PipelineBuildInfo>::Pipeline,
    ) -> Self {
        ComputePipeline(Arc::new(item))
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
        let specialization_info = self.shader.raw_specialization_info();
        let stage = self
            .shader
            .raw_pipeline_stage(assets, &specialization_info)?;
        let send_box = SendBox((specialization_info, stage));
        let layout = self.layout.clone();
        let flags = self.flags;
        Some(pool.schedule(move || {
            let (specialization_info, mut stage) = send_box.into_inner();
            stage.p_specialization_info = &specialization_info; // fix up specialization info ptr after move.
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
