use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use ash::{prelude::VkResult, vk};
use bevy::asset::{AssetId, Assets};

use crate::shader::{ShaderModule, SpecializedShader};
use crate::{
    deferred::{DeferredOperationTaskPool, Task},
    utils::SendBox,
    Device, HasDevice,
};

pub struct GraphicsPipeline {
    device: Device,
    pipeline: vk::Pipeline,
}
impl GraphicsPipeline {
    pub fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }
}
impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

impl super::Pipeline for GraphicsPipeline {
    type BuildInfo = BoxedGraphicsPipelineBuildInfo;
}

pub struct Builder<'a> {
    device: &'a Device,
    cache: vk::PipelineCache,
    info: vk::GraphicsPipelineCreateInfo,
}
impl Deref for Builder<'_> {
    type Target = vk::GraphicsPipelineCreateInfo;
    fn deref(&self) -> &Self::Target {
        &self.info
    }
}
impl DerefMut for Builder<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.info
    }
}

pub struct BuilderResult(VkResult<vk::Pipeline>);
impl<'a> Builder<'a> {
    pub fn build(&mut self) -> BuilderResult {
        let mut pipeline = vk::Pipeline::null();
        let result = unsafe {
            (self.device.fp_v1_0().create_graphics_pipelines)(
                self.device.handle(),
                self.cache,
                1,
                &self.info,
                std::ptr::null(),
                &mut pipeline,
            )
        };
        BuilderResult(result.result_with_success(pipeline))
    }
}

pub struct GraphicsPipelineBuildInfo<F>
where
    F: for<'a> Fn(Builder<'a>) -> BuilderResult,
{
    pub device: Device,
    pub stages: Vec<SpecializedShader>,
    pub builder: F,
}

pub struct BoxedGraphicsPipelineBuildInfo {
    pub device: Device,
    pub stages: Vec<SpecializedShader>,
    pub builder: Arc<dyn for<'a> Fn(Builder<'a>) -> BuilderResult + Send + Sync>,
}

impl super::PipelineBuildInfo for BoxedGraphicsPipelineBuildInfo {
    type Pipeline = GraphicsPipeline;

    fn build(
        &mut self,
        pool: &DeferredOperationTaskPool,
        assets: &Assets<ShaderModule>,
        cache: vk::PipelineCache,
    ) -> Option<Task<Self::Pipeline>> {
        let device = self.device.clone();
        let builder = self.builder.clone();
        let specialization_info = self
            .stages
            .iter()
            .map(SpecializedShader::as_raw)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let stages = self
            .stages
            .iter()
            .zip(specialization_info.iter())
            .map(|(shader, specialization_info)| {
                let module = assets.get(&shader.shader)?;
                Some(vk::PipelineShaderStageCreateInfo {
                    stage: shader.stage,
                    module: module.raw(),
                    p_name: shader.entry_point.as_ptr(),
                    p_specialization_info: specialization_info,
                    flags: shader.flags,
                    ..Default::default()
                })
            })
            .collect::<Option<Vec<_>>>()?
            .into_boxed_slice();

        let specialization_info = SendBox(specialization_info);
        let stages = SendBox(stages);
        Some(pool.schedule(move || {
            let stages = stages.into_inner();
            let mut info = vk::GraphicsPipelineCreateInfo::default();
            info.stage_count = stages.len() as u32;
            info.p_stages = stages.as_ptr();
            let result = builder(Builder {
                device: &device,
                cache,
                info,
            })
            .0?;
            drop(stages);
            drop(specialization_info);
            Ok(GraphicsPipeline {
                device,
                pipeline: result,
            })
        }))
    }
    fn all_shaders(&self) -> impl Iterator<Item = AssetId<ShaderModule>> {
        self.stages.iter().map(|shader| shader.shader.id())
    }
}
