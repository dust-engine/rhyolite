use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use ash::{prelude::VkResult, vk};
use bevy::asset::{AssetId, Assets};

use crate::shader::{ShaderModule, SpecializedShader};
use crate::{
    deferred::{DeferredOperationTaskPool, Task},
    Device,
};

use super::PipelineInner;

#[derive(Clone)]
pub struct GraphicsPipeline(Arc<PipelineInner>);
impl GraphicsPipeline {
    pub fn raw(&self) -> vk::Pipeline {
        self.0.pipeline
    }
}
impl super::Pipeline for GraphicsPipeline {
    type BuildInfo = BoxedGraphicsPipelineBuildInfo;
    const TYPE: vk::PipelineBindPoint = vk::PipelineBindPoint::GRAPHICS;
    fn from_built(
        _info: &mut BoxedGraphicsPipelineBuildInfo,
        item: <Self::BuildInfo as super::PipelineBuildInfo>::Pipeline,
    ) -> Self {
        GraphicsPipeline(Arc::new(item))
    }

    fn as_raw(&self) -> vk::Pipeline {
        self.0.pipeline
    }
}

pub struct Builder<'a> {
    device: &'a Device,
    cache: vk::PipelineCache,
    info: vk::GraphicsPipelineCreateInfo<'a>,
}
impl<'a> Deref for Builder<'a> {
    type Target = vk::GraphicsPipelineCreateInfo<'a>;
    fn deref(&self) -> &Self::Target {
        &self.info
    }
}
impl<'a> DerefMut for Builder<'a> {
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
    type Pipeline = PipelineInner;

    fn build(
        &mut self,
        pool: &DeferredOperationTaskPool,
        assets: &Assets<ShaderModule>,
        cache: vk::PipelineCache,
    ) -> Option<Task<Self::Pipeline>> {
        let device = self.device.clone();
        let builder = self.builder.clone();
        let modules = self.stages.iter().map(|shader| assets.get(&shader.shader).map(|s| s.raw())).collect::<Option<Vec<_>>>()?;
        let stages = self.stages.clone();
        Some(pool.schedule(move || {
            let raw_specialization_info = stages
                .iter()
                .map(SpecializedShader::raw_specialization_info)
                .collect::<Vec<_>>();
            let stages = stages.iter().zip(raw_specialization_info.iter()).zip(modules.into_iter()).map(|((shader, specialization_info), module)| {
                vk::PipelineShaderStageCreateInfo::default().flags(shader.flags).stage(shader.stage).module(module).name(&shader.entry_point).specialization_info(specialization_info)
            }).collect::<Vec<_>>();
            let mut info = vk::GraphicsPipelineCreateInfo::default().stages(&stages);
            let result = builder(Builder {
                device: &device,
                cache,
                info,
            })
            .0?;
            drop(stages);
            drop(raw_specialization_info);
            Ok(PipelineInner {
                device,
                pipeline: result,
            })
        }))
    }
    fn all_shaders(&self) -> impl Iterator<Item = AssetId<ShaderModule>> {
        self.stages.iter().map(|shader| shader.shader.id())
    }
}
