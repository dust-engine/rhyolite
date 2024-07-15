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

use super::{PipelineInner, PipelineLayout};

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
    stages: &'a [vk::PipelineShaderStageCreateInfo<'a>],
    layout: &'a PipelineLayout,
}

pub struct BuilderResult(VkResult<vk::Pipeline>);
impl<'a> Builder<'a> {
    /// All fields except `stages` and `layout` of the [`vk::GraphicsPipelineCreateInfo`] are required to be configured
    /// as specified in [the Vulkan specification](https://vkdoc.net/man/VkGraphicsPipelineCreateInfo).
    /// `stages` and `layout` will be automatically filled with [`Self::stages`] and [`Self::layout`] respectively.
    #[must_use]
    #[inline]
    pub fn build(self, info: vk::GraphicsPipelineCreateInfo<'a>) -> BuilderResult {
        let mut pipeline = vk::Pipeline::null();
        let info = info.stages(&self.stages).layout(self.layout.raw());
        let result = unsafe {
            (self.device.fp_v1_0().create_graphics_pipelines)(
                self.device.handle(),
                self.cache,
                1,
                &info,
                std::ptr::null(),
                &mut pipeline,
            )
        };
        BuilderResult(result.result_with_success(pipeline))
    }
}

pub struct GraphicsPipelineBuildInfo<F>
where
    F: for<'a> Fn(Builder<'a>) -> BuilderResult + Send + Sync,
{
    pub device: Device,
    pub stages: Vec<SpecializedShader>,
    /// The builder function that will be called asynchronously to create the pipeline.
    /// 
    /// This function is expected to call [`Builder::build`] with a [`vk::GraphicsPipelineCreateInfo`].
    /// All fields except `stages` and `layout` of the [`vk::GraphicsPipelineCreateInfo`] are required to be configured
    /// as specified in [the Vulkan specification](https://vkdoc.net/man/VkGraphicsPipelineCreateInfo).
    /// `stages` and `layout` will be automatically filled with [`Self::stages`] and [`Self::layout`] respectively.
    pub builder: F,
    pub layout: Arc<PipelineLayout>,
}

pub struct BoxedGraphicsPipelineBuildInfo {
    pub device: Device,
    pub stages: Vec<SpecializedShader>,
    pub builder: Arc<dyn for<'a> Fn(Builder<'a>) -> BuilderResult + Send + Sync>,
    pub layout: Arc<PipelineLayout>,
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
        let modules = self
            .stages
            .iter()
            .map(|shader| assets.get(&shader.shader).map(|s| s.raw()))
            .collect::<Option<Vec<_>>>()?;
        let stages = self.stages.clone();
        let layout = self.layout.clone();
        Some(pool.schedule(move || {
            let raw_specialization_info = stages
                .iter()
                .map(SpecializedShader::raw_specialization_info)
                .collect::<Vec<_>>();
            let stages = stages
                .iter()
                .zip(raw_specialization_info.iter())
                .zip(modules.into_iter())
                .map(|((shader, specialization_info), module)| {
                    vk::PipelineShaderStageCreateInfo::default()
                        .flags(shader.flags)
                        .stage(shader.stage)
                        .module(module)
                        .name(&shader.entry_point)
                        .specialization_info(specialization_info)
                })
                .collect::<Vec<_>>();
            let result = builder(Builder {
                device: &device,
                cache,
                stages: &stages,
                layout: &layout,
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
