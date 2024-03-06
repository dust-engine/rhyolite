use ash::vk;
use bevy::asset::{AssetId, Assets};

use crate::{
    deferred::{DeferredOperationTaskPool, Task},
    shader::ShaderModule,
};

mod cache;
mod graphics;
mod layout;

pub use cache::*;
pub use graphics::*;
pub use layout::*;

trait Pipeline {
    type BuildInfo: PipelineBuildInfo<Pipeline = Self>;
}

trait PipelineBuildInfo {
    type Pipeline: Pipeline<BuildInfo = Self>;
    fn build_owned(
        mut self,
        pool: &DeferredOperationTaskPool,
        assets: &Assets<ShaderModule>,
        cache: vk::PipelineCache,
    ) -> Option<Task<Self::Pipeline>>
    where
        Self: Sized,
    {
        self.build(pool, assets, cache)
    }
    fn build(
        &mut self,
        pool: &DeferredOperationTaskPool,
        assets: &Assets<ShaderModule>,
        cache: vk::PipelineCache,
    ) -> Option<Task<Self::Pipeline>>;
    fn all_shaders(&self) -> impl Iterator<Item = AssetId<ShaderModule>>;
}
