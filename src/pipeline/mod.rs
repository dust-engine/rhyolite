use ash::vk;
use bevy::asset::{AssetId, Assets};

use crate::{
    deferred::{DeferredOperationTaskPool, Task},
    dispose::RenderObject,
    shader::ShaderModule,
};

mod cache;
mod graphics;
mod layout;
mod ray_tracing;

pub use cache::*;
pub use graphics::*;
pub use layout::*;

pub mod sbt;

pub trait Pipeline: Sized + Send + Sync + 'static {
    type BuildInfo: PipelineBuildInfo;
    fn from_built(
        info: &mut Self::BuildInfo,
        item: <Self::BuildInfo as PipelineBuildInfo>::Pipeline,
    ) -> Self;

    /// Same as `from_built`, but takes ownership of the build info.
    /// This function is called instead when shader hot reloading is disabled.
    /// Because the pipeline will never be built again, the implementation may take ownership of the build info.
    fn from_built_with_owned_info(
        mut info: Self::BuildInfo,
        item: <Self::BuildInfo as PipelineBuildInfo>::Pipeline,
    ) -> Self {
        Self::from_built(&mut info, item)
    }
}

impl<T: Pipeline> Pipeline for RenderObject<T> {
    type BuildInfo = T::BuildInfo;
    fn from_built(
        info: &mut Self::BuildInfo,
        item: <Self::BuildInfo as PipelineBuildInfo>::Pipeline,
    ) -> Self {
        RenderObject::new(T::from_built(info, item))
    }
}

pub trait PipelineBuildInfo {
    type Pipeline: Pipeline<BuildInfo = Self>;
    fn build(
        &mut self,
        pool: &DeferredOperationTaskPool,
        assets: &Assets<ShaderModule>,
        cache: vk::PipelineCache,
    ) -> Option<Task<Self::Pipeline>>;

    /// List of all shaders used by this pipeline.
    /// Only called when shader hot reloading is enabled.
    fn all_shaders(&self) -> impl Iterator<Item = AssetId<ShaderModule>>;
}
