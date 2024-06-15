use std::{collections::HashMap, sync::Arc};

use ash::vk;
use bevy::app::Plugin;
use bevy::asset::{AssetEvent, AssetId, Assets};
use bevy::ecs::world::FromWorld;
use bevy::ecs::{
    prelude::EventReader,
    system::{ResMut, Resource},
};

use crate::deferred::{DeferredOperationTaskPool, Task};
use crate::dispose::RenderObject;
use crate::shader::ShaderModule;
use crate::Device;

use super::compute::{ComputePipeline, ComputePipelineCreateInfo};
use super::{
    BoxedGraphicsPipelineBuildInfo, Builder, BuilderResult, GraphicsPipeline,
    GraphicsPipelineBuildInfo, Pipeline, PipelineBuildInfo,
};

#[derive(Resource)]
pub struct PipelineCache {
    device: Device,
    cache: vk::PipelineCache,
    shader_generations: HashMap<AssetId<ShaderModule>, u32>,
    hot_reload_enabled: bool,
}
impl Drop for PipelineCache {
    fn drop(&mut self) {
        if self.cache != vk::PipelineCache::null() {
            unsafe {
                self.device.destroy_pipeline_cache(self.cache, None);
            }
        }
    }
}

impl FromWorld for PipelineCache {
    fn from_world(world: &mut bevy::ecs::world::World) -> Self {
        Self {
            device: world.resource::<Device>().clone(),
            cache: vk::PipelineCache::null(), // TODO
            shader_generations: Default::default(),
            hot_reload_enabled: true,
        }
    }
}

pub struct CachedPipeline<T: Pipeline> {
    build_info: Option<T::BuildInfo>,
    pipeline: Option<T>,
    task: Option<Task<<T::BuildInfo as PipelineBuildInfo>::Pipeline>>,
    shader_generations: HashMap<AssetId<ShaderModule>, u32>,
}
impl<T: Pipeline> CachedPipeline<T> {
    pub fn is_ready(&self) -> bool {
        if self.pipeline.is_some() {
            return true;
        }
        if let Some(task) = &self.task {
            if task.is_finished() {
                return true;
            }
        }
        return false;
    }
    pub fn get(&self) -> Option<&T> {
        self.pipeline.as_ref()
    }
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.pipeline.as_mut()
    }
}

impl PipelineCache {
    pub fn create<T: Pipeline>(&self, build_info: T::BuildInfo) -> CachedPipeline<T> {
        CachedPipeline {
            pipeline: None,
            task: None,
            shader_generations: if self.hot_reload_enabled {
                let mut map = HashMap::new();
                map.extend(build_info.all_shaders().map(|shader| (shader, 0)));
                map
            } else {
                Default::default()
            },
            build_info: Some(build_info),
        }
    }
    pub fn create_graphics<F>(
        &self,
        build_info: GraphicsPipelineBuildInfo<F>,
    ) -> CachedPipeline<RenderObject<GraphicsPipeline>>
    where
        F: for<'a> Fn(Builder) -> BuilderResult + Send + Sync + 'static,
    {
        let boxed = BoxedGraphicsPipelineBuildInfo {
            device: build_info.device,
            stages: build_info.stages,
            builder: Arc::new(build_info.builder),
        };
        self.create::<RenderObject<GraphicsPipeline>>(boxed)
    }
    pub fn create_compute(
        &self,
        build_info: ComputePipelineCreateInfo,
    ) -> CachedPipeline<RenderObject<ComputePipeline>> {
        self.create::<RenderObject<ComputePipeline>>(build_info)
    }
    pub fn is_outdated<T: Pipeline>(&self, cached_pipeline: &CachedPipeline<T>) -> bool {
        for (shader, generation) in cached_pipeline.shader_generations.iter() {
            if let Some(latest_generation) = self.shader_generations.get(shader) {
                if latest_generation > generation {
                    return true;
                }
            }
        }
        return false;
    }
    pub fn retrieve<'a, T: Pipeline>(
        &self,
        cached_pipeline: &'a mut CachedPipeline<T>,
        assets: &Assets<ShaderModule>,
        pool: &DeferredOperationTaskPool,
    ) -> Option<&'a mut T> {
        self.retrieve_pipeline(cached_pipeline, assets, pool, true)
    }
    pub fn retrieve_pipeline<'a, T: Pipeline>(
        &self,
        cached_pipeline: &'a mut CachedPipeline<T>,
        assets: &Assets<ShaderModule>,
        pool: &DeferredOperationTaskPool,
        allow_stale: bool,
    ) -> Option<&'a mut T> {
        if let Some(pipeline) = &mut cached_pipeline.task {
            if pipeline.is_finished() {
                let new_pipeline = cached_pipeline.task.take().unwrap().unwrap().unwrap();
                let built = if self.hot_reload_enabled {
                    let build_info = cached_pipeline.build_info.as_mut().unwrap();
                    T::from_built(build_info, new_pipeline)
                } else {
                    let build_info = cached_pipeline.build_info.take().unwrap();
                    T::from_built_with_owned_info(build_info, new_pipeline)
                };
                cached_pipeline.pipeline.replace(built);
            } else if !allow_stale {
                // A build task is pending, and we don't want to return a stale pipeline.
                return None;
            }
        }

        if self.hot_reload_enabled {
            if self.is_outdated(cached_pipeline) {
                // schedule.
                cached_pipeline.task = cached_pipeline
                    .build_info
                    .as_mut()
                    .unwrap()
                    .build(pool, assets, self.cache);
                if cached_pipeline.task.is_some() {
                    // if cached.pipeline.task is still none, it means that some of the shaders hasn't finished loading.
                    // in that case we skip updating the generations and we'll retry on the next frame.
                    for (shader, generation) in cached_pipeline.shader_generations.iter_mut() {
                        if let Some(latest_generation) = self.shader_generations.get(shader) {
                            *generation = *latest_generation;
                        }
                    }
                    tracing::info!("Shader hot reload: updated");
                }
            }
        }

        if let Some(pipeline) = cached_pipeline.pipeline.as_mut() {
            if cached_pipeline.task.is_none() {
                return Some(pipeline);
            } else if allow_stale {
                return Some(pipeline);
            } else {
                return None;
            }
        } else {
            if cached_pipeline.task.is_none() {
                // schedule
                cached_pipeline.task = cached_pipeline
                    .build_info
                    .as_mut()
                    .unwrap()
                    .build(pool, assets, self.cache);
            }
            return None;
        }
    }
}

fn pipeline_cache_shader_updated_system(
    mut pipeline_cache: ResMut<PipelineCache>,
    mut events: EventReader<AssetEvent<ShaderModule>>,
) {
    for event in events.read() {
        match event {
            AssetEvent::Added { id: _ } => (),
            AssetEvent::Modified { id } => {
                let generation = pipeline_cache.shader_generations.entry(*id).or_default();
                *generation += 1;
            }
            AssetEvent::Removed { id } => {
                pipeline_cache.shader_generations.remove(id);
            }
            _ => (),
        }
    }
}

pub struct PipelineCachePlugin {
    shader_hot_reload: bool,
    pipeline_cache_enabled: bool, // TODO: use pipeline cache
}

impl Default for PipelineCachePlugin {
    fn default() -> Self {
        Self {
            shader_hot_reload: true,
            pipeline_cache_enabled: false,
        }
    }
}

impl Plugin for PipelineCachePlugin {
    fn build(&self, app: &mut bevy::prelude::App) {}
    fn finish(&self, app: &mut bevy::app::App) {
        let cache = PipelineCache {
            device: app.world().resource::<Device>().clone(),
            cache: vk::PipelineCache::null(), // TODO
            shader_generations: Default::default(),
            hot_reload_enabled: self.shader_hot_reload,
        };
        app.insert_resource(cache);
        if self.shader_hot_reload {
            app.add_systems(bevy::app::Update, pipeline_cache_shader_updated_system);
        }
    }
}
