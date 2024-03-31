use std::{
    collections::BTreeMap,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use ash::{prelude::VkResult, vk};
use bevy::asset::{AssetId, Assets};

use crate::{
    deferred::{DeferredOperationTaskPool, Task}, dispose::RenderObject, utils::SendBox, Device
};
use crate::{
    shader::{ShaderModule, SpecializedShader},
    HasDevice,
};

use super::{CachedPipeline, PipelineCache, PipelineLayout};

pub struct RayTracingPipeline {
    device: Device,
    pipeline: vk::Pipeline,
}
impl RayTracingPipeline {
    pub fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }
}
impl Drop for RayTracingPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

impl super::Pipeline for RayTracingPipeline {
    type BuildInfo = RayTracingPipelineBuildInfo;
    fn from_built(_info: &mut RayTracingPipelineBuildInfo, item: <Self::BuildInfo as super::PipelineBuildInfo>::Pipeline) -> Self {
        item
    }
}

#[derive(Clone)]
pub struct RayTracingPipelineBuildInfoCommon {
    pub layout: Arc<PipelineLayout>,
    pub flags: vk::PipelineCreateFlags,

    pub max_pipeline_ray_recursion_depth: u32,
    pub max_pipeline_ray_payload_size: u32,
    pub max_pipeline_ray_hit_attribute_size: u32,
    pub dynamic_states: Vec<vk::DynamicState>,
}
pub struct RayTracingPipelineBuildInfo {
    pub common: RayTracingPipelineBuildInfoCommon,
    pub stages: Vec<SpecializedShader>,
    pub groups: Vec<vk::RayTracingShaderGroupCreateInfoKHR>,
    pub libraries: Vec<RayTracingPipelineLibrary>,
}

impl super::PipelineBuildInfo for RayTracingPipelineBuildInfo {
    type Pipeline = RayTracingPipeline;

    fn build(
        &mut self,
        pool: &DeferredOperationTaskPool,
        assets: &Assets<ShaderModule>,
        cache: vk::PipelineCache,
    ) -> Option<Task<Self::Pipeline>> {
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
        let groups = SendBox(self.groups.clone());
        let common = self.common.clone();
        let libraries = self.libraries.clone();

        Some(pool.schedule_dho(move |dho| {
            let stages = stages.into_inner();
            let mut info = vk::RayTracingPipelineCreateInfoKHR::default();
            info.stage_count = stages.len() as u32;
            info.p_stages = stages.as_ptr();

            info.group_count = groups.len() as u32;
            info.p_groups = groups.as_ptr();
            info.max_pipeline_ray_recursion_depth = common.max_pipeline_ray_recursion_depth;
            let library_interface = vk::RayTracingPipelineInterfaceCreateInfoKHR {
                max_pipeline_ray_payload_size: common.max_pipeline_ray_payload_size,
                max_pipeline_ray_hit_attribute_size: common.max_pipeline_ray_hit_attribute_size,
                ..Default::default()
            };
            info.p_library_interface = &library_interface;

            let dynamic_state = vk::PipelineDynamicStateCreateInfo {
                dynamic_state_count: common.dynamic_states.len() as u32,
                p_dynamic_states: common.dynamic_states.as_ptr(),
                ..Default::default()
            };
            info.p_dynamic_state = &dynamic_state;
            info.layout = common.layout.raw();

            let raw_libraries = libraries.iter().map(|library| library.pipeline.pipeline).collect::<Vec<_>>();
            let library_info = vk::PipelineLibraryCreateInfoKHR {
                library_count: raw_libraries.len() as u32,
                p_libraries: raw_libraries.as_ptr(),
                ..Default::default()
            };
            info.p_library_info = &library_info;

            let (result, pipeline) = unsafe {
                let mut pipeline = vk::Pipeline::null();
                let result = (common
                    .layout
                    .device()
                    .extension::<ash::extensions::khr::RayTracingPipeline>()
                    .fp()
                    .create_ray_tracing_pipelines_khr)(
                    common.layout.device().handle(),
                    dho,
                    cache,
                    1,
                    &info,
                    std::ptr::null(),
                    &mut pipeline,
                );
                (result, pipeline)
            };
            let device = common.layout.device().clone();

            drop(stages);
            drop(common);
            drop(specialization_info);
            drop(groups);
            drop(libraries);
            drop(raw_libraries);
            (result, RayTracingPipeline { device, pipeline })
        }))
    }
    fn all_shaders(&self) -> impl Iterator<Item = AssetId<ShaderModule>> {
        self.stages.iter().map(|shader| shader.shader.id()).chain(
            self.libraries
                .iter()
                .flat_map(|library| library.shaders.iter().cloned()),
        )
    }
}

#[derive(Clone)]
pub struct RayTracingPipelineLibrary {
    pipeline: Arc<RayTracingPipeline>,
    /// Should always be empty if shader hot reloading is disabled
    shaders: Vec<AssetId<ShaderModule>>
}
impl super::Pipeline for RayTracingPipelineLibrary {
    type BuildInfo = RayTracingPipelineBuildInfo;
    fn from_built(info: &mut RayTracingPipelineBuildInfo, item: <Self::BuildInfo as super::PipelineBuildInfo>::Pipeline) -> Self {
        RayTracingPipelineLibrary {
            pipeline: Arc::new(item),
            shaders: info.stages.iter().map(|shader| shader.shader.id()).collect(),
        }
    }
    fn from_built_with_owned_info(info: Self::BuildInfo, item: <Self::BuildInfo as super::PipelineBuildInfo>::Pipeline) -> Self {
        RayTracingPipelineLibrary {
            pipeline: Arc::new(item),
            shaders: Vec::new(),
        }
    }
}


#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct HitgroupHandle(u64);

/// Utility for managing ray tracing pipeline lifetimes.
/// It assumes that the ray gen shaders, miss shaders, and callable shaders are immutable,
/// while the hitgroup shaders may change frequently.
/// As such, we create a pipeline library for the ray gen, miss, and callable shaders,
/// and a pipeline library for each of the hitgroups.
pub enum RayTracingPipelineManager {
    Native(RayTracingPipelineManagerNative),
    PipelineLibrary(RayTracingPipelineManagerPipelineLibrary),
}

impl RayTracingPipelineManager {
    pub fn new(
        common: RayTracingPipelineBuildInfoCommon,
        raygen_shaders: Vec<SpecializedShader>,
        miss_shaders: Vec<SpecializedShader>,
        callable_shaders: Vec<SpecializedShader>,
        pipeline_cache: &PipelineCache,
    ) -> Self {
        if true {
            Self::PipelineLibrary(RayTracingPipelineManagerPipelineLibrary::new(
                common,
                raygen_shaders,
                miss_shaders,
                callable_shaders,
                pipeline_cache,
            ))
        } else {
            Self::Native(RayTracingPipelineManagerNative::new(
                common,
                raygen_shaders,
                miss_shaders,
                callable_shaders,
            ))
        }
    }

    pub fn add_hitgroup(
        &mut self,
        ty: vk::RayTracingShaderGroupTypeKHR,
        closest_hit_shader: Option<SpecializedShader>,
        anyhit_shader: Option<SpecializedShader>,
        intersection_shader: Option<SpecializedShader>,
        pipeline_cache: &PipelineCache,
    ) -> HitgroupHandle {
        match self {
            Self::Native(manager) => manager.add_hitgroup(ty, closest_hit_shader, anyhit_shader, intersection_shader),
            Self::PipelineLibrary(manager) => manager.add_hitgroup(ty, closest_hit_shader, anyhit_shader, intersection_shader, &pipeline_cache),
        }
    }

    pub fn remove_hitgroup(&mut self, handle: HitgroupHandle) {
        match self {
            Self::Native(manager) => manager.remove_hitgroup(handle),
            Self::PipelineLibrary(manager) => manager.remove_hitgroup(handle),
        }
    }

    pub fn build(
        &mut self,
        pipeline_cache: &PipelineCache,
        assets: &Assets<ShaderModule>,
        pool: &DeferredOperationTaskPool,
    ) -> Option<CachedPipeline<RenderObject<RayTracingPipeline>>> {
        match self {
            Self::Native(manager) => manager.build(pipeline_cache),
            Self::PipelineLibrary(manager) => manager.build(pipeline_cache, assets, pool),
        }
    }

}


struct RayTracingPipelineManagerNative {
    common: RayTracingPipelineBuildInfoCommon,
    hitgroup_table: BTreeMap<HitgroupHandle, u32>,
    /// closest hit, any hit, intersection
    hitgroup_shaders: Vec<(
        HitgroupHandle,
        vk::RayTracingShaderGroupTypeKHR,
        Option<SpecializedShader>,
        Option<SpecializedShader>,
        Option<SpecializedShader>,
    )>,
    base_stages: Vec<SpecializedShader>,
}
impl RayTracingPipelineManagerNative {
    fn new(
        common: RayTracingPipelineBuildInfoCommon,
        raygen_shaders: Vec<SpecializedShader>,
        miss_shaders: Vec<SpecializedShader>,
        callable_shaders: Vec<SpecializedShader>,
    ) -> Self {
        if raygen_shaders.len() == 0 {
            panic!("At least one raygen shader must be provided");
        }
        Self {
            common,
            base_stages: raygen_shaders
                .into_iter()
                .chain(miss_shaders)
                .chain(callable_shaders)
                .collect(),
            hitgroup_shaders: Vec::new(),
            hitgroup_table: BTreeMap::new(),
        }
    }
    fn add_hitgroup(
        &mut self,
        ty: vk::RayTracingShaderGroupTypeKHR,
        closest_hit_shader: Option<SpecializedShader>,
        anyhit_shader: Option<SpecializedShader>,
        intersection_shader: Option<SpecializedShader>,
    ) -> HitgroupHandle {
        let handle = loop {
            let handle = HitgroupHandle(fastrand::u64(..));
            if self.hitgroup_table.contains_key(&handle) {
                continue;
            }
            self.hitgroup_table
                .insert(handle, self.hitgroup_shaders.len() as u32);
            break handle;
        };
        self.hitgroup_shaders.push((
            handle,
            ty,
            closest_hit_shader,
            anyhit_shader,
            intersection_shader,
        ));
        handle
    }
    #[track_caller]
    pub fn remove_hitgroup(&mut self, handle: HitgroupHandle) {
        let index = self
            .hitgroup_table
            .remove(&handle)
            .expect("Hitgroup handle does not exist");

        if index as usize != self.hitgroup_shaders.len() - 1 {
            // not last
            let last = self.hitgroup_shaders.len() - 1;
            let last_handle = self.hitgroup_shaders[last].0;
            *self.hitgroup_table.get_mut(&last_handle).unwrap() = index;
        }

        self.hitgroup_shaders.swap_remove(index as usize);
    }

    fn build(
        &mut self,
        pipeline_cache: &PipelineCache,
    ) -> Option<CachedPipeline<RenderObject<RayTracingPipeline>>> {
        let mut stages = Vec::new();
        let mut groups = Vec::new();
        build_general_shader(&mut stages, &mut groups, &self.base_stages);
        build_hitgroup_shaders(&mut stages, &mut groups, &self.hitgroup_shaders);
        let info = RayTracingPipelineBuildInfo {
            common: self.common.clone(),
            stages,
            groups,
            libraries: Vec::new(),
        };
        Some(pipeline_cache.create(info))
    }
}

struct RayTracingPipelineManagerPipelineLibrary {
    common: RayTracingPipelineBuildInfoCommon,
    base_library: CachedPipeline<RayTracingPipelineLibrary>,
    hitgroup_table: BTreeMap<HitgroupHandle, u32>,
    hitgroup_libraries: Vec<(
        HitgroupHandle,
        vk::RayTracingShaderGroupTypeKHR,
        CachedPipeline<RayTracingPipelineLibrary>,
    )>,
}

impl RayTracingPipelineManagerPipelineLibrary {
    pub fn new(
        common: RayTracingPipelineBuildInfoCommon,
        raygen_shaders: Vec<SpecializedShader>,
        miss_shaders: Vec<SpecializedShader>,
        callable_shaders: Vec<SpecializedShader>,
        pipeline_cache: &PipelineCache,
    ) -> Self {
        if raygen_shaders.len() == 0 {
            panic!("At least one raygen shader must be provided");
        }
        let stages: Vec<_> = raygen_shaders
            .into_iter()
            .chain(miss_shaders)
            .chain(callable_shaders)
            .collect();
        let groups: Vec<_> = stages
            .iter()
            .enumerate()
            .map(|(i, shader)| vk::RayTracingShaderGroupCreateInfoKHR {
                ty: vk::RayTracingShaderGroupTypeKHR::GENERAL,
                general_shader: i as u32,
                closest_hit_shader: vk::SHADER_UNUSED_KHR,
                any_hit_shader: vk::SHADER_UNUSED_KHR,
                intersection_shader: vk::SHADER_UNUSED_KHR,
                ..Default::default()
            })
            .collect();
        let info = RayTracingPipelineBuildInfo {
            common: common.clone(),
            stages,
            groups,
            libraries: Vec::new(),
        };
        let base_library: CachedPipeline<RayTracingPipelineLibrary> = pipeline_cache.create(info);
        Self {
            common,
            base_library,
            hitgroup_libraries: Vec::new(),
            hitgroup_table: BTreeMap::new(),
        }
    }

    pub fn add_hitgroup(
        &mut self,
        ty: vk::RayTracingShaderGroupTypeKHR,
        closest_hit_shader: Option<SpecializedShader>,
        anyhit_shader: Option<SpecializedShader>,
        intersection_shader: Option<SpecializedShader>,
        pipeline_cache: &PipelineCache,
    ) -> HitgroupHandle {
        let handle = loop {
            let handle = HitgroupHandle(fastrand::u64(..));
            if self.hitgroup_table.contains_key(&handle) {
                continue;
            }
            self.hitgroup_table
                .insert(handle, self.hitgroup_libraries.len() as u32);
            break handle;
        };
        let mut stages = Vec::new();
        let mut groups = Vec::new();
        build_hitgroup_shaders(
            &mut stages,
            &mut groups,
            &[(
                handle,
                ty,
                closest_hit_shader,
                anyhit_shader,
                intersection_shader,
            )],
        );
        let info = RayTracingPipelineBuildInfo {
            common: self.common.clone(),
            stages,
            groups,
            libraries: Vec::new(),
        };
        let library: CachedPipeline<RayTracingPipelineLibrary> = pipeline_cache.create(info);
        self.hitgroup_libraries.push((handle, ty, library));
        handle
    }

    #[track_caller]
    pub fn remove_hitgroup(&mut self, handle: HitgroupHandle) {
        let index = self
            .hitgroup_table
            .remove(&handle)
            .expect("Hitgroup handle does not exist");

        if index as usize != self.hitgroup_libraries.len() - 1 {
            // not last
            let last = self.hitgroup_libraries.len() - 1;
            let last_handle = self.hitgroup_libraries[last].0;
            *self.hitgroup_table.get_mut(&last_handle).unwrap() = index;
        }

        self.hitgroup_libraries.swap_remove(index as usize);
    }

    pub fn build(
        &mut self,
        pipeline_cache: &PipelineCache,
        assets: &Assets<ShaderModule>,
        pool: &DeferredOperationTaskPool,
    ) -> Option<CachedPipeline<RenderObject<RayTracingPipeline>>> {
        let libraries = std::iter::once(&mut self.base_library)
        .chain(self.hitgroup_libraries.iter_mut().map(|(_, _, library)| library))
        .map(|library| pipeline_cache.retrieve_inner(library, assets, pool, false).cloned())
        .collect::<Option<Vec<_>>>()?;
        let info = RayTracingPipelineBuildInfo {
            common: self.common.clone(),
            stages: Vec::new(),
            groups: Vec::new(),
            libraries,
        };
        Some(pipeline_cache.create(info))
    }
}

fn build_general_shader(
    stages: &mut Vec<SpecializedShader>,
    groups: &mut Vec<vk::RayTracingShaderGroupCreateInfoKHR>,
    shaders: &[SpecializedShader],
) {
    for (i, shader) in shaders.iter().enumerate() {
        stages.push(shader.clone());
        groups.push(vk::RayTracingShaderGroupCreateInfoKHR {
            ty: vk::RayTracingShaderGroupTypeKHR::GENERAL,
            general_shader: i as u32,
            closest_hit_shader: vk::SHADER_UNUSED_KHR,
            any_hit_shader: vk::SHADER_UNUSED_KHR,
            intersection_shader: vk::SHADER_UNUSED_KHR,
            ..Default::default()
        });
    }
}

fn build_hitgroup_shaders(
    stages: &mut Vec<SpecializedShader>,
    groups: &mut Vec<vk::RayTracingShaderGroupCreateInfoKHR>,
    hitgroups: &[(
        HitgroupHandle,
        vk::RayTracingShaderGroupTypeKHR,
        Option<SpecializedShader>, // rchit
        Option<SpecializedShader>, // rahit
        Option<SpecializedShader>, // rint
    )],
) {
    for (_, ty, rchit, rahit, rint) in hitgroups {
        let mut rchit_stage: u32 = vk::SHADER_UNUSED_KHR;
        let mut rint_stage: u32 = vk::SHADER_UNUSED_KHR;
        let mut rahit_stage: u32 = vk::SHADER_UNUSED_KHR;
        if let Some(shader) = rchit.as_ref() {
            assert_eq!(shader.stage, vk::ShaderStageFlags::CLOSEST_HIT_KHR);
            rchit_stage = stages.len() as u32;
            stages.push(shader.clone());
        }

        if let Some(shader) = rint.as_ref() {
            assert_eq!(shader.stage, vk::ShaderStageFlags::INTERSECTION_KHR);
            rint_stage = stages.len() as u32;
            stages.push(shader.clone());
        }
        if let Some(shader) = rahit.as_ref() {
            assert_eq!(shader.stage, vk::ShaderStageFlags::ANY_HIT_KHR);
            rahit_stage = stages.len() as u32;
            stages.push(shader.clone());
        }
        groups.push(vk::RayTracingShaderGroupCreateInfoKHR {
            ty: *ty,
            closest_hit_shader: rchit_stage,
            any_hit_shader: rahit_stage,
            intersection_shader: rint_stage,
            general_shader: vk::SHADER_UNUSED_KHR,
            ..Default::default()
        })
    }
}
