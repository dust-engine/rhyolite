use std::{alloc::Layout, collections::BTreeSet, sync::Arc};

use ash::{prelude::VkResult, vk};
use bevy::{
    asset::{AssetId, Assets},
    utils::{
        nonmax::{self, NonMaxU16, NonMaxU32},
        smallvec::SmallVec,
    },
};

use crate::{
    deferred::{DeferredOperationTaskPool, Task},
    dispose::RenderObject,
    utils::SendBox,
    Device,
};
use crate::{
    shader::{ShaderModule, SpecializedShader},
    HasDevice,
};

use super::{CachedPipeline, PipelineCache, PipelineInner, PipelineLayout};

#[derive(Clone)]
pub struct RayTracingPipeline {
    inner: Arc<PipelineInner>,
    handles: SbtHandles,
}
impl RayTracingPipeline {
    pub fn handles(&self) -> &SbtHandles {
        &self.handles
    }
}

impl RayTracingPipeline {
    pub fn raw(&self) -> vk::Pipeline {
        self.inner.pipeline
    }
}
impl super::Pipeline for RayTracingPipeline {
    type BuildInfo = RayTracingPipelineBuildInfo;
    const TYPE: vk::PipelineBindPoint = vk::PipelineBindPoint::RAY_TRACING_KHR;

    fn as_raw(&self) -> vk::Pipeline {
        self.inner.pipeline
    }
    fn from_built(
        info: &mut RayTracingPipelineBuildInfo,
        item: <Self::BuildInfo as super::PipelineBuildInfo>::Pipeline,
    ) -> Self {
        RayTracingPipeline {
            handles: SbtHandles::new(
                &item.device,
                item.pipeline,
                info.hitgroup_mapping.clone(),
                info.num_raygen,
                info.num_miss,
                info.num_callable,
                info.num_hitgroup,
            )
            .unwrap(),
            inner: Arc::new(item),
        }
    }

    fn from_built_with_owned_info(
        info: RayTracingPipelineBuildInfo,
        item: <Self::BuildInfo as super::PipelineBuildInfo>::Pipeline,
    ) -> Self {
        RayTracingPipeline {
            handles: SbtHandles::new(
                &item.device,
                item.pipeline,
                info.hitgroup_mapping,
                info.num_raygen,
                info.num_miss,
                info.num_callable,
                info.num_hitgroup,
            )
            .unwrap(),
            inner: Arc::new(item),
        }
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
unsafe impl Send for RayTracingPipelineBuildInfoCommon {}
unsafe impl Sync for RayTracingPipelineBuildInfoCommon {}
pub struct RayTracingPipelineBuildInfo {
    pub common: RayTracingPipelineBuildInfoCommon,
    stages: Vec<SpecializedShader>,
    groups: Vec<vk::RayTracingShaderGroupCreateInfoKHR>,
    libraries: Vec<RayTracingPipelineLibrary>,
    hitgroup_mapping: Vec<u32>,
    num_raygen: u8,
    num_miss: u8,
    num_callable: u8,
    num_hitgroup: u32,
}
unsafe impl Send for RayTracingPipelineBuildInfo {}
unsafe impl Sync for RayTracingPipelineBuildInfo {}

impl super::PipelineBuildInfo for RayTracingPipelineBuildInfo {
    type Pipeline = PipelineInner;

    fn build(
        &mut self,
        pool: &DeferredOperationTaskPool,
        assets: &Assets<ShaderModule>,
        cache: vk::PipelineCache,
    ) -> Option<Task<Self::Pipeline>> {
        let stages = SendBox(SpecializedShader::as_raw_many(&self.stages, assets)?);
        let groups = SendBox(self.groups.clone());
        let common = self.common.clone();
        let libraries = self.libraries.clone();

        Some(pool.schedule_dho(move |dho| {
            let (specialization_info, stages) = stages.into_inner();
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
            info.flags = common.flags;

            let raw_libraries = libraries
                .iter()
                .map(|library| library.pipeline.pipeline)
                .collect::<Vec<_>>();
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
            (result, PipelineInner { device, pipeline })
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
    pipeline: Arc<PipelineInner>,
    /// Should always be empty if shader hot reloading is disabled
    shaders: Vec<AssetId<ShaderModule>>,
}
impl super::Pipeline for RayTracingPipelineLibrary {
    type BuildInfo = RayTracingPipelineBuildInfo;
    const TYPE: vk::PipelineBindPoint = vk::PipelineBindPoint::RAY_TRACING_KHR;

    fn as_raw(&self) -> vk::Pipeline {
        self.pipeline.pipeline
    }
    fn from_built(
        info: &mut RayTracingPipelineBuildInfo,
        item: <Self::BuildInfo as super::PipelineBuildInfo>::Pipeline,
    ) -> Self {
        RayTracingPipelineLibrary {
            pipeline: Arc::new(item),
            shaders: info
                .stages
                .iter()
                .map(|shader| shader.shader.id())
                .collect(),
        }
    }
    fn from_built_with_owned_info(
        _info: Self::BuildInfo,
        item: <Self::BuildInfo as super::PipelineBuildInfo>::Pipeline,
    ) -> Self {
        RayTracingPipelineLibrary {
            pipeline: Arc::new(item),
            // if we're building with owned info, that means the library would never be rebuilt.
            // so, we don't need to keep a list of shaders to tell when we should rebuild.
            shaders: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
pub struct HitgroupHandle(NonMaxU16);

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
        hitgroup: HitGroup,
        pipeline_cache: &PipelineCache,
    ) -> HitgroupHandle {
        match self {
            Self::Native(manager) => manager.add_hitgroup(hitgroup),
            Self::PipelineLibrary(manager) => manager.add_hitgroup(hitgroup, &pipeline_cache),
        }
    }
    pub fn set_hitgroup(
        &mut self,
        handle: HitgroupHandle,
        hitgroup: HitGroup,
        pipeline_cache: &PipelineCache,
    ) {
        match self {
            Self::Native(manager) => manager.set_hitgroup(handle, hitgroup),
            Self::PipelineLibrary(manager) => {
                manager.set_hitgroup(handle, hitgroup, pipeline_cache)
            }
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
    num_raygen: u8,
    num_miss: u8,
    num_callable: u8,
    /// closest hit, any hit, intersection
    hitgroups: Vec<Option<HitGroup>>,
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
            num_raygen: raygen_shaders.len() as u8,
            num_miss: miss_shaders.len() as u8,
            num_callable: callable_shaders.len() as u8,
            base_stages: raygen_shaders
                .into_iter()
                .chain(miss_shaders)
                .chain(callable_shaders)
                .collect(),
            hitgroups: Vec::new(),
        }
    }
    fn add_hitgroup(&mut self, hitgroup: HitGroup) -> HitgroupHandle {
        let handle = self.hitgroups.len() as u16;
        let handle = HitgroupHandle(NonMaxU16::new(handle).unwrap());
        self.hitgroups.push(Some(hitgroup));
        handle
    }
    fn set_hitgroup(&mut self, handle: HitgroupHandle, hitgroup: HitGroup) {
        self.hitgroups[handle.0.get() as usize] = Some(hitgroup);
    }
    #[track_caller]
    fn remove_hitgroup(&mut self, handle: HitgroupHandle) {
        assert!(self.hitgroups[handle.0.get() as usize].is_some());
        self.hitgroups[handle.0.get() as usize] = None;
    }

    fn build(
        &mut self,
        pipeline_cache: &PipelineCache,
    ) -> Option<CachedPipeline<RenderObject<RayTracingPipeline>>> {
        let mut stages = Vec::new();
        let mut groups = Vec::new();
        for (i, shader) in self.base_stages.iter().enumerate() {
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
        for hitgroup in self.hitgroups.iter() {
            if let Some(hitgroup) = hitgroup {
                let base_shader_index = stages.len() as u32;
                stages.extend(hitgroup.shaders.iter().cloned());
                groups.extend(hitgroup.to_vk_groups(base_shader_index))
            }
        }

        let mut group_num: u32 = 0;
        let hitgroup_mapping = self
            .hitgroups
            .iter()
            .map(|hitgroup| {
                if hitgroup.is_some() {
                    let index = group_num;
                    group_num += 1;
                    index
                } else {
                    u32::MAX
                }
            })
            .collect();
        let info = RayTracingPipelineBuildInfo {
            common: self.common.clone(),
            stages,
            groups,
            libraries: Vec::new(),
            num_raygen: self.num_raygen,
            num_miss: self.num_miss,
            num_callable: self.num_callable,
            num_hitgroup: self.hitgroups.len() as u32,
            hitgroup_mapping,
        };
        Some(pipeline_cache.create(info))
    }
}

struct RayTracingPipelineManagerPipelineLibrary {
    common: RayTracingPipelineBuildInfoCommon,
    base_library: CachedPipeline<RayTracingPipelineLibrary>,
    hitgroup_libraries: Vec<Option<CachedPipeline<RayTracingPipelineLibrary>>>,
    num_raygen: u8,
    num_miss: u8,
    num_callable: u8,
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

        let num_raygen = raygen_shaders.len() as u8;
        let num_miss = miss_shaders.len() as u8;
        let num_callable = callable_shaders.len() as u8;

        let stages: Vec<_> = raygen_shaders
            .into_iter()
            .chain(miss_shaders)
            .chain(callable_shaders)
            .collect();
        let groups: Vec<_> = stages
            .iter()
            .enumerate()
            .map(|(i, _)| vk::RayTracingShaderGroupCreateInfoKHR {
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
            num_raygen,
            num_miss,
            num_callable,
            num_hitgroup: 0,
            hitgroup_mapping: Vec::new(),
        };
        let base_library: CachedPipeline<RayTracingPipelineLibrary> = pipeline_cache.create(info);
        Self {
            common,
            base_library,
            hitgroup_libraries: Vec::new(),
            num_raygen,
            num_miss,
            num_callable,
        }
    }

    pub fn add_hitgroup(
        &mut self,
        hitgroup: HitGroup,
        pipeline_cache: &PipelineCache,
    ) -> HitgroupHandle {
        let handle = self.hitgroup_libraries.len() as u16;
        let handle = HitgroupHandle(NonMaxU16::new(handle).unwrap());
        self.hitgroup_libraries.push(None);
        self.set_hitgroup(handle, hitgroup, pipeline_cache);
        handle
    }
    pub fn set_hitgroup(
        &mut self,
        handle: HitgroupHandle,
        hitgroup: HitGroup,
        pipeline_cache: &PipelineCache,
    ) {
        let info = RayTracingPipelineBuildInfo {
            common: self.common.clone(),
            groups: hitgroup.to_vk_groups(0),
            stages: hitgroup.shaders,
            libraries: Vec::new(),
            num_raygen: 0,
            num_miss: 0,
            num_callable: 0,
            num_hitgroup: 1,
            hitgroup_mapping: Vec::new(),
        };
        let library: CachedPipeline<RayTracingPipelineLibrary> = pipeline_cache.create(info);

        assert!(self.hitgroup_libraries[handle.0.get() as usize].is_none());
        self.hitgroup_libraries[handle.0.get() as usize] = Some(library);
    }

    #[track_caller]
    pub fn remove_hitgroup(&mut self, handle: HitgroupHandle) {
        assert!(self.hitgroup_libraries[handle.0.get() as usize].is_some());
        self.hitgroup_libraries[handle.0.get() as usize] = None;
    }

    pub fn build(
        &mut self,
        pipeline_cache: &PipelineCache,
        assets: &Assets<ShaderModule>,
        pool: &DeferredOperationTaskPool,
    ) -> Option<CachedPipeline<RenderObject<RayTracingPipeline>>> {
        let libraries = std::iter::once(&mut self.base_library)
            .chain(
                self.hitgroup_libraries
                    .iter_mut()
                    .filter_map(Option::as_mut),
            )
            .map(|library| {
                pipeline_cache
                    .retrieve_inner(library, assets, pool, false)
                    .cloned()
            })
            .collect::<Option<Vec<RayTracingPipelineLibrary>>>()?;

        let mut group_num = 0;
        let hitgroup_mapping = self
            .hitgroup_libraries
            .iter()
            .map(|hitgroup| {
                if hitgroup.is_some() {
                    let index = group_num;
                    group_num += 1;
                    index
                } else {
                    u32::MAX
                }
            })
            .collect();
        let info = RayTracingPipelineBuildInfo {
            num_raygen: self.num_raygen,
            num_miss: self.num_miss,
            num_callable: self.num_callable,
            num_hitgroup: self.hitgroup_libraries.len() as u32,
            common: self.common.clone(),
            stages: Vec::new(),
            groups: Vec::new(),
            libraries,
            hitgroup_mapping,
        };
        Some(pipeline_cache.create(info))
    }
}

#[derive(Clone)]
pub struct SbtHandles {
    data: Box<[u8]>,
    hitgroup_handle_map: Vec<u32>,
    handle_layout: Layout,
    group_base_alignment: u32,
    num_raygen: u8,
    num_miss: u8,
    num_callable: u8,
    num_hitgroup: u32,
}
impl SbtHandles {
    pub fn handle_layout(&self) -> &Layout {
        &self.handle_layout
    }
    fn new(
        device: &Device,
        pipeline: vk::Pipeline,
        hitgroup_handle_map: Vec<u32>,
        num_raygen: u8,
        num_miss: u8,
        num_callable: u8,
        num_hitgroup: u32,
    ) -> VkResult<SbtHandles> {
        let total_num_groups =
            num_hitgroup as u32 + num_miss as u32 + num_callable as u32 + num_raygen as u32;
        let rtx_properties = device
            .physical_device()
            .properties()
            .get::<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
        let sbt_handles_host_vec = unsafe {
            device
                .extension::<ash::extensions::khr::RayTracingPipeline>()
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    total_num_groups,
                    // VUID-vkGetRayTracingShaderGroupHandlesKHR-dataSize-02420
                    // dataSize must be at least VkPhysicalDeviceRayTracingPipelinePropertiesKHR::shaderGroupHandleSize × groupCount
                    rtx_properties.shader_group_handle_size as usize * total_num_groups as usize,
                )?
                .into_boxed_slice()
        };
        Ok(SbtHandles {
            hitgroup_handle_map,
            data: sbt_handles_host_vec,
            handle_layout: Layout::from_size_align(
                rtx_properties.shader_group_handle_size as usize,
                rtx_properties.shader_group_handle_alignment as usize,
            )
            .unwrap(),
            group_base_alignment: rtx_properties.shader_group_base_alignment,
            num_raygen: num_raygen,
            num_miss: num_miss,
            num_callable: num_callable,
            num_hitgroup: num_hitgroup,
        })
    }

    pub fn rgen(&self, index: usize) -> &[u8] {
        // Note that
        // VUID-vkGetRayTracingShaderGroupHandlesKHR-dataSize-02420
        // dataSize must be at least VkPhysicalDeviceRayTracingPipelinePropertiesKHR::shaderGroupHandleSize × groupCount
        // This implies all handles are tightly packed. No need to call `pad_to_align` here
        let start = self.handle_layout.size() * index;
        let end = start + self.handle_layout.size();
        &self.data[start..end]
    }
    pub fn rmiss(&self, index: usize) -> &[u8] {
        let start = self.handle_layout.size() * (index + self.num_raygen as usize);
        let end = start + self.handle_layout.size();
        &self.data[start..end]
    }
    pub fn callable(&self, index: usize) -> &[u8] {
        let start =
            self.handle_layout.size() * (index + self.num_raygen as usize + self.num_miss as usize);
        let end = start + self.handle_layout.size();
        &self.data[start..end]
    }
    pub fn hitgroup(&self, hitgroup_index: HitgroupHandle) -> &[u8] {
        let index = self.hitgroup_handle_map[hitgroup_index.0.get() as usize];
        assert_ne!(index, u32::MAX);
        let index = index as usize;
        let start = self.handle_layout.size()
            * (index
                + self.num_miss as usize
                + self.num_callable as usize
                + self.num_raygen as usize);
        let end = start + self.handle_layout.size();
        &self.data[start..end]
    }
}

#[derive(Clone)]
pub struct HitGroup {
    ty: vk::RayTracingShaderGroupTypeKHR,
    shaders: Vec<SpecializedShader>,
    groups: Vec<(
        Option<HitGroupShaderIndexClosestHit>,
        Option<HitGroupShaderIndexAnyHit>,
        Option<HitGroupShaderIndexIntersection>,
    )>,
}

#[derive(Clone, Copy)]
pub struct HitGroupShaderIndexClosestHit(nonmax::NonMaxU32);
#[derive(Clone, Copy)]
pub struct HitGroupShaderIndexAnyHit(nonmax::NonMaxU32);
#[derive(Clone, Copy)]
pub struct HitGroupShaderIndexIntersection(nonmax::NonMaxU32);
impl HitGroup {
    pub fn new(ty: vk::RayTracingShaderGroupTypeKHR) -> Self {
        Self {
            ty,
            shaders: Vec::new(),
            groups: Vec::new(),
        }
    }
    pub fn new_triangles() -> Self {
        Self {
            ty: vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP,
            shaders: Vec::new(),
            groups: Vec::new(),
        }
    }
    pub fn new_procedural() -> Self {
        Self {
            ty: vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP,
            shaders: Vec::new(),
            groups: Vec::new(),
        }
    }
    pub fn add_closest_hit_shader(
        &mut self,
        shader: SpecializedShader,
    ) -> HitGroupShaderIndexClosestHit {
        let index = self.shaders.len() as u32;
        self.shaders.push(shader);
        HitGroupShaderIndexClosestHit(NonMaxU32::new(index).unwrap())
    }
    pub fn add_any_hit_shader(&mut self, shader: SpecializedShader) -> HitGroupShaderIndexAnyHit {
        let index = self.shaders.len() as u32;
        self.shaders.push(shader);
        HitGroupShaderIndexAnyHit(NonMaxU32::new(index).unwrap())
    }
    pub fn add_intersection_shader(
        &mut self,
        shader: SpecializedShader,
    ) -> HitGroupShaderIndexIntersection {
        let index = self.shaders.len() as u32;
        self.shaders.push(shader);
        HitGroupShaderIndexIntersection(NonMaxU32::new(index).unwrap())
    }
    pub fn add_group(
        &mut self,
        closest_hit: Option<HitGroupShaderIndexClosestHit>,
        anyhit: Option<HitGroupShaderIndexAnyHit>,
        intersection: Option<HitGroupShaderIndexIntersection>,
    ) {
        self.groups.push((closest_hit, anyhit, intersection));
    }

    fn to_vk_groups(&self, base_shader_index: u32) -> Vec<vk::RayTracingShaderGroupCreateInfoKHR> {
        self.groups
            .iter()
            .map(
                |(rchit, rahit, rint)| vk::RayTracingShaderGroupCreateInfoKHR {
                    ty: self.ty,
                    general_shader: vk::SHADER_UNUSED_KHR,
                    closest_hit_shader: rchit
                        .as_ref()
                        .map(|a| a.0.get() + base_shader_index)
                        .unwrap_or(vk::SHADER_UNUSED_KHR),
                    any_hit_shader: rahit
                        .as_ref()
                        .map(|a| a.0.get() + base_shader_index)
                        .unwrap_or(vk::SHADER_UNUSED_KHR),
                    intersection_shader: rint
                        .as_ref()
                        .map(|a| a.0.get() + base_shader_index)
                        .unwrap_or(vk::SHADER_UNUSED_KHR),
                    ..Default::default()
                },
            )
            .collect()
    }

    pub fn remove_unused_shaders(&mut self) {
        let used_shader_indices = self
            .groups
            .iter()
            .flat_map(|(rchit, rahit, rint)| {
                rchit
                    .iter()
                    .map(|a| a.0)
                    .chain(rahit.iter().map(|a| a.0))
                    .chain(rint.iter().map(|a| a.0))
            })
            .collect::<BTreeSet<NonMaxU32>>();
        let has_unused_shaders = used_shader_indices.len() != self.shaders.len();
        if !has_unused_shaders {
            return;
        }

        let mut shader_map: Vec<Option<NonMaxU32>> = vec![None; self.shaders.len()];
        for (new_index, old_index) in used_shader_indices.iter().enumerate() {
            shader_map[old_index.get() as usize] = Some(NonMaxU32::new(new_index as u32).unwrap());
        }
        self.shaders = used_shader_indices
            .into_iter()
            .map(|index| self.shaders[index.get() as usize].clone())
            .collect::<Vec<_>>();
        self.groups = self
            .groups
            .iter()
            .map(|(rchit, rahit, rint)| {
                (
                    rchit.as_ref().map(|a| {
                        HitGroupShaderIndexClosestHit(shader_map[a.0.get() as usize].unwrap())
                    }),
                    rahit.as_ref().map(|a| {
                        HitGroupShaderIndexAnyHit(shader_map[a.0.get() as usize].unwrap())
                    }),
                    rint.as_ref().map(|a| {
                        HitGroupShaderIndexIntersection(shader_map[a.0.get() as usize].unwrap())
                    }),
                )
            })
            .collect();
    }
    pub fn pick(&self, indices: impl Iterator<Item = u32> + Clone) -> HitGroup {
        let used_shader_indices = indices
            .clone()
            .map(|i| &self.groups[i as usize])
            .flat_map(|(rchit, rahit, rint)| {
                rchit
                    .iter()
                    .map(|a| a.0)
                    .chain(rahit.iter().map(|a| a.0))
                    .chain(rint.iter().map(|a| a.0))
            })
            .collect::<BTreeSet<NonMaxU32>>();

        let mut shader_map: Vec<Option<NonMaxU32>> = vec![None; self.shaders.len()];
        for (new_index, old_index) in used_shader_indices.iter().enumerate() {
            shader_map[old_index.get() as usize] = Some(NonMaxU32::new(new_index as u32).unwrap());
        }
        let shaders = used_shader_indices
            .into_iter()
            .map(|index| self.shaders[index.get() as usize].clone())
            .collect::<Vec<_>>();
        let groups = indices
            .map(|i| &self.groups[i as usize])
            .map(|(rchit, rahit, rint)| {
                (
                    rchit.as_ref().map(|a| {
                        HitGroupShaderIndexClosestHit(shader_map[a.0.get() as usize].unwrap())
                    }),
                    rahit.as_ref().map(|a| {
                        HitGroupShaderIndexAnyHit(shader_map[a.0.get() as usize].unwrap())
                    }),
                    rint.as_ref().map(|a| {
                        HitGroupShaderIndexIntersection(shader_map[a.0.get() as usize].unwrap())
                    }),
                )
            })
            .collect();

        HitGroup {
            ty: self.ty,
            shaders,
            groups,
        }
    }
}

pub struct PipelineGroupManager<const NUM_RAYTYPES: usize> {
    /// Mapping from ray type index to `pipelines` index.
    ray_types: [u8; NUM_RAYTYPES],

    pipelines: SmallVec<[RayTracingPipelineManager; NUM_RAYTYPES]>,
    free_hitgroup_handles: Vec<HitgroupHandle>,
}

impl<const NUM_RAYTYPES: usize> PipelineGroupManager<NUM_RAYTYPES> {
    pub fn pipeline_index_of_raytype(&self, raytype: u32) -> u8 {
        self.ray_types[raytype as usize]
    }
    pub fn build(
        &mut self,
        pipeline_cache: &PipelineCache,
        assets: &Assets<ShaderModule>,
        pool: &DeferredOperationTaskPool,
    ) -> Option<SmallVec<[CachedPipeline<RenderObject<RayTracingPipeline>>; NUM_RAYTYPES]>> {
        let mut pipelines =
            SmallVec::<[CachedPipeline<RenderObject<RayTracingPipeline>>; NUM_RAYTYPES]>::new();
        for pipeline in self.pipelines.iter_mut() {
            pipelines.push(pipeline.build(pipeline_cache, assets, pool)?);
        }
        Some(pipelines)
    }
    pub fn add_hitgroup(
        &mut self,
        hitgroup: HitGroup,
        pipeline_cache: &PipelineCache,
    ) -> HitgroupHandle {
        assert_eq!(hitgroup.groups.len(), NUM_RAYTYPES);
        let handle = self.free_hitgroup_handles.pop();

        let mut new_handle: Option<HitgroupHandle> = None;
        for (i, pipeline_manager) in self.pipelines.iter_mut().enumerate() {
            let indices =
                self.ray_types
                    .iter()
                    .enumerate()
                    .filter_map(|(raytype, &pipeline_index)| {
                        if pipeline_index as usize == i {
                            Some(raytype as u32)
                        } else {
                            None
                        }
                    });
            let hitgroup = hitgroup.pick(indices);
            if let Some(handle) = handle {
                pipeline_manager.set_hitgroup(handle, hitgroup, pipeline_cache);
            } else {
                let handle = pipeline_manager.add_hitgroup(hitgroup, pipeline_cache);
                if let Some(existing) = &new_handle {
                    // Handle obtained from all pipelines should be the same.
                    assert_eq!(existing, &handle);
                } else {
                    new_handle = Some(handle);
                }
            }
        }
        if let Some(handle) = handle {
            assert!(new_handle.is_none());
            handle
        } else {
            new_handle.unwrap()
        }
    }
    pub fn remove_hitgroup(&mut self, handles: HitgroupHandle) {
        for pipeline in self.pipelines.iter_mut() {
            pipeline.remove_hitgroup(handles);
        }
        self.free_hitgroup_handles.push(handles);
    }
}
