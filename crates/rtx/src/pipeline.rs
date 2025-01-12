use std::{alloc::Layout, collections::BTreeSet, fmt::Debug, num::NonZeroU64, sync::Arc};

use bevy::{
    asset::{AssetId, Assets},
    utils::tracing,
};
use nonmax::{self, NonMaxU32};

use rhyolite::{
    ash::{khr::ray_tracing_pipeline::Meta as RayTracingPipelineExt, prelude::VkResult, vk},
    deferred::{DeferredOperationTaskPool, Task},
    pipeline::{
        CachedPipeline, Pipeline, PipelineBuildInfo, PipelineCache, PipelineInner, PipelineLayout,
    },
    shader::{ShaderModule, SpecializedShader},
    sync::GPUBorrowed,
    Device, HasDevice,
};

use crate::SbtManager;

#[derive(Clone)]
pub struct RayTracingPipeline {
    inner: Arc<PipelineInner>,
    handles: SbtHandles,
}
impl HasDevice for RayTracingPipeline {
    fn device(&self) -> &Device {
        self.inner.device()
    }
}
impl RayTracingPipeline {
    pub fn handles(&self) -> &SbtHandles {
        &self.handles
    }
}

impl RayTracingPipeline {
    pub fn raw(&self) -> vk::Pipeline {
        self.inner.raw()
    }
}
impl Pipeline for RayTracingPipeline {
    type BuildInfo = RayTracingPipelineBuildInfo;
    const TYPE: vk::PipelineBindPoint = vk::PipelineBindPoint::RAY_TRACING_KHR;

    fn as_raw(&self) -> vk::Pipeline {
        self.inner.raw()
    }
    fn from_built(
        info: &mut RayTracingPipelineBuildInfo,
        item: <Self::BuildInfo as PipelineBuildInfo>::Pipeline,
    ) -> Self {
        tracing::info!(
            "Ray Tracing Pipeline built with {} libs, {} raygen, {} miss, {} callable, {} hitgroup",
            info.libraries.len(),
            info.num_raygen,
            info.num_miss,
            info.num_callable,
            info.num_hitgroup
        );
        RayTracingPipeline {
            handles: SbtHandles::new(
                item.device(),
                item.raw(),
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
        item: <Self::BuildInfo as PipelineBuildInfo>::Pipeline,
    ) -> Self {
        tracing::info!(
            "Ray Tracing Pipeline built with {} libs, {} raygen, {} miss, {} callable, {} hitgroup",
            info.libraries.len(),
            info.num_raygen,
            info.num_miss,
            info.num_callable,
            info.num_hitgroup
        );
        RayTracingPipeline {
            handles: SbtHandles::new(
                item.device(),
                item.raw(),
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
    groups: Vec<vk::RayTracingShaderGroupCreateInfoKHR<'static>>,
    libraries: Vec<RayTracingPipelineLibrary>,
    hitgroup_mapping: Vec<u32>,
    num_raygen: u8,
    num_miss: u8,
    num_callable: u8,
    num_hitgroup: u32,
}
unsafe impl Send for RayTracingPipelineBuildInfo {}
unsafe impl Sync for RayTracingPipelineBuildInfo {}

impl PipelineBuildInfo for RayTracingPipelineBuildInfo {
    type Pipeline = PipelineInner;

    fn build(
        &mut self,
        pool: &DeferredOperationTaskPool,
        assets: &Assets<ShaderModule>,
        cache: vk::PipelineCache,
    ) -> Option<Task<Self::Pipeline>> {
        let modules = self
            .stages
            .iter()
            .map(|shader| assets.get(&shader.shader).map(ShaderModule::raw))
            .collect::<Option<Vec<_>>>()?;
        let stages = self.stages.clone();
        let groups = self.groups.clone();
        let common = self.common.clone();
        let libraries = self.libraries.clone();

        let driver_properties = common
            .layout
            .device()
            .physical_device()
            .properties()
            .get::<vk::PhysicalDeviceDriverProperties>();
        // White listing drivers correctly implementing dho for rtx pipeline
        let should_use_dho = [vk::DriverId::MESA_RADV].contains(&driver_properties.driver_id);

        Some(pool.schedule_dho(should_use_dho, move |dho| {
            let device = common.layout.device().clone();
            let specialization_info: Vec<vk::SpecializationInfo<'static>> = stages
                .iter()
                .map(|shader| vk::SpecializationInfo {
                    map_entry_count: shader.specialization_info.entries().len() as u32,
                    p_map_entries: if shader.specialization_info.entries().is_empty() {
                        std::ptr::null()
                    } else {
                        shader.specialization_info.entries().as_ptr()
                    },
                    p_data: if shader.specialization_info.data().is_empty() {
                        std::ptr::null()
                    } else {
                        shader.specialization_info.data().as_ptr() as *const std::ffi::c_void
                    },
                    data_size: shader.specialization_info.data().len(),
                    _marker: std::marker::PhantomData,
                })
                .collect::<Vec<_>>();
            let raw_stages: Vec<vk::PipelineShaderStageCreateInfo<'static>> = stages
                .iter()
                .zip(modules)
                .zip(specialization_info.iter())
                .map(
                    |((shader, module), specialization_info)| vk::PipelineShaderStageCreateInfo {
                        stage: shader.stage,
                        module,
                        flags: shader.flags,
                        p_specialization_info: specialization_info,
                        p_name: shader.entry_point.as_ptr(),
                        ..Default::default()
                    },
                )
                .collect::<Vec<_>>();
            // Create self-referencing pipeline create args
            let mut args = Box::new((
                vk::RayTracingPipelineCreateInfoKHR {
                    max_pipeline_ray_recursion_depth: common.max_pipeline_ray_recursion_depth,
                    layout: common.layout.raw(),
                    flags: common.flags,
                    ..Default::default()
                },
                vk::RayTracingPipelineInterfaceCreateInfoKHR {
                    max_pipeline_ray_payload_size: common.max_pipeline_ray_payload_size,
                    max_pipeline_ray_hit_attribute_size: common.max_pipeline_ray_hit_attribute_size,
                    ..Default::default()
                },
                common.dynamic_states,
                vk::PipelineDynamicStateCreateInfo::default(),
                libraries
                    .iter()
                    .map(|library| library.pipeline.raw())
                    .collect::<Vec<_>>(),
                vk::PipelineLibraryCreateInfoKHR::default(),
                groups,
                stages,
                specialization_info,
                raw_stages,
                common.layout,
                libraries,
            ));
            args.0.p_library_interface = &args.1;
            args.3.dynamic_state_count = args.2.len() as u32;
            if args.3.dynamic_state_count > 0 {
                args.0.p_dynamic_state = &args.3;
            }

            args.5.library_count = args.4.len() as u32;
            if args.5.library_count > 0 {
                args.5.p_libraries = args.4.as_ptr();
                args.0.p_library_info = &args.5;
            }
            args.0.group_count = args.6.len() as u32;
            if args.0.group_count > 0 {
                args.0.p_groups = args.6.as_ptr();
            }
            args.0.stage_count = args.9.len() as u32;
            if args.0.stage_count > 0 {
                args.0.p_stages = args.9.as_ptr();
            }
            let (result, pipeline) = unsafe {
                let mut pipeline = vk::Pipeline::null();
                let result = (device
                    .extension::<RayTracingPipelineExt>()
                    .fp()
                    .create_ray_tracing_pipelines_khr)(
                    device.handle(),
                    dho,
                    cache,
                    1,
                    &args.0,
                    std::ptr::null(),
                    &mut pipeline,
                );
                (result, pipeline)
            };
            (result, PipelineInner::from_raw(device, pipeline), args)
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
impl Pipeline for RayTracingPipelineLibrary {
    type BuildInfo = RayTracingPipelineBuildInfo;
    const TYPE: vk::PipelineBindPoint = vk::PipelineBindPoint::RAY_TRACING_KHR;

    fn as_raw(&self) -> vk::Pipeline {
        self.pipeline.raw()
    }
    fn from_built(
        info: &mut RayTracingPipelineBuildInfo,
        item: <Self::BuildInfo as PipelineBuildInfo>::Pipeline,
    ) -> Self {
        assert!(info
            .common
            .flags
            .contains(vk::PipelineCreateFlags::LIBRARY_KHR));
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
        info: Self::BuildInfo,
        item: <Self::BuildInfo as PipelineBuildInfo>::Pipeline,
    ) -> Self {
        assert!(info
            .common
            .flags
            .contains(vk::PipelineCreateFlags::LIBRARY_KHR));
        RayTracingPipelineLibrary {
            pipeline: Arc::new(item),
            // if we're building with owned info, that means the library would never be rebuilt.
            // so, we don't need to keep a list of shaders to tell when we should rebuild.
            shaders: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
pub struct HitgroupHandle {
    pub(crate) generation: NonZeroU64,
    pub(crate) handle: u16,
}

/// Utility for managing ray tracing pipeline lifetimes.
/// It assumes that the ray gen shaders, miss shaders, and callable shaders are immutable,
/// while the hitgroup shaders may change frequently.
/// As such, we create a pipeline library for the ray gen, miss, and callable shaders,
/// and a pipeline library for each of the hitgroups.
enum RayTracingPipelineManagerImpl {
    Native(RayTracingPipelineManagerNative),
    PipelineLibrary(RayTracingPipelineManagerPipelineLibrary),
}
pub struct RayTracingPipelineManager {
    inner: RayTracingPipelineManagerImpl,
    latest_generation: u64,
    latest_pipeline: Option<CachedPipeline<RayTracingPipeline>>,

    /// The pipeline generation of [`Self::pipeline`]
    pipeline_generation: u64,
    pipeline: Option<CachedPipeline<RayTracingPipeline>>,
}

impl RayTracingPipelineManager {
    pub fn new(
        common: RayTracingPipelineBuildInfoCommon,
        raygen_shaders: Vec<SpecializedShader>,
        miss_shaders: Vec<SpecializedShader>,
        callable_shaders: Vec<SpecializedShader>,
        pipeline_cache: &PipelineCache,
    ) -> Self {
        let driver_properties = common
            .layout
            .device()
            .physical_device()
            .properties()
            .get::<vk::PhysicalDeviceDriverProperties>();
        // White listing drivers correctly implementing pipeline libraries.
        // Tested platforms:
        // - NVIDIA_PROPRIETARY, Windows PC: Ok
        // - AMD_PROPRIETARY, Windows, 31.0.24033.1003: DEVICE_LOST
        // - AMD_PROPRIETARY, Windows, 31.0.24027.1012, ASUS Ally: Ok, but crash on hot reload
        let should_use_pipeline_library =
            [vk::DriverId::NVIDIA_PROPRIETARY].contains(&driver_properties.driver_id);
        let inner = if should_use_pipeline_library {
            RayTracingPipelineManagerImpl::PipelineLibrary(
                RayTracingPipelineManagerPipelineLibrary::new(
                    common,
                    raygen_shaders,
                    miss_shaders,
                    callable_shaders,
                    pipeline_cache,
                ),
            )
        } else {
            RayTracingPipelineManagerImpl::Native(RayTracingPipelineManagerNative::new(
                common,
                raygen_shaders,
                miss_shaders,
                callable_shaders,
            ))
        };
        RayTracingPipelineManager {
            inner,
            pipeline_generation: 0,
            latest_generation: 1,
            pipeline: None,
            latest_pipeline: None,
        }
    }

    pub fn add_hitgroup(
        &mut self,
        hitgroup: HitGroup,
        pipeline_cache: &PipelineCache,
    ) -> HitgroupHandle {
        let handle = match &mut self.inner {
            RayTracingPipelineManagerImpl::Native(manager) => manager.add_hitgroup(hitgroup),
            RayTracingPipelineManagerImpl::PipelineLibrary(manager) => {
                manager.add_hitgroup(hitgroup, &pipeline_cache)
            }
        };
        self.latest_generation += 1;
        HitgroupHandle {
            generation: NonZeroU64::new(self.latest_generation).unwrap(),
            handle,
        }
    }

    pub fn remove_hitgroup(&mut self, handle: HitgroupHandle) {
        self.latest_generation += 1;
        match &mut self.inner {
            RayTracingPipelineManagerImpl::Native(manager) => {
                manager.remove_hitgroup(handle.handle)
            }
            RayTracingPipelineManagerImpl::PipelineLibrary(manager) => {
                manager.remove_hitgroup(handle.handle)
            }
        }
    }

    pub fn try_build<T>(
        &mut self,
        pipeline_cache: &PipelineCache,
        assets: &Assets<ShaderModule>,
        pool: &DeferredOperationTaskPool,
        sbt_manager: &mut SbtManager<T>,
    ) {
        if let Some(pipeline) = &mut self.latest_pipeline {
            let pipeline = pipeline_cache.retrieve_pipeline(pipeline, assets, pool, true);
            if let Some(pipeline) = pipeline {
                tracing::info!(
                    "Current pipeline updated: {:?} ({})",
                    pipeline.as_raw(),
                    self.pipeline_generation
                );
                sbt_manager.pipeline_updated(self.pipeline_generation);
                self.pipeline = self.latest_pipeline.take();
            } else if let Some(pipeline) = &mut self.pipeline {
                pipeline_cache.retrieve_pipeline(pipeline, assets, pool, true);
            }
        } else if let Some(pipeline) = &mut self.pipeline {
            pipeline_cache.retrieve_pipeline(pipeline, assets, pool, true);
        }

        if self.latest_pipeline.is_none() {
            if self.pipeline_generation != self.latest_generation // rebuild if add_hitgroup or remove_hitgroup was called
                || self
                    .pipeline
                    .as_ref()
                    .map(|p| {
                        // rebuild if self.pipeline becomes outdated due to hot reload
                        self.latest_pipeline.is_none() && pipeline_cache.is_outdated(p)
                    })
                    .unwrap_or(false)
            {
                let new_pipeline = match &mut self.inner {
                    RayTracingPipelineManagerImpl::Native(manager) => {
                        Some(manager.build(pipeline_cache))
                    }
                    RayTracingPipelineManagerImpl::PipelineLibrary(manager) => {
                        manager.build(pipeline_cache, assets, pool)
                    }
                };
                if let Some(new_pipeline) = new_pipeline {
                    tracing::info!(
                        "Latest pipeline updated: generation {} -> {}",
                        self.pipeline_generation,
                        self.latest_generation
                    );
                    self.latest_pipeline = Some(new_pipeline);
                    self.pipeline_generation = self.latest_generation;
                }
            }
        }
    }

    pub fn get_pipeline(&mut self) -> Option<&mut RayTracingPipeline> {
        self.pipeline.as_mut()?.get_mut()
    }
}

struct RayTracingPipelineManagerNative {
    common: RayTracingPipelineBuildInfoCommon,
    num_raygen: u8,
    num_miss: u8,
    num_callable: u8,
    /// closest hit, any hit, intersection
    hitgroups: Vec<Option<HitGroup>>,
    available_indices: Vec<u16>,
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
            available_indices: Vec::new(),
        }
    }
    fn add_hitgroup(&mut self, hitgroup: HitGroup) -> u16 {
        tracing::info!("Adding hitgroup");
        let handle = self.available_indices.pop().unwrap_or_else(|| {
            let handle = self.hitgroups.len() as u16;
            self.hitgroups.push(None);
            handle
        });
        self.hitgroups[handle as usize] = Some(hitgroup);
        handle
    }
    #[track_caller]
    fn remove_hitgroup(&mut self, handle: u16) {
        assert!(self.hitgroups[handle as usize].is_some());
        self.hitgroups[handle as usize] = None;
    }
    fn build(&mut self, pipeline_cache: &PipelineCache) -> CachedPipeline<RayTracingPipeline> {
        let mut stages = Vec::new();
        let mut groups: Vec<vk::RayTracingShaderGroupCreateInfoKHR<'static>> = Vec::new();
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
        pipeline_cache.create(info)
    }
}

struct RayTracingPipelineManagerPipelineLibrary {
    common: RayTracingPipelineBuildInfoCommon,
    base_library: CachedPipeline<RayTracingPipelineLibrary>,
    hitgroup_libraries: Vec<Option<CachedPipeline<RayTracingPipelineLibrary>>>,
    available_indices: Vec<u16>,
    num_raygen: u8,
    num_miss: u8,
    num_callable: u8,
    dirty: bool,
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
        let mut info = RayTracingPipelineBuildInfo {
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
        info.common.flags |= vk::PipelineCreateFlags::LIBRARY_KHR;
        let base_library: CachedPipeline<RayTracingPipelineLibrary> = pipeline_cache.create(info);
        Self {
            common,
            base_library,
            hitgroup_libraries: Vec::new(),
            num_raygen,
            num_miss,
            num_callable,
            dirty: false,
            available_indices: Vec::new(),
        }
    }

    pub fn add_hitgroup(&mut self, hitgroup: HitGroup, pipeline_cache: &PipelineCache) -> u16 {
        let handle = self.available_indices.pop().unwrap_or_else(|| {
            let handle = self.hitgroup_libraries.len() as u16;
            self.hitgroup_libraries.push(None);
            handle
        });
        self.set_hitgroup(handle, hitgroup, pipeline_cache);
        handle
    }
    fn set_hitgroup(&mut self, handle: u16, hitgroup: HitGroup, pipeline_cache: &PipelineCache) {
        let mut info = RayTracingPipelineBuildInfo {
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
        info.common.flags |= vk::PipelineCreateFlags::LIBRARY_KHR;
        let library: CachedPipeline<RayTracingPipelineLibrary> = pipeline_cache.create(info);

        assert!(self.hitgroup_libraries[handle as usize].is_none());
        self.hitgroup_libraries[handle as usize] = Some(library);
        self.dirty = true;
    }

    #[track_caller]
    pub fn remove_hitgroup(&mut self, handle: u16) {
        assert!(self.hitgroup_libraries[handle as usize].is_some());
        self.hitgroup_libraries[handle as usize] = None;
        self.dirty = true;
    }
    pub fn build(
        &mut self,
        pipeline_cache: &PipelineCache,
        assets: &Assets<ShaderModule>,
        pool: &DeferredOperationTaskPool,
    ) -> Option<CachedPipeline<RayTracingPipeline>> {
        let libraries = std::iter::once(&mut self.base_library)
            .chain(
                self.hitgroup_libraries
                    .iter_mut()
                    .filter_map(Option::as_mut),
            )
            .map(|library| {
                pipeline_cache
                    .retrieve_pipeline(library, assets, pool, false)
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
    num_raygen: u8,
    num_miss: u8,
    num_callable: u8,
    num_hitgroup: u32,
}
impl Debug for SbtHandles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct HexStr<'a>(&'a [u8]);
        impl<'a> Debug for HexStr<'a> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("0x")?;
                for item in self.0 {
                    f.write_fmt(format_args!("{:x?}", item))?;
                }
                Ok(())
            }
        }
        let mut f_struct = f.debug_struct("SbtHandles");
        for i in 0..self.num_raygen {
            let data = self.rgen(i as usize);
            f_struct.field("raygen", &HexStr(data));
        }
        for i in 0..self.num_miss {
            let data = self.rmiss(i as usize);
            f_struct.field("miss", &HexStr(data));
        }
        for i in 0..self.num_callable {
            let data = self.callable(i as usize);
            f_struct.field("callable", &HexStr(data));
        }
        Ok(())
    }
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
                .extension::<RayTracingPipelineExt>()
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
            num_raygen: num_raygen,
            num_miss: num_miss,
            num_callable: num_callable,
            num_hitgroup: num_hitgroup,
        })
    }

    pub fn rgen(&self, index: usize) -> &[u8] {
        assert!(index < self.num_raygen as usize);
        // Note that
        // VUID-vkGetRayTracingShaderGroupHandlesKHR-dataSize-02420
        // dataSize must be at least VkPhysicalDeviceRayTracingPipelinePropertiesKHR::shaderGroupHandleSize × groupCount
        // This implies all handles are tightly packed. No need to call `pad_to_align` here
        let start = self.handle_layout.size() * index;
        let end = start + self.handle_layout.size();
        &self.data[start..end]
    }
    pub fn rmiss(&self, index: usize) -> &[u8] {
        assert!(index < self.num_miss as usize);
        let start = self.handle_layout.size() * (index + self.num_raygen as usize);
        let end = start + self.handle_layout.size();
        &self.data[start..end]
    }
    pub fn callable(&self, index: usize) -> &[u8] {
        assert!(index < self.num_callable as usize);
        let start =
            self.handle_layout.size() * (index + self.num_raygen as usize + self.num_miss as usize);
        let end = start + self.handle_layout.size();
        &self.data[start..end]
    }
    pub fn num_raygen(&self) -> u8 {
        self.num_raygen
    }
    pub fn num_miss(&self) -> u8 {
        self.num_miss
    }
    pub fn num_callable(&self) -> u8 {
        self.num_callable
    }
    pub fn num_hitgroup(&self) -> u32 {
        self.num_hitgroup
    }
    pub fn hitgroup(&self, hitgroup_index: u16) -> &[u8] {
        let index = self.hitgroup_handle_map[hitgroup_index as usize];
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

    fn to_vk_groups(
        &self,
        base_shader_index: u32,
    ) -> Vec<vk::RayTracingShaderGroupCreateInfoKHR<'static>> {
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
