use std::{alloc::Layout, sync::Arc};

use ash::{prelude::VkResult, vk};
use macros::set_layout;

use crate::{
    shader::{self, SpecializedReflectedShader, SpecializedShader},
    HasDevice, PipelineCache, PipelineLayout, ReflectedShaderModule,
};

pub struct RayTracingPipeline {
    layout: Arc<PipelineLayout>,
    pipeline: vk::Pipeline,
    num_raygen: u32,
    num_raymiss: u32,
    num_callable: u32,
    num_hitgroup: u32,
}
pub enum RayTracingHitGroupType {
    Procedural,
    Triangle,
}
impl From<RayTracingHitGroupType> for vk::RayTracingShaderGroupTypeKHR {
    fn from(value: RayTracingHitGroupType) -> Self {
        match value {
            RayTracingHitGroupType::Triangle => {
                vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP
            }
            RayTracingHitGroupType::Procedural => {
                vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP
            }
        }
    }
}
impl HasDevice for RayTracingPipeline {
    fn device(&self) -> &Arc<crate::Device> {
        self.layout.device()
    }
}

pub struct RayTracingPipelineLibrary {
    layout: Arc<PipelineLayout>,
    pipeline: vk::Pipeline,
    num_raygen: u32,
    num_raymiss: u32,
    num_callable: u32,
    num_hitgroup: u32,
}
impl HasDevice for RayTracingPipelineLibrary {
    fn device(&self) -> &Arc<crate::Device> {
        self.layout.device()
    }
}

pub struct RayTracingPipelineLibraryCreateInfo<'a> {
    pub pipeline_cache: Option<&'a PipelineCache>,
    pub pipeline_create_flags: vk::PipelineCreateFlags,
    pub max_pipeline_ray_recursion_depth: u32,
    pub max_pipeline_ray_payload_size: u32,
    pub max_pipeline_ray_hit_attribute_size: u32,
}
impl RayTracingPipelineLibrary {
    pub fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }
    pub fn create_for_hitgroup<'a>(
        layout: Arc<PipelineLayout>,
        rchit_shader: Option<SpecializedShader<'a>>,
        rint_shader: Option<SpecializedShader<'a>>,
        rahit_shader: Option<SpecializedShader<'a>>,
        info: RayTracingPipelineLibraryCreateInfo<'a>,
        ty: RayTracingHitGroupType,
    ) -> VkResult<Self> {
        let mut stages: [vk::PipelineShaderStageCreateInfo; 3] = [Default::default(); 3];
        let mut specialization_info: [vk::SpecializationInfo; 3] = [Default::default(); 3];
        let mut num_stages: usize = 0;
        let mut rchit_stage: u32 = 0;
        let mut rint_stage: u32 = 0;
        let mut rahit_stage: u32 = 0;
        if let Some(shader) = rchit_shader.as_ref() {
            assert_eq!(shader.stage, vk::ShaderStageFlags::CLOSEST_HIT_KHR);
            rchit_stage = num_stages as u32;
            specialization_info[num_stages] = vk::SpecializationInfo {
                map_entry_count: shader.specialization_info.entries.len() as u32,
                p_map_entries: shader.specialization_info.entries.as_ptr(),
                data_size: shader.specialization_info.data.len(),
                p_data: shader.specialization_info.data.as_ptr() as *const _,
            };
            stages[num_stages] = vk::PipelineShaderStageCreateInfo {
                flags: shader.flags,
                stage: vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                module: shader.shader.raw(),
                p_name: shader.entry_point.as_ptr(),
                p_specialization_info: &specialization_info[num_stages],
                ..Default::default()
            };
            num_stages += 1;
        }

        if let Some(shader) = rint_shader.as_ref() {
            assert_eq!(shader.stage, vk::ShaderStageFlags::INTERSECTION_KHR);
            rint_stage = num_stages as u32;
            specialization_info[num_stages] = vk::SpecializationInfo {
                map_entry_count: shader.specialization_info.entries.len() as u32,
                p_map_entries: shader.specialization_info.entries.as_ptr(),
                data_size: shader.specialization_info.data.len(),
                p_data: shader.specialization_info.data.as_ptr() as *const _,
            };
            stages[num_stages] = vk::PipelineShaderStageCreateInfo {
                flags: shader.flags,
                stage: vk::ShaderStageFlags::INTERSECTION_KHR,
                module: shader.shader.raw(),
                p_name: shader.entry_point.as_ptr(),
                p_specialization_info: &specialization_info[num_stages],
                ..Default::default()
            };
            num_stages += 1;
        }

        if let Some(shader) = rahit_shader.as_ref() {
            assert_eq!(shader.stage, vk::ShaderStageFlags::ANY_HIT_KHR);
            rahit_stage = num_stages as u32;
            specialization_info[num_stages] = vk::SpecializationInfo {
                map_entry_count: shader.specialization_info.entries.len() as u32,
                p_map_entries: shader.specialization_info.entries.as_ptr(),
                data_size: shader.specialization_info.data.len(),
                p_data: shader.specialization_info.data.as_ptr() as *const _,
            };
            stages[num_stages] = vk::PipelineShaderStageCreateInfo {
                flags: shader.flags,
                stage: vk::ShaderStageFlags::ANY_HIT_KHR,
                module: shader.shader.raw(),
                p_name: shader.entry_point.as_ptr(),
                p_specialization_info: &specialization_info[num_stages],
                ..Default::default()
            };
            num_stages += 1;
        }

        unsafe {
            let mut pipeline = vk::Pipeline::null();
            (layout
                .device()
                .rtx_loader()
                .fp()
                .create_ray_tracing_pipelines_khr)(
                layout.device().handle(),
                vk::DeferredOperationKHR::null(),
                info.pipeline_cache.map(|a| a.raw()).unwrap_or_default(),
                1,
                &vk::RayTracingPipelineCreateInfoKHR {
                    flags: vk::PipelineCreateFlags::LIBRARY_KHR | info.pipeline_create_flags,
                    stage_count: num_stages as u32,
                    p_stages: stages.as_ptr(),
                    group_count: 1,
                    p_groups: &vk::RayTracingShaderGroupCreateInfoKHR {
                        ty: ty.into(),
                        closest_hit_shader: rchit_stage,
                        any_hit_shader: rahit_stage,
                        intersection_shader: rint_stage,
                        ..Default::default()
                    },
                    max_pipeline_ray_recursion_depth: info.max_pipeline_ray_recursion_depth,
                    p_library_interface: &vk::RayTracingPipelineInterfaceCreateInfoKHR {
                        max_pipeline_ray_payload_size: info.max_pipeline_ray_payload_size,
                        max_pipeline_ray_hit_attribute_size: info
                            .max_pipeline_ray_hit_attribute_size,
                        ..Default::default()
                    },
                    layout: layout.raw(),
                    ..Default::default()
                },
                std::ptr::null(),
                &mut pipeline,
            )
            .result()?;
            Ok(Self {
                layout,
                pipeline,
                num_raygen: 0,
                num_raymiss: 0,
                num_callable: 0,
                num_hitgroup: 1,
            })
        }
    }
    pub fn create_for_shaders<'a>(
        layout: Arc<PipelineLayout>,
        shaders: &[SpecializedShader<'a>],
        info: RayTracingPipelineLibraryCreateInfo<'a>,
    ) -> VkResult<Self> {
        let mut num_raygen: u32 = 0;
        let mut num_raymiss: u32 = 0;
        let mut num_callable: u32 = 0;
        for shader in shaders.iter() {
            match shader.stage {
                vk::ShaderStageFlags::RAYGEN_KHR => {
                    assert!(
                        num_callable == 0 && num_raymiss == 0,
                        "Ray Generation Shader must be specified before everything else"
                    );
                    num_raygen += 1;
                }
                vk::ShaderStageFlags::MISS_KHR => {
                    assert!(
                        num_callable == 0,
                        "Miss Shader must be specified before Callable shaders"
                    );
                    num_raymiss += 1;
                }
                vk::ShaderStageFlags::CALLABLE_KHR => {
                    num_callable += 1;
                }
                vk::ShaderStageFlags::ANY_HIT_KHR
                | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                | vk::ShaderStageFlags::INTERSECTION_KHR => {
                    panic!("For hitgroup shaders, use `RayTracingPipelineLibrary::create_for_hitgroup`")
                }
                _ => {
                    panic!()
                }
            }
        }
        unsafe {
            let specialization_infos: Vec<_> = shaders
                .iter()
                .map(|shader| shader.specialization_info.raw_info())
                .collect();
            let (stages, groups): (Vec<_>, Vec<_>) = shaders
                .iter()
                .enumerate()
                .map(|(i, shader)| {
                    let stage = vk::PipelineShaderStageCreateInfo {
                        flags: shader.flags,
                        stage: shader.stage,
                        module: shader.shader.raw(),
                        p_name: shader.entry_point.as_ptr(),
                        p_specialization_info: specialization_infos.as_ptr().add(i),
                        ..Default::default()
                    };
                    let group = vk::RayTracingShaderGroupCreateInfoKHR {
                        ty: vk::RayTracingShaderGroupTypeKHR::GENERAL,
                        general_shader: i as u32,
                        ..Default::default()
                    };
                    (stage, group)
                })
                .unzip();

            let mut pipeline = vk::Pipeline::null();
            (layout
                .device()
                .rtx_loader()
                .fp()
                .create_ray_tracing_pipelines_khr)(
                layout.device().handle(),
                vk::DeferredOperationKHR::null(),
                info.pipeline_cache.map(|a| a.raw()).unwrap_or_default(),
                1,
                &vk::RayTracingPipelineCreateInfoKHR {
                    flags: vk::PipelineCreateFlags::LIBRARY_KHR | info.pipeline_create_flags,
                    stage_count: stages.len() as u32,
                    p_stages: stages.as_ptr(),
                    group_count: groups.len() as u32,
                    p_groups: groups.as_ptr(),
                    max_pipeline_ray_recursion_depth: info.max_pipeline_ray_recursion_depth,
                    p_library_interface: &vk::RayTracingPipelineInterfaceCreateInfoKHR {
                        max_pipeline_ray_payload_size: info.max_pipeline_ray_payload_size,
                        max_pipeline_ray_hit_attribute_size: info
                            .max_pipeline_ray_hit_attribute_size,
                        ..Default::default()
                    },
                    layout: layout.raw(),
                    ..Default::default()
                },
                std::ptr::null(),
                &mut pipeline,
            )
            .result()?;
            Ok(Self {
                layout,
                pipeline,
                num_raygen,
                num_raymiss,
                num_callable,
                num_hitgroup: 0,
            })
        }
    }
}

impl RayTracingPipeline {
    pub fn create_from_libraries<'a>(
        libs: &'a [RayTracingPipelineLibrary],
        info: RayTracingPipelineLibraryCreateInfo<'a>,
    ) -> VkResult<Self> {
        let device = libs[0].device().clone();
        let layout = libs[0].layout.clone();

        let raw_libs: Vec<_> = libs.iter().map(|lib| lib.raw()).collect();

        let mut num_raygen: u32 = 0;
        let mut num_raymiss: u32 = 0;
        let mut num_hitgroup: u32 = 0;
        let mut num_callable: u32 = 0;
        for lib in libs.iter() {
            if lib.num_raygen != 0 {
                assert!(
                    num_callable == 0 && num_raymiss == 0 && num_hitgroup == 0,
                    "Ray Generation Shader must be specified before everything else"
                );
            }
            if lib.num_raymiss != 0 {
                assert!(
                    num_callable == 0 && num_hitgroup == 0,
                    "Miss Shader must be specified before callable and hitgroup shaders"
                );
            }
            if lib.num_callable != 0 {
                assert!(
                    num_hitgroup == 0,
                    "Callable Shader must be specified before hitgroup shaders"
                );
            }
            num_raygen += lib.num_raygen;
            num_raymiss += lib.num_raymiss;
            num_hitgroup += lib.num_hitgroup;
            num_callable += lib.num_callable;
        }
        let mut pipeline: vk::Pipeline = vk::Pipeline::null();
        unsafe {
            (device.rtx_loader().fp().create_ray_tracing_pipelines_khr)(
                layout.device().handle(),
                vk::DeferredOperationKHR::null(),
                info.pipeline_cache.map(|a| a.raw()).unwrap_or_default(),
                1,
                &vk::RayTracingPipelineCreateInfoKHR {
                    flags: info.pipeline_create_flags,
                    max_pipeline_ray_recursion_depth: info.max_pipeline_ray_recursion_depth,
                    layout: layout.raw(),
                    p_library_info: &vk::PipelineLibraryCreateInfoKHR {
                        library_count: raw_libs.len() as u32,
                        p_libraries: raw_libs.as_ptr(),
                        ..Default::default()
                    },
                    p_library_interface: &vk::RayTracingPipelineInterfaceCreateInfoKHR {
                        max_pipeline_ray_payload_size: info.max_pipeline_ray_payload_size,
                        max_pipeline_ray_hit_attribute_size: info
                            .max_pipeline_ray_hit_attribute_size,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                std::ptr::null(),
                &mut pipeline,
            )
            .result()?;
        }

        Ok(Self {
            layout,
            pipeline,
            num_callable,
            num_hitgroup,
            num_raygen,
            num_raymiss,
        })
    }
    pub fn get_shader_group_handles(&self) -> VkResult<SbtHandles> {
        SbtHandles::new(self)
    }
}

pub struct SbtHandles {
    data: Box<[u8]>,
    handle_layout: Layout,
    group_base_alignment: u32,
    num_raygen: u32,
    num_miss: u32,
    num_callable: u32,
    num_hitgroup: u32,
}
impl SbtHandles {
    pub fn handle_layout(&self) -> &Layout {
        &self.handle_layout
    }
    fn new(pipeline: &RayTracingPipeline) -> VkResult<SbtHandles> {
        let total_num_groups = pipeline.num_hitgroup
            + pipeline.num_raymiss
            + pipeline.num_callable
            + pipeline.num_raygen;
        let rtx_properties = &pipeline.device().physical_device().properties().ray_tracing;
        let sbt_handles_host_vec = unsafe {
            pipeline
                .device()
                .rtx_loader()
                .get_ray_tracing_shader_group_handles(
                    pipeline.pipeline,
                    0,
                    total_num_groups,
                    // VUID-vkGetRayTracingShaderGroupHandlesKHR-dataSize-02420
                    // dataSize must be at least VkPhysicalDeviceRayTracingPipelinePropertiesKHR::shaderGroupHandleSize × groupCount
                    rtx_properties.shader_group_handle_size as usize * total_num_groups as usize,
                )?
                .into_boxed_slice()
        };
        Ok(SbtHandles {
            data: sbt_handles_host_vec,
            handle_layout: Layout::from_size_align(
                rtx_properties.shader_group_handle_size as usize,
                rtx_properties.shader_group_handle_alignment as usize,
            )
            .unwrap(),
            group_base_alignment: rtx_properties.shader_group_base_alignment,
            num_raygen: pipeline.num_raygen,
            num_miss: pipeline.num_raymiss,
            num_callable: pipeline.num_callable,
            num_hitgroup: pipeline.num_hitgroup,
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
    pub fn hitgroup(&self, index: usize) -> &[u8] {
        let start = self.handle_layout.size()
            * (index
                + self.num_miss as usize
                + self.num_callable as usize
                + self.num_raygen as usize);
        let end = start + self.handle_layout.size();
        &self.data[start..end]
    }
}
