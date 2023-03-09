use std::{alloc::Layout, sync::Arc};

use ash::{prelude::VkResult, vk};
use macros::set_layout;

use crate::{
    shader::{self, SpecializedReflectedShader},
    HasDevice, PipelineCache, PipelineLayout, PipelineLayoutBuilder, ReflectedShaderModule,
};

pub struct RayTracingPipeline {
    layout: Arc<PipelineLayout>,
    pipeline: vk::Pipeline,
}
impl HasDevice for RayTracingPipeline {
    fn device(&self) -> &Arc<crate::Device> {
        self.layout.device()
    }
}

pub struct RayTracingPipelineLibrary {
    layout: Arc<PipelineLayout>,
    pipeline: vk::Pipeline,
}

pub struct RayTracingPipelineLibraryCreateInfo<'a> {
    pub pipeline_cache: Option<&'a PipelineCache>,
    pipeline_create_flags: vk::PipelineCreateFlags,
    pipeline_layout_create_flags: vk::PipelineLayoutCreateFlags,
    max_pipeline_ray_recursion_depth: u32,
}
impl RayTracingPipelineLibrary {
    pub fn create_for_shaders<'a>(
        shaders: &[SpecializedReflectedShader<'a>],
        info: RayTracingPipelineLibraryCreateInfo<'a>,
    ) -> VkResult<Self> {
        unsafe {
            let specialization_infos: Vec<_> = shaders
                .iter()
                .map(|shader| shader.specialization_info.raw_info())
                .collect();
            let mut layout_builder: Option<PipelineLayoutBuilder> = None;
            let (stages, groups): (Vec<_>, Vec<_>) = shaders
                .iter()
                .enumerate()
                .map(|(i, shader)| {
                    let stage = vk::PipelineShaderStageCreateInfo {
                        flags: shader.flags,
                        stage: shader.entry_point().stage,
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
                    if layout_builder.is_none() {
                        layout_builder = Some(PipelineLayoutBuilder::new(
                            shader.device().clone(),
                            info.pipeline_layout_create_flags,
                        ));
                    }
                    layout_builder
                        .as_mut()
                        .unwrap()
                        .add_entry_point(shader.entry_point());
                    (stage, group)
                })
                .unzip();
            let layout = layout_builder.unwrap().build()?;

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
                    layout: layout.raw(),
                    ..Default::default()
                },
                std::ptr::null(),
                &mut pipeline,
            )
            .result()?;
            Ok(Self {
                layout: Arc::new(layout),
                pipeline,
            })
        }
    }
}

impl RayTracingPipeline {
    pub fn get_shader_group_handles(&self) -> SbtHandles {
        todo!()
        // SbtHandles::new(self, num_raygen, num_miss, num_callable, num_hitgroup)
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
    fn new(
        pipeline: &RayTracingPipeline,
        num_raygen: u32,
        num_miss: u32,
        num_callable: u32,
        num_hitgroup: u32,
    ) -> VkResult<SbtHandles> {
        let total_num_groups = num_hitgroup + num_miss + num_callable + num_raygen;
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
            num_raygen,
            num_miss,
            num_callable,
            num_hitgroup,
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
