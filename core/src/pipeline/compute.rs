use std::ffi::CString;

use ash::{prelude::VkResult, vk};

use crate::{shader::SpecializationInfo, HasDevice, ShaderModule};

use super::PipelineLayout;
use std::sync::Arc;

pub struct ComputePipeline {
    layout: Arc<PipelineLayout>,
    pipeline: vk::Pipeline,
}
impl ComputePipeline {
    pub fn layout(&self) -> &Arc<PipelineLayout> {
        &self.layout
    }
    pub unsafe fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }
    pub unsafe fn raw_layout(&self) -> vk::PipelineLayout {
        self.layout.raw()
    }
}
impl HasDevice for ComputePipeline {
    fn device(&self) -> &Arc<crate::Device> {
        self.layout.device()
    }
}
impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe { self.layout.device().destroy_pipeline(self.pipeline, None) }
    }
}
pub struct ComputePipelineCreateInfo<'a> {
    pub module: &'a ShaderModule,
    pub specialization: &'a SpecializationInfo,
    pub pipeline_layout_create_flags: vk::PipelineLayoutCreateFlags,
    pub pipeline_create_flags: vk::PipelineCreateFlags,
    pub compute_stage_flags: vk::PipelineShaderStageCreateFlags,
    pub entry_point: &'a str,
}

impl<'a> ComputePipelineCreateInfo<'a> {
    const DEFAULT_SPECIALIZATION: &'static SpecializationInfo = &SpecializationInfo::new();
    pub fn with_module(module: &'a ShaderModule) -> Self {
        Self {
            pipeline_create_flags: Default::default(),
            pipeline_layout_create_flags: Default::default(),
            compute_stage_flags: Default::default(),
            entry_point: "main",
            specialization: &Self::DEFAULT_SPECIALIZATION,
            module,
        }
    }
}

impl ComputePipeline {
    pub fn create(info: ComputePipelineCreateInfo) -> VkResult<Self> {
        let device = info.module.device().clone();
        let layout = PipelineLayout::for_layout(
            device.clone(),
            info.module
                .entry_points
                .get(info.entry_point)
                .unwrap()
                .clone(),
            info.pipeline_layout_create_flags,
        )?;
        let pipeline = unsafe {
            let entry_point = CString::new(info.entry_point).unwrap();
            let mut pipeline = vk::Pipeline::null();
            let specialization_info = vk::SpecializationInfo {
                map_entry_count: info.specialization.entries.len() as u32,
                p_map_entries: info.specialization.entries.as_ptr(),
                data_size: info.specialization.data.len(),
                p_data: info.specialization.data.as_ptr() as *const _,
            };
            (device.fp_v1_0().create_compute_pipelines)(
                device.handle(),
                vk::PipelineCache::null(),
                1,
                &vk::ComputePipelineCreateInfo {
                    flags: info.pipeline_create_flags,
                    stage: vk::PipelineShaderStageCreateInfo {
                        flags: info.compute_stage_flags,
                        stage: vk::ShaderStageFlags::COMPUTE,
                        module: info.module.raw(),
                        p_name: entry_point.as_c_str().as_ptr(),
                        p_specialization_info: &specialization_info,
                        ..Default::default()
                    },
                    layout: layout.raw(),
                    // Do not use pipeline derivative as they're not beneficial.
                    // https://stackoverflow.com/questions/37135130/vulkan-creating-and-benefit-of-pipeline-derivatives
                    base_pipeline_handle: vk::Pipeline::null(),
                    base_pipeline_index: 0,
                    ..Default::default()
                },
                std::ptr::null(),
                (&mut pipeline) as *mut _,
            )
            .result_with_success(pipeline)
        }?;
        Ok(Self {
            layout: Arc::new(layout),
            pipeline,
        })
    }
}
