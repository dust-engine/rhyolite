use std::{any::Any, os::raw::c_void, sync::Arc};

use ash::vk;
use bevy::asset::Assets;

use crate::{deferred::{DeferredOperationTaskPool, Task}, utils::SendBox, Device, ShaderModule, SpecializedShader};

use super::PipelineLayout;

pub struct GraphicsPipeline {
    device: Device,
    pipeline: vk::Pipeline,
}

impl super::Pipeline for GraphicsPipeline {
    type BuildInfo = GraphicsPipelineBuildInfo;
}


pub struct GraphicsPipelineBuildInfo {
    pub device: Device,
    pub flags: vk::PipelineCreateFlags,
    pub stages: Vec<SpecializedShader>,
    pub vertex_input_state: Option<Box<vk::PipelineVertexInputStateCreateInfo>>,
    pub input_assembly_state: Option<Box<vk::PipelineInputAssemblyStateCreateInfo>>,
    pub tessellation_state: Option<Box<vk::PipelineTessellationStateCreateInfo>>,
    pub viewport_state: Option<Box<vk::PipelineViewportStateCreateInfo>>,
    pub rasterization_state: Option<Box<vk::PipelineRasterizationStateCreateInfo>>,
    pub multisample_state: Option<Box<vk::PipelineMultisampleStateCreateInfo>>,
    pub depth_stencil_state: Option<Box<vk::PipelineDepthStencilStateCreateInfo>>,
    pub color_blend_state: Option<Box<vk::PipelineColorBlendStateCreateInfo>>,
    pub dynamic_state: Option<Box<vk::PipelineDynamicStateCreateInfo>>,
    pub layout: Arc<PipelineLayout>,
    p_next: *const std::ffi::c_void,
    _extensions: Vec<Box<dyn Any + Send + Sync>>
}
impl GraphicsPipelineBuildInfo {
    pub fn with_dynamic_rendering(&mut self, mut info: vk::PipelineRenderingCreateInfo) -> &mut Self {
        assert!(info.p_next.is_null());
        info.p_next = self.p_next;
        let boxed: Box<dyn Any + Send + Sync> = Box::new(unsafe{SendBox::new( info)});
        self.p_next = boxed.downcast_ref::<SendBox<vk::PipelineRenderingCreateInfo>>().unwrap() as *const SendBox<vk::PipelineRenderingCreateInfo> as *const c_void;
        self._extensions.push(boxed);
        self
    }
}

impl super::PipelineBuildInfo for GraphicsPipelineBuildInfo {
    type Pipeline = GraphicsPipeline;
    
    fn build(
        &mut self,
        pool: &DeferredOperationTaskPool,
        assets: &Assets<ShaderModule>,
        cache: vk::PipelineCache,
    ) -> Option<Task<Self::Pipeline>> {
        let device = self.device.clone();
        let specialization_info = self.stages.iter().map(SpecializedShader::as_raw).collect::<Vec<_>>().into_boxed_slice();
        let stages = self.stages.iter().zip(specialization_info.iter()).map(|(shader, specialization_info)| {
            let module = assets.get(&shader.shader)?;
            Some(vk::PipelineShaderStageCreateInfo {
                stage: shader.stage,
                module: module.raw(),
                p_name: shader.entry_point.as_ptr(),
                p_specialization_info: specialization_info,
                flags: shader.flags,
                ..Default::default()
            })
        }).collect::<Option<Vec<_>>>()?.into_boxed_slice();
        let layout = self.layout.clone();
        let create_info = Box::new(vk::GraphicsPipelineCreateInfo {
            flags: self.flags,
            stage_count: stages.len() as u32,
            p_stages: stages.as_ptr() as *const vk::PipelineShaderStageCreateInfo,
            p_vertex_input_state: self.vertex_input_state.as_ref().map(|x| x.as_ref() as *const _).unwrap_or_else(|| std::ptr::null()),
            p_input_assembly_state: self.input_assembly_state.as_ref().map(|x| x.as_ref() as *const _).unwrap_or_else(|| std::ptr::null()),
            p_tessellation_state: self.tessellation_state.as_ref().map(|x| x.as_ref() as *const _).unwrap_or_else(|| std::ptr::null()),
            p_viewport_state: self.viewport_state.as_ref().map(|x| x.as_ref() as *const _).unwrap_or_else(|| std::ptr::null()),
            p_rasterization_state: self.rasterization_state.as_ref().map(|x| x.as_ref() as *const _).unwrap_or_else(|| std::ptr::null()),
            p_multisample_state: self.multisample_state.as_ref().map(|x| x.as_ref() as *const _).unwrap_or_else(|| std::ptr::null()),
            p_depth_stencil_state: self.depth_stencil_state.as_ref().map(|x| x.as_ref() as *const _).unwrap_or_else(|| std::ptr::null()),
            p_color_blend_state: self.color_blend_state.as_ref().map(|x| x.as_ref() as *const _).unwrap_or_else(|| std::ptr::null()),
            p_dynamic_state: self.dynamic_state.as_ref().map(|x| x.as_ref() as *const _).unwrap_or_else(|| std::ptr::null()),
            layout: layout.raw(),
            render_pass: vk::RenderPass::null(),
            subpass: 0,
            base_pipeline_index: -1,
            p_next: self.p_next,
            ..Default::default()
        });
        let create_info = unsafe { SendBox::new(create_info) };
        let stages = unsafe { SendBox::new(stages) };
        let specialization_info = unsafe { SendBox::new(specialization_info) };
        Some(pool.schedule(move || unsafe {
            let create_info = create_info.into_inner();
            let mut pipeline = vk::Pipeline::null();
            let result = (device.fp_v1_0().create_graphics_pipelines)(device.handle(), cache, 1, create_info.as_ref() as *const _, std::ptr::null(), &mut pipeline);
            drop(create_info);
            drop(stages);
            drop(specialization_info);
            drop(layout);
            result.result_with_success(GraphicsPipeline {
                device,
                pipeline
            })
        }))
    }
}
