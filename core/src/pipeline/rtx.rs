use std::{alloc::Layout, sync::Arc};

use ash::{prelude::VkResult, vk};

use crate::{HasDevice, PipelineLayout};

pub struct RayTracingPipeline {
    layout: Arc<PipelineLayout>,
    pipeline: vk::Pipeline,
}
impl HasDevice for RayTracingPipeline {
    fn device(&self) -> &Arc<crate::Device> {
        self.layout.device()
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
