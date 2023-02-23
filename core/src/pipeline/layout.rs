use std::sync::Arc;
use crate::{Device, ShaderModule, HasDevice};
use ash::{vk, prelude::VkResult};

pub struct PipelineLayout {
    device: Arc<Device>,
    inner: vk::PipelineLayout
}

impl PipelineLayout {
    pub fn new(device: Arc<Device>, info: &vk::PipelineLayoutCreateInfo) -> VkResult<Self> {
        let layout = unsafe {
            device.create_pipeline_layout(info, None)?
        };
        Ok(Self {
            device,
            inner: layout
        })
    }
    pub fn new_from_module(module: &ShaderModule, entry_point: &str, flags: vk::PipelineLayoutCreateFlags) -> VkResult<Self> {
        let entry_point = module.entry_points.get(entry_point).unwrap();
        let info = vk::PipelineLayoutCreateInfo {
            flags,
            set_layout_count: entry_point.desc_sets.len() as u32,
            p_set_layouts: entry_point.desc_sets.as_ptr(),
            push_constant_range_count: entry_point.push_constant_ranges.len() as u32,
            p_push_constant_ranges: entry_point.push_constant_ranges.as_ptr(),
            ..Default::default()
        };
        Self::new(
            module.device().clone(),
            &info,
        )
    }
    pub unsafe fn raw(&self) -> vk::PipelineLayout {
        self.inner
    }
}
impl HasDevice for PipelineLayout {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}
impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline_layout(self.inner, None);
        }
    }
}

