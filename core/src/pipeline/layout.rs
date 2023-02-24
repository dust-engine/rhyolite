use crate::{shader::ShaderModuleEntryPoint, Device, HasDevice, ShaderModule};
use ash::{prelude::VkResult, vk};
use std::sync::Arc;

pub struct PipelineLayout {
    device: Arc<Device>,
    inner: vk::PipelineLayout,
    pub info: ShaderModuleEntryPoint,
}

impl PipelineLayout {
    pub fn for_layout(
        device: Arc<Device>,
        entry_point: ShaderModuleEntryPoint,
        flags: vk::PipelineLayoutCreateFlags,
    ) -> VkResult<Self> {
        let set_layouts: Vec<_> = entry_point
            .desc_sets
            .iter()
            .map(|a| unsafe { a.raw() })
            .collect();
        let info = vk::PipelineLayoutCreateInfo {
            flags,
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: entry_point.push_constant_ranges.len() as u32,
            p_push_constant_ranges: entry_point.push_constant_ranges.as_ptr(),
            ..Default::default()
        };
        let layout = unsafe { device.create_pipeline_layout(&info, None)? };
        Ok(Self {
            device,
            inner: layout,
            info: entry_point,
        })
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
