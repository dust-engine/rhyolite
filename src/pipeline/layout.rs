use std::{collections::BTreeMap, sync::Arc};

use crate::{Device, HasDevice};
use ash::{prelude::VkResult, vk};

pub struct DescriptorSetLayout {
    device: Device,
    pub(crate) raw: vk::DescriptorSetLayout,
    pub(crate) desc_types: Vec<(vk::DescriptorType, u32)>,
}
impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_set_layout(self.raw, None);
        }
    }
}
impl DescriptorSetLayout {
    /// Users should obtain the layout from the cache.
    /// TODO: Actually cache this, or not.
    pub fn new(
        device: Device,
        binding_infos: &[vk::DescriptorSetLayoutBinding],
        flags: vk::DescriptorSetLayoutCreateFlags,
    ) -> VkResult<Self> {
        let raw = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo {
                    flags,
                    binding_count: binding_infos.len() as u32,
                    p_bindings: binding_infos.as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }?;

        let mut desc_types = BTreeMap::new();

        for binding in binding_infos.iter() {
            if binding.p_immutable_samplers.is_null() {
                let count = desc_types.entry(binding.descriptor_type).or_insert(0);
                if binding.descriptor_type == vk::DescriptorType::INLINE_UNIFORM_BLOCK {
                    // We take the next multiple of 8 here because on AMD, descriptor pool allocations seem
                    // to be aligned to the 8 byte boundary. See
                    // https://gist.github.com/Neo-Zhixing/992a0e789e34b59285026dd8161b9112
                    *count += binding.descriptor_count.next_multiple_of(8);
                } else {
                    *count += binding.descriptor_count;
                }
            } else {
                if binding.descriptor_type == vk::DescriptorType::COMBINED_IMAGE_SAMPLER {
                    let count = desc_types.entry(binding.descriptor_type).or_insert(0);
                    *count += binding.descriptor_count;
                } else {
                    // Don't need separate descriptor if the sampler was built into the layout
                    assert_eq!(binding.descriptor_type, vk::DescriptorType::SAMPLER);
                }
            }
        }

        Ok(Self {
            device,
            raw,
            desc_types: desc_types.into_iter().collect(),
        })
    }
    pub unsafe fn raw(&self) -> vk::DescriptorSetLayout {
        self.raw
    }
}

pub struct PipelineLayout {
    device: Device,
    inner: vk::PipelineLayout,

    desc_sets: Vec<Arc<DescriptorSetLayout>>,
    push_constant_range: Vec<vk::PushConstantRange>,
}

impl PipelineLayout {
    pub fn desc_sets(&self) -> &[Arc<DescriptorSetLayout>] {
        &self.desc_sets
    }
    pub fn push_constant_range(&self) -> &[vk::PushConstantRange] {
        &self.push_constant_range
    }
    pub fn new(
        device: Device,
        set_layouts: Vec<Arc<DescriptorSetLayout>>,
        push_constant_ranges: &[vk::PushConstantRange],
        flags: vk::PipelineLayoutCreateFlags,
    ) -> VkResult<Self> {
        let raw_set_layouts: Vec<_> = set_layouts.iter().map(|a| unsafe { a.raw() }).collect();
        let info = vk::PipelineLayoutCreateInfo {
            flags,
            set_layout_count: raw_set_layouts.len() as u32,
            p_set_layouts: raw_set_layouts.as_ptr(),
            push_constant_range_count: push_constant_ranges.len() as u32,
            p_push_constant_ranges: push_constant_ranges.as_ptr(),
            ..Default::default()
        };

        let layout = unsafe { device.create_pipeline_layout(&info, None)? };
        Ok(Self {
            device,
            inner: layout,
            desc_sets: set_layouts,
            push_constant_range: Vec::new(),
        })
    }
    pub fn raw(&self) -> vk::PipelineLayout {
        self.inner
    }
}
impl HasDevice for PipelineLayout {
    fn device(&self) -> &Device {
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
