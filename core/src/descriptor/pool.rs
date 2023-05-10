use ash::{prelude::VkResult, vk};
use std::{collections::BTreeMap, sync::Arc};

use crate::{Device, HasDevice, PipelineLayout};

pub struct DescriptorPool {
    device: Arc<Device>,
    pool: vk::DescriptorPool,
}
impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}
impl HasDevice for DescriptorPool {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl DescriptorPool {
    pub fn allocate_for_pipeline_layout(
        &mut self,
        pipeline: &PipelineLayout,
    ) -> VkResult<Vec<vk::DescriptorSet>> {
        let set_layouts: Vec<_> = pipeline
            .desc_sets()
            .iter()
            .map(|a| unsafe { a.raw() })
            .collect();
        let info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.pool,
            descriptor_set_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            ..Default::default()
        };
        unsafe { self.device.allocate_descriptor_sets(&info) }
    }
    /// Create a descriptor pool just big enough to accommodate one of each pipeline layout,
    /// for `multiplier` times. This is useful when you have multiple pipeline layouts, each
    /// having distinct descriptor layouts and bindings. `multiplier` would generally match the
    /// max number of frames in flight.
    pub fn for_pipeline_layouts<T: std::ops::Deref<Target = PipelineLayout>>(
        layouts: impl IntoIterator<Item = T>,
        multiplier: u32,
    ) -> VkResult<Self> {
        let mut desc_types: BTreeMap<vk::DescriptorType, u32> = BTreeMap::new();
        let mut max_sets: u32 = 0;
        let mut device: Option<Arc<Device>> = None;
        for pipeline_layout in layouts.into_iter() {
            let pipeline_layout = pipeline_layout.deref();
            max_sets += pipeline_layout.desc_sets().len() as u32;
            if let Some(device) = device.as_ref() {
                assert!(Arc::ptr_eq(device, pipeline_layout.device()));
            } else {
                device.replace(pipeline_layout.device().clone());
            }
            for desc_set_layout in pipeline_layout.desc_sets().iter() {
                for binding in desc_set_layout.binding_infos.iter() {
                    if binding.immutable_samplers.is_empty() {
                        let count = desc_types.entry(binding.descriptor_type).or_insert(0);
                        *count += binding.descriptor_count;
                    } else {
                        // Don't need separate descriptor if the sampler was built into the layout.
                        assert_eq!(
                            binding.immutable_samplers.len() as u32,
                            binding.descriptor_count
                        );
                    }
                }
            }
        }
        let pool_sizes: Vec<_> = desc_types
            .into_iter()
            .map(|(ty, descriptor_count)| vk::DescriptorPoolSize {
                ty,
                descriptor_count: descriptor_count * multiplier,
            })
            .collect();
        let info = vk::DescriptorPoolCreateInfo {
            max_sets: max_sets * multiplier,
            p_pool_sizes: pool_sizes.as_ptr(),
            pool_size_count: pool_sizes.len() as u32,
            ..Default::default()
        };
        let device = device.expect("Expects at least one pipeline layout.");
        let pool = unsafe { device.create_descriptor_pool(&info, None)? };
        Ok(Self { device, pool })
    }
}
