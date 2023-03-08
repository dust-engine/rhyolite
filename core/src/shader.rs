use ash::prelude::VkResult;
use ash::vk;
use std::ffi::CStr;
use std::ops::Deref;
use std::{collections::HashMap, sync::Arc};

use crate::descriptor::{DescriptorSetLayout, DescriptorSetLayoutCache};
use crate::sampler::Sampler;
use crate::{device, Device, HasDevice};
pub struct SpirvShader<T: Deref<Target = [u32]>> {
    pub data: T,
    pub entry_points: HashMap<String, SpirvEntryPoint>,
}

pub use crate::descriptor::DescriptorSetLayoutBindingInfo as SpirvDescriptorSetBinding;
pub use crate::descriptor::DescriptorSetLayoutCacheKey as SpirvDescriptorSet;
#[derive(Debug)]
pub struct SpirvEntryPoint {
    pub stage: vk::ShaderStageFlags,
    pub descriptor_sets: Vec<SpirvDescriptorSet>,
    pub push_constant_range: Option<vk::PushConstantRange>,
}

impl<T: Deref<Target = [u32]>> SpirvShader<T> {
    pub fn set_flags(
        &mut self,
        entry_point: &str,
        set_id: u32,
        flags: vk::DescriptorSetLayoutCreateFlags,
    ) {
        self.entry_points
            .get_mut(entry_point)
            .unwrap()
            .descriptor_sets[set_id as usize]
            .flags = flags;
    }
    pub fn add_immutable_samplers(
        &mut self,
        entry_point: &str,
        set_id: u32,
        binding_id: u32,
        samplers: Vec<Arc<Sampler>>,
    ) {
        let binding = self
            .entry_points
            .get_mut(entry_point)
            .expect("Entry point not found")
            .descriptor_sets
            .get_mut(set_id as usize)
            .expect("Set not found")
            .bindings
            .iter_mut()
            .find(|binding| binding.binding == binding_id)
            .expect("Binding not found");
        assert!(
            binding.immutable_samplers.is_empty(),
            "Immutable samplers already added"
        );
        assert!(binding.descriptor_count == samplers.len() as u32);
        assert!(
            binding.descriptor_type == vk::DescriptorType::SAMPLER
                || binding.descriptor_type == vk::DescriptorType::COMBINED_IMAGE_SAMPLER
        );
        binding.immutable_samplers = samplers;
    }
    pub fn build(self, cache: &mut DescriptorSetLayoutCache) -> VkResult<ShaderModule> {
        let module = unsafe {
            cache.device().create_shader_module(
                &vk::ShaderModuleCreateInfo {
                    code_size: std::mem::size_of_val(self.data.as_ref()),
                    p_code: self.data.as_ref().as_ptr(),
                    ..Default::default()
                },
                None,
            )
        }?;
        let entry_points = self
            .entry_points
            .into_iter()
            .map(|(name, entry_point)| {
                (
                    name.clone(),
                    ShaderModuleEntryPoint {
                        stage: entry_point.stage,
                        desc_sets: entry_point
                            .descriptor_sets
                            .into_iter()
                            .map(|desc_set| cache.get(desc_set.bindings, Default::default()))
                            .collect(),
                        push_constant_range: entry_point.push_constant_range.clone(),
                    },
                )
            })
            .collect();
        Ok(ShaderModule {
            device: cache.device().clone(),
            module,
            entry_points,
        })
    }
}

pub struct ShaderModule {
    device: Arc<Device>,
    module: vk::ShaderModule,
    pub entry_points: HashMap<String, ShaderModuleEntryPoint>,
}
impl ShaderModule {
    pub unsafe fn raw(&self) -> vk::ShaderModule {
        self.module
    }
    pub fn specialized<'a>(&'a self, entry_point: &'a CStr) -> SpecializedShader {
        SpecializedShader {
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            shader: self,
            specialization_info: Default::default(),
            entry_point,
        }
    }
}
impl HasDevice for ShaderModule {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

#[derive(Clone)]
pub struct ShaderModuleEntryPoint {
    pub stage: vk::ShaderStageFlags,
    pub desc_sets: Vec<Arc<DescriptorSetLayout>>,
    pub push_constant_range: Option<vk::PushConstantRange>,
}
impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.module, None);
        }
    }
}

#[derive(Clone, Default, Debug)]
pub struct SpecializationInfo {
    pub(super) data: Vec<u8>,
    pub(super) entries: Vec<vk::SpecializationMapEntry>,
}
impl SpecializationInfo {
    pub unsafe fn raw_info(&self) -> vk::SpecializationInfo {
        vk::SpecializationInfo {
            map_entry_count: self.entries.len() as u32,
            p_map_entries: self.entries.as_ptr(),
            data_size: self.data.len(),
            p_data: self.data.as_ptr() as *const _,
        }
    }
    pub const fn new() -> Self {
        Self {
            data: Vec::new(),
            entries: Vec::new(),
        }
    }
    pub fn push<T: Copy + 'static>(&mut self, constant_id: u32, item: T) {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<bool>() {
            panic!("Use push_bool")
        }
        let size = std::mem::size_of::<T>();
        self.entries.push(vk::SpecializationMapEntry {
            constant_id,
            offset: self.data.len() as u32,
            size,
        });
        self.data.reserve(size);
        unsafe {
            let target_ptr = self.data.as_mut_ptr().add(self.data.len());
            std::ptr::copy_nonoverlapping(&item as *const T as *const u8, target_ptr, size);
            self.data.set_len(self.data.len() + size);
        }
    }
    pub fn push_bool(&mut self, constant_id: u32, item: bool) {
        let size = std::mem::size_of::<vk::Bool32>();
        self.entries.push(vk::SpecializationMapEntry {
            constant_id,
            offset: self.data.len() as u32,
            size,
        });
        self.data.reserve(size);
        unsafe {
            let item: vk::Bool32 = if item { vk::TRUE } else { vk::FALSE };
            let target_ptr = self.data.as_mut_ptr().add(self.data.len());
            std::ptr::copy_nonoverlapping(
                &item as *const vk::Bool32 as *const u8,
                target_ptr,
                size,
            );
            self.data.set_len(self.data.len() + size);
        }
    }
}

#[derive(Clone)]
pub struct SpecializedShader<'a> {
    pub flags: vk::PipelineShaderStageCreateFlags,
    pub shader: &'a ShaderModule,
    pub specialization_info: SpecializationInfo,
    pub entry_point: &'a CStr,
}
impl<'a> SpecializedShader<'a> {
    pub fn stage(&self) -> vk::ShaderStageFlags {
        todo!()
    }
    pub fn entry_point(&self) -> &ShaderModuleEntryPoint {
        self.shader
            .entry_points
            .get(self.entry_point.to_str().unwrap())
            .unwrap()
    }
}
impl<'a> HasDevice for SpecializedShader<'a> {
    fn device(&self) -> &Arc<Device> {
        &self.shader.device
    }
}
