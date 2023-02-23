use std::{collections::HashMap, sync::Arc};
use ash::prelude::VkResult;
use ash::vk;
use std::ops::Deref;

use crate::sampler::Sampler;
use crate::{Device, HasDevice, device};
pub struct SpirvShader<T: Deref<Target = [u32]>> {
    pub data: T,
    pub entry_points: HashMap<String, SpirvEntryPoint>
}

#[derive(Debug)]
pub struct SpirvDescriptorSetBinding {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
    pub stage_flags: vk::ShaderStageFlags,
    pub immutable_samplers: Vec<Arc<Sampler>>,
}
#[derive(Debug)]
pub struct SpirvDescriptorSet {
    pub bindings: Vec<SpirvDescriptorSetBinding>
}

#[derive(Debug)]
pub struct SpirvEntryPoint {
    pub descriptor_sets: Vec<SpirvDescriptorSet>,
    pub push_constant_ranges: Vec<vk::PushConstantRange>,
}

impl<T: Deref<Target = [u32]>> SpirvShader<T> {
    pub fn add_immutable_samplers(&mut self, entry_point: &str, set_id: u32, binding_id: u32, samplers: Vec<Arc<Sampler>>) {
        let binding = self.entry_points
        .get_mut(entry_point)
        .expect("Entry point not found")
        .descriptor_sets
        .get_mut(set_id as usize)
        .expect("Set not found")
        .bindings
        .iter_mut()
        .find(|binding| binding.binding == binding_id)
        .expect("Binding not found");
        assert!(binding.immutable_samplers.is_empty(), "Immutable samplers already added");
        assert!(binding.descriptor_count == samplers.len() as u32);
        assert!(binding.descriptor_type == vk::DescriptorType::SAMPLER || binding.descriptor_type == vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
        binding.immutable_samplers = samplers;
    }
    pub fn build(&self, device: Arc<Device>) -> VkResult<ShaderModule> {
        let module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo {
                    code_size: std::mem::size_of_val(self.data.as_ref()),
                    p_code: self.data.as_ref().as_ptr(),
                    ..Default::default()
                },
                None
            )
        }?;            
        let mut referenced_immutable_samplers: Vec<Arc<Sampler>> = Vec::new();
        let entry_points = self.entry_points.iter().map(|(name, entry_point)| {

            (name.clone(), ShaderModuleEntryPoint {
                desc_sets: entry_point.descriptor_sets.iter().map(|desc_set| unsafe {
                    let total_immutable_samplers = desc_set.bindings.iter().map(|a| a.immutable_samplers.len()).sum();
                    let mut immutable_samplers: Vec<vk::Sampler> = Vec::with_capacity(total_immutable_samplers);
                    let bindings: Vec<_> = desc_set.bindings.iter().map(|binding| {
                        let immutable_samplers_offset = immutable_samplers.len();
                        immutable_samplers.extend(binding.immutable_samplers.iter().map(|a| a.raw()));
                        referenced_immutable_samplers.extend(binding.immutable_samplers.iter().map(|a| a.clone()));
                        if binding.immutable_samplers.len() > 0 {
                            assert_eq!(binding.immutable_samplers.len() as u32, binding.descriptor_count);
                        }
                        vk::DescriptorSetLayoutBinding {
                            binding: binding.binding,
                            descriptor_type: binding.descriptor_type,
                            descriptor_count: binding.descriptor_count,
                            stage_flags: binding.stage_flags,
                            p_immutable_samplers: if binding.immutable_samplers.is_empty() {
                                std::ptr::null()
                            } else {
                                immutable_samplers.as_ptr().add(immutable_samplers_offset)
                            },
                        }
                    }).collect();
                    device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo {
                        flags: vk::DescriptorSetLayoutCreateFlags::empty(),
                        binding_count: bindings.len() as u32,
                        p_bindings: bindings.as_ptr(),
                        ..Default::default()
                    }, None).unwrap()
                }).collect(),
                push_constant_ranges: entry_point.push_constant_ranges.clone()
            })
        }).collect();
        Ok(ShaderModule {
            device,
            module,
            entry_points,
            _referenced_immutable_samplers: referenced_immutable_samplers
        })
    }
}

pub struct ShaderModule {
    device: Arc<Device>,
    module: vk::ShaderModule,
    pub(crate) entry_points: HashMap<String, ShaderModuleEntryPoint>,
    _referenced_immutable_samplers: Vec<Arc<Sampler>>
}
impl ShaderModule {
    pub unsafe fn raw(&self) -> vk::ShaderModule {
        self.module
    }
}
impl HasDevice for ShaderModule {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}
pub(crate) struct ShaderModuleEntryPoint {
    pub desc_sets: Vec<vk::DescriptorSetLayout>,
    pub push_constant_ranges: Vec<vk::PushConstantRange>
}
impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            for entry_point in self.entry_points.values() {
                for layout in entry_point.desc_sets.iter() {
                    self.device.destroy_descriptor_set_layout(*layout, None);
                }
            }
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
    pub fn new() -> Self {
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