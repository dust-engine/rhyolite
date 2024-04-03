use std::ffi::CStr;

use crate::Device;
use ash::prelude::VkResult;
use ash::vk;
use bevy::asset::{Asset, Handle};
use bevy::reflect::TypePath;
use cstr::cstr;

#[cfg(feature = "glsl")]
mod glsl;
mod spirv;
pub mod loader {
    #[cfg(feature = "glsl")]
    pub use super::glsl::*;
    pub use super::spirv::*;
}

#[derive(TypePath, Asset)]
pub struct ShaderModule {
    device: Device,
    module: vk::ShaderModule,
}
impl ShaderModule {
    pub fn new(device: Device, code: &[u32]) -> VkResult<Self> {
        let module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo {
                    p_code: code.as_ptr(),
                    code_size: std::mem::size_of_val(code),
                    ..Default::default()
                },
                None,
            )?
        };
        Ok(Self { device, module })
    }
    pub fn raw(&self) -> vk::ShaderModule {
        self.module
    }
}
impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.module, None);
        }
    }
}
#[derive(Clone)]
pub struct SpecializedShader {
    pub stage: vk::ShaderStageFlags,
    pub flags: vk::PipelineShaderStageCreateFlags,
    pub shader: Handle<ShaderModule>,
    pub specialization_info: SpecializationInfo,
    pub entry_point: &'static CStr,
}
impl Default for SpecializedShader {
    fn default() -> Self {
        Self {
            stage: vk::ShaderStageFlags::empty(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            shader: Handle::default(),
            specialization_info: SpecializationInfo::new(),
            entry_point: cstr!("main"),
        }
    }
}
impl SpecializedShader {
    pub fn with_const<T: Copy + 'static>(mut self, constant_id: u32, item: T) -> Self {
        self.specialization_info.push(constant_id, item);
        self
    }
    pub fn raw_pipeline_stage(
        &self,
        assets: &bevy::asset::Assets<ShaderModule>,
        specialization_info: &vk::SpecializationInfo,
    ) -> Option<vk::PipelineShaderStageCreateInfo> {
        let module = assets.get(&self.shader)?;
        Some(vk::PipelineShaderStageCreateInfo {
            stage: self.stage,
            module: module.raw(),
            p_name: self.entry_point.as_ptr(),
            p_specialization_info: specialization_info,
            flags: self.flags,
            ..Default::default()
        })
    }
    pub fn raw_specialization_info(&self) -> vk::SpecializationInfo {
        vk::SpecializationInfo {
            map_entry_count: self.specialization_info.entries.len() as u32,
            p_map_entries: if self.specialization_info.entries.is_empty() {
                std::ptr::null()
            } else {
                self.specialization_info.entries.as_ptr()
            },
            data_size: self.specialization_info.data.len(),
            p_data: if self.specialization_info.data.is_empty() {
                std::ptr::null()
            } else {
                self.specialization_info.data.as_ptr() as *const _
            },
        }
    }

    pub fn as_raw_many(
        stages: &[Self],
        assets: &bevy::asset::Assets<ShaderModule>,
    ) -> Option<(
        Box<[vk::SpecializationInfo]>,
        Box<[vk::PipelineShaderStageCreateInfo]>,
    )> {
        let specialization_info = stages
            .iter()
            .map(SpecializedShader::raw_specialization_info)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let stages = stages
            .iter()
            .zip(specialization_info.iter())
            .map(|(shader, specialization_info)| {
                let module = assets.get(&shader.shader)?;
                shader.raw_pipeline_stage(assets, specialization_info)
            })
            .collect::<Option<Vec<_>>>()?
            .into_boxed_slice();
        Some((specialization_info, stages))
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
            p_map_entries: if self.entries.is_empty() {
                std::ptr::null()
            } else {
                self.entries.as_ptr()
            },
            data_size: self.data.len(),
            p_data: if self.data.is_empty() {
                std::ptr::null()
            } else {
                self.data.as_ptr() as *const _
            },
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
            unsafe {
                let value: bool = std::mem::transmute_copy(&item);
                self.push_bool(constant_id, value);
                return;
            }
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
    fn push_bool(&mut self, constant_id: u32, item: bool) {
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
