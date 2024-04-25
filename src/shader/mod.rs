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
    pub fn raw_specialization_info<'a>(&'a self) -> vk::SpecializationInfo<'a> {
        vk::SpecializationInfo::default()
            .map_entries(&self.specialization_info.entries)
            .data(&self.specialization_info.data)
    }
}
#[derive(Clone, Default, Debug)]
pub struct SpecializationInfo {
    pub(super) data: Vec<u8>,
    pub(super) entries: Vec<vk::SpecializationMapEntry>,
}
impl SpecializationInfo {
    pub fn raw_info<'a>(&'a self) -> vk::SpecializationInfo<'a> {
        vk::SpecializationInfo::default()
            .map_entries(&self.entries)
            .data(&self.data)
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
