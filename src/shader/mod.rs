use std::{ffi::CStr, sync::Arc};

use ash::prelude::VkResult;
use bevy::asset::{Asset, AssetLoader, Handle, AsyncReadExt};
use bevy::reflect::TypePath;
use crate::Device;
use ash::vk;
use cstr::cstr;
use thiserror::Error;


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
impl SpecializedShader {
    pub fn with_const<T: Copy + 'static>(mut self, constant_id: u32, item: T) -> Self {
        self.specialization_info.push(constant_id, item);
        self
    }
    pub fn as_raw(&self) -> vk::SpecializationInfo {
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
}

#[derive(Debug, Error)]
pub enum SpirvLoaderError {
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("vulkan error: {0:?}")]
    VkError(#[from] vk::Result),
}
pub struct SpirvLoader {
    device: Device,
}
impl SpirvLoader {
    pub(crate) fn new(device: Device) -> Self {
        Self { device }
    }
}
impl AssetLoader for SpirvLoader {
    type Asset = ShaderModule;
    type Settings = ();
    type Error = SpirvLoaderError;
    fn load<'a>(
        &'a self,
        reader: &'a mut bevy::asset::io::Reader,
        _settings: &'a Self::Settings,
        _load_context: &'a mut bevy::asset::LoadContext,
    ) -> bevy::utils::BoxedFuture<'a, Result<ShaderModule, Self::Error>> {
        let device = self.device.clone();
        return Box::pin(async move {
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes).await?;
            assert!(bytes.len() % 4 == 0);
            let bytes = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4)
            };
            let shader = ShaderModule::new(device, bytes)?;
            Ok(shader)
        });
    }

    fn extensions(&self) -> &[&str] {
        &["spv"]
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
