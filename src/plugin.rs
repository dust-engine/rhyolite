use ash::{prelude::VkResult, vk};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::{c_char, CStr, CString},
    fmt::Display,
    sync::Arc,
};
use thiserror::Error;

use crate::{Device, Instance, Version};
use cstr::cstr;

pub struct LayerProperties {
    pub spec_version: Version,
    pub implementation_version: Version,
    pub description: String,
}

pub struct RhyolitePlugin {
    pub application_name: CString,
    pub application_version: Version,
    pub engine_name: CString,
    pub engine_version: Version,
    pub api_version: Version,

    pub physical_device_index: usize,

    enabled_instance_extensions: Vec<*const c_char>,
    enabled_instance_layers: Vec<*const c_char>,

    entry: Arc<ash::Entry>,
    available_layers: BTreeMap<CString, LayerProperties>,
    available_extensions: BTreeMap<CString, Version>,
}
unsafe impl Send for RhyolitePlugin {}
unsafe impl Sync for RhyolitePlugin {}
impl RhyolitePlugin {
    pub fn with_instance_extension(&mut self, extension: &'static CStr) -> Option<Version> {
        if let Some(v) = self.available_extensions.get(extension) {
            self.enabled_instance_extensions.push(extension.as_ptr());
            Some(*v)
        } else {
            None
        }
    }
    pub fn with_instance_extension_for_layer(
        &mut self,
        extension: &'static CStr,
        layer: &'static CStr,
    ) -> Option<Version> {
        let all_extensions = self
            .entry
            .enumerate_instance_extension_properties(Some(layer))
            .unwrap();
        let ext = all_extensions.into_iter().find(|ext| unsafe {
            let ext_name = CStr::from_bytes_until_nul(std::slice::from_raw_parts(
                ext.extension_name.as_ptr() as *const u8,
                ext.extension_name.len(),
            ))
            .unwrap();
            ext_name == extension
        });
        if let Some(ext) = ext {
            self.enabled_instance_extensions.push(extension.as_ptr());
            Some(Version(ext.spec_version))
        } else {
            None
        }
    }
    pub fn with_instance_layer(&mut self, layer: &'static CStr) -> Option<&LayerProperties> {
        if let Some(layer_properties) = self.available_layers.get(layer) {
            self.enabled_instance_layers.push(layer.as_ptr());
            Some(layer_properties)
        } else {
            None
        }
    }
}

#[derive(Error, Debug)]
pub enum RhyoliltePluginInitError {
    #[error("Ash loading failed")]
    LoadingError(#[from] ash::LoadingError),
    #[error("Vulkan Implementation error: Invalid C String")]
    ImplementationError(#[from] std::ffi::FromBytesUntilNulError),
    #[error("Vulkan Implementation error: Invalid UTF-8 String")]
    ImplementationErrorUtf8(#[from] std::str::Utf8Error),
    #[error("Vulkan error: {0}")]
    VulkanError(#[from] vk::Result),
}

impl RhyolitePlugin {
    fn new() -> Result<Self, RhyoliltePluginInitError> {
        let entry = unsafe { ash::Entry::load().unwrap() };
        let entry = Arc::new(entry);
        let available_layers = entry
            .enumerate_instance_layer_properties()?
            .into_iter()
            .map(
                |layer| -> Result<(CString, LayerProperties), RhyoliltePluginInitError> {
                    let str = CStr::from_bytes_until_nul(unsafe {
                        std::slice::from_raw_parts(
                            layer.layer_name.as_ptr() as *const u8,
                            layer.layer_name.len(),
                        )
                    })?;
                    Ok((
                        str.to_owned(),
                        LayerProperties {
                            implementation_version: Version(layer.implementation_version),
                            spec_version: Version(layer.spec_version),
                            description: CStr::from_bytes_until_nul(unsafe {
                                std::slice::from_raw_parts(
                                    layer.description.as_ptr() as *const u8,
                                    layer.description.len(),
                                )
                            })?
                            .to_str()?
                            .to_string(),
                        },
                    ))
                },
            )
            .collect::<Result<BTreeMap<CString, LayerProperties>, RhyoliltePluginInitError>>()?;

        let available_extensions = entry
            .enumerate_instance_extension_properties(None)?
            .into_iter()
            .map(
                |ext| -> Result<(CString, Version), RhyoliltePluginInitError> {
                    let str = CStr::from_bytes_until_nul(unsafe {
                        std::slice::from_raw_parts(
                            ext.extension_name.as_ptr() as *const u8,
                            ext.extension_name.len(),
                        )
                    })?;
                    Ok((str.to_owned(), Version(ext.spec_version)))
                },
            )
            .collect::<Result<BTreeMap<CString, Version>, RhyoliltePluginInitError>>()?;
        Ok(Self {
            application_name: cstr!(b"Unnamed Application").to_owned(),
            application_version: Default::default(),
            engine_name: cstr!(b"Unnamed Engine").to_owned(),
            engine_version: Default::default(),
            api_version: Version::new(0, 1, 3, 0),
            physical_device_index: 0,
            enabled_instance_extensions: Vec::new(),
            enabled_instance_layers: Vec::new(),
            entry,
            available_layers,
            available_extensions,
        })
    }
}

#[derive(Default, Resource)]
struct ExtensionSettings {
    available_extensions: BTreeMap<CString, Version>,
    device_extensions: Vec<*const c_char>,
}
unsafe impl Send for ExtensionSettings {}
unsafe impl Sync for ExtensionSettings {}

impl Plugin for RhyolitePlugin {
    fn build(&self, app: &mut App) {
        let settings = ExtensionSettings::default();
        let instance = Instance::create(
            self.entry.clone(),
            &crate::InstanceCreateInfo {
                enabled_extension_names: &self.enabled_instance_extensions,
                enabled_layer_names: &self.enabled_instance_layers,
                api_version: self.api_version.clone(),
                engine_name: self.engine_name.as_c_str(),
                engine_version: self.engine_version,
                application_name: self.application_name.as_c_str(),
                application_version: self.application_version,
            },
        )
        .unwrap();

        app.insert_resource(settings).insert_resource(instance);
    }
    fn cleanup(&self, app: &mut App) {
        app.world.remove_resource::<ExtensionSettings>();
    }
    fn finish(&self, app: &mut App) {
        let instance: &Instance = app.world.resource();
        let physical_device = crate::PhysicalDevice::enumerate(&instance)
            .unwrap()
            .into_iter()
            .skip(self.physical_device_index)
            .next()
            .unwrap();
        tracing::info!(
            "Using {:?} {:?} with memory model {:?}",
            physical_device.properties().inner.properties.device_type,
            physical_device.properties().device_name(),
            physical_device.memory_model()
        );

        let extension_settings: &ExtensionSettings = app.world.resource();
        let device = Device::create(
            instance.clone(),
            physical_device,
            &[],
            &extension_settings.device_extensions,
        )
        .unwrap();

        app.insert_resource(device);
    }
}

pub trait RhyoliteApp {
    fn add_device_extension(&mut self, extension: &'static CStr) -> Option<Version>;
}

impl RhyoliteApp for App {
    fn add_device_extension(&mut self, extension: &'static CStr) -> Option<Version> {
        let mut extension_settings = self.world.resource_mut::<ExtensionSettings>();
        if let Some(v) = extension_settings.available_extensions.get(extension) {
            let v = *v;
            extension_settings
                .device_extensions
                .push(extension.as_ptr());
            Some(v)
        } else {
            None
        }
    }
}
