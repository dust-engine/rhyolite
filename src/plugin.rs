use ash::{
    prelude::VkResult,
    vk::{self},
};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use std::{
    collections::BTreeMap,
    ffi::{c_char, CStr, CString},
    sync::Arc,
};
use thiserror::Error;

use crate::{
    Device, Feature, Instance, PhysicalDevice, PhysicalDeviceFeatures, PhysicalDeviceProperties,
    Version,
};
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
    pub fn new() -> Result<Self, RhyoliltePluginInitError> {
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

#[derive(Resource)]
struct ExtensionSettings {
    available_extensions: BTreeMap<CString, Version>,
    device_extensions: Vec<*const c_char>,
}
impl ExtensionSettings {
    fn new(pdevice: &PhysicalDevice) -> VkResult<Self> {
        let extension_names = unsafe {
            pdevice
                .instance()
                .enumerate_device_extension_properties(pdevice.raw())?
        };
        let extension_names = extension_names
            .into_iter()
            .map(|ext| {
                let str = CStr::from_bytes_until_nul(unsafe {
                    std::slice::from_raw_parts(
                        ext.extension_name.as_ptr() as *const u8,
                        ext.extension_name.len(),
                    )
                })
                .unwrap();
                (str.to_owned(), Version(ext.spec_version))
            })
            .collect::<BTreeMap<CString, Version>>();
        Ok(Self {
            available_extensions: extension_names,
            device_extensions: Vec::new(),
        })
    }
}
unsafe impl Send for ExtensionSettings {}
unsafe impl Sync for ExtensionSettings {}

impl Plugin for RhyolitePlugin {
    fn build(&self, app: &mut App) {
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
        let physical_device = instance
            .enumerate_physical_devices()
            .unwrap()
            .skip(self.physical_device_index)
            .next()
            .unwrap();
        let properties = PhysicalDeviceProperties::new(physical_device.clone());
        tracing::info!(
            "Using {:?} {:?} with memory model {:?}",
            properties.device_type,
            properties.device_name(),
            properties.memory_model
        );
        let features = PhysicalDeviceFeatures::new(physical_device.clone());
        let extensions = ExtensionSettings::new(&physical_device).unwrap();
        app.insert_resource(extensions)
            .insert_resource(instance)
            .insert_resource(physical_device)
            .insert_resource(properties)
            .insert_resource(features);
    }
    fn finish(&self, app: &mut App) {
        let extension_settings: ExtensionSettings =
            app.world.remove_resource::<ExtensionSettings>().unwrap();
        app.world
            .resource_mut::<PhysicalDeviceFeatures>()
            .finalize();
        let physical_device: &PhysicalDevice = app.world.resource();
        let features = app.world.resource::<PhysicalDeviceFeatures>();
        let array = [1.0_f32];
        let device = Device::create(
            physical_device.clone(),
            &[vk::DeviceQueueCreateInfo {
                queue_family_index: 0,
                queue_count: 1,
                p_queue_priorities: array.as_ptr(),
                ..Default::default()
            }],
            &extension_settings.device_extensions,
            features.pdevice_features2(),
        )
        .unwrap();

        app.insert_resource(device);
    }
}

pub trait RhyoliteApp {
    fn add_device_extension(&mut self, extension: &'static CStr) -> Option<Version>;
    fn enable_feature<T: Feature + Default>(
        &mut self,
        selector: impl FnMut(&mut T) -> &mut vk::Bool32,
    ) -> Option<()>;
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
    fn enable_feature<T: Feature + Default>(
        &mut self,
        selector: impl FnMut(&mut T) -> &mut vk::Bool32,
    ) -> Option<()> {
        let mut features = self.world.resource_mut::<PhysicalDeviceFeatures>();
        features.enable_feature::<T>(selector)
    }
}
