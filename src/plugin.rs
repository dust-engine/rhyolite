use std::{
    ffi::{c_char, CStr, CString},
    sync::Arc, collections::{BTreeMap, BTreeSet}, fmt::Display,
};
use thiserror::Error;
use ash::{vk, prelude::VkResult};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use crate::{Version, Instance, Device};
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
    enabled_instance_layers: Vec<*const CStr>,

    entry: Arc<ash::Entry>,
    available_layers: BTreeMap<String, LayerProperties>,
    available_extensions: BTreeMap<String, Version>,
}
unsafe impl Send for RhyolitePlugin{}
unsafe impl Sync for RhyolitePlugin{}
impl RhyolitePlugin {
    pub fn with_instance_extension(&mut self, extension: &CStr) {
        self.enabled_instance_extensions.push(extension.as_ptr());
    }
    pub fn with_instance_layer(&mut self, layer: &CStr) -> Option<Version> {
        self.enabled_instance_layers.push(layer);
    }
}


#[derive(Error, Debug)]
pub enum RhyoliltePluginInitError {
    #[error("Ash loading failed")]
    LoadingError(#[from] ash::LoadingError),
    
    #[error("Vulkan Implementation error: Invalid C String")]
    ImplementationError(#[from] std::ffi::FromBytesUntilNulError),

    
    #[error("Vulkan Implementation error: Invalid UTF-8 String")]
    ImplementationErrorUtf8(#[from] std::str::Utf8Error)
}

impl RhyolitePlugin {
    fn new() -> Result<Self, RhyoliltePluginInitError> {
        let entry = unsafe { ash::Entry::load().unwrap() };
        let entry = Arc::new(entry);
        let available_layers = entry
            .enumerate_instance_layer_properties()
            .unwrap()
            .into_iter()
            .map(|layer| -> Result<_, RhyoliltePluginInitError> {
                let str = CStr::from_bytes_until_nul(unsafe {
                    std::slice::from_raw_parts(layer.layer_name.as_ptr() as *const u8, layer.layer_name.len())
                })?;
                let str = str.to_str()?.to_string();
                Ok((str, LayerProperties {
                    implementation_version: Version(layer.implementation_version),
                    spec_version: Version(layer.spec_version),
                    description: CStr::from_bytes_until_nul(unsafe {
                        std::slice::from_raw_parts(layer.description.as_ptr() as *const u8, layer.description.len())
                    })?
                        .to_str()?
                        .to_string(),
                }))
            })
            .collect::<Vec<_>>();

        let available_extensions = entry
            .enumerate_instance_extension_properties(None)?
            .into_iter()
            .map(|ext| {
                let str = CStr::from_bytes_until_nul(&ext.extension_name)?;
                let str = str.to_str()?.to_string();
                (str, ext.spec_version.into())
            })
            .collect::<BTreeMap<_, _>>();
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
            available_extensions
        })
    }
}

#[derive(Default, Resource)]
struct ExtensionSettings {
    pub instance_extensions: Vec<*const c_char>,
    pub instance_layers: Vec<*const CStr>,
    pub device_extensions: Vec<*const CStr>,
}
unsafe impl Send for ExtensionSettings{}
unsafe impl Sync for ExtensionSettings{}


impl Plugin for RhyolitePlugin {
    fn build(&self, app: &mut App) {
    }
    fn cleanup(&self, app: &mut App) {
        app.world.remove_resource::<ExtensionSettings>();
    }
    fn finish(&self, app: &mut App) {
        let entry = unsafe { ash::Entry::load().unwrap() };
        let instance = {
            Arc::new(
                Instance::create(
                    Arc::new(entry),
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
                .unwrap(),
            )
        };
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

        let device = Device::create(
            instance,
            physical_device,
            &[],
            &self.enabled_device_extensions,
        ).unwrap();

        app.insert_resource(device);
    }
}

pub trait RhyoliteApp {
    fn add_device_extension(&mut self) -> Option<()>;
    fn add_instance_extension(&mut self) -> Option<()>;
    fn add_layer(&mut self) -> Option<()>;
}

impl RhyoliteApp for App {
}
