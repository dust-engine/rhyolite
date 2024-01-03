use std::{
    ffi::{c_char, CStr, CString},
    sync::Arc,
};

use ash::vk;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use crate::{Version, Instance, Device};
use cstr::cstr;


pub struct RhyolitePlugin {
    pub enabled_instance_extensions: Vec<&'static CStr>,
    pub enabled_instance_layers: Vec<&'static CStr>,

    pub enabled_device_extensions: Vec<&'static CStr>,

    pub application_name: CString,
    pub application_version: Version,
    pub engine_name: CString,
    pub engine_version: Version,
    pub api_version: Version,

    pub physical_device_index: usize,
}
impl Default for RhyolitePlugin {
    fn default() -> Self {
        Self {
            application_name: cstr!(b"Unnamed Application").to_owned(),
            application_version: Default::default(),
            engine_name: cstr!(b"Unnamed Engine").to_owned(),
            engine_version: Default::default(),
            api_version: Version::new(0, 1, 3, 0),
            enabled_instance_layers: vec![],
            enabled_instance_extensions: vec![ash::extensions::khr::Surface::name()],
            physical_device_index: 0,
            enabled_device_extensions: vec![ash::extensions::khr::Swapchain::name()],
        }
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, PartialOrd, Ord, SystemSet)]
pub enum RenderSystems {
    SetUp,
    Render,
    CleanUp,
}

impl Plugin for RhyolitePlugin {
    fn build(&self, app: &mut bevy_app::App) {
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
