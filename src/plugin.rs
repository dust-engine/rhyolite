use ash::{
    prelude::VkResult,
    vk::{self},
};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use std::{
    collections::BTreeMap,
    ffi::{c_char, CStr, CString},
    ops::Deref,
    sync::Arc,
};

use crate::{
    command_pool::RecordingCommandBuffer,
    ecs::{RenderResRegistry, RenderSystemPass},
    Device, Feature, Instance, PhysicalDevice, PhysicalDeviceFeatures, PhysicalDeviceProperties,
    QueuesRouter, Version,
};
use cstr::cstr;

#[derive(Clone)]
pub struct LayerProperties {
    pub spec_version: Version,
    pub implementation_version: Version,
    pub description: String,
}

/// This is the point where the Vulkan instance and device are created.
/// All instance plugins must be added before RhyolitePlugin.
/// All device plugins must be added after RhyolitePlugin.
pub struct RhyolitePlugin {
    pub application_name: CString,
    pub application_version: Version,
    pub engine_name: CString,
    pub engine_version: Version,
    pub api_version: Version,

    pub physical_device_index: usize,
}
unsafe impl Send for RhyolitePlugin {}
unsafe impl Sync for RhyolitePlugin {}
impl Default for RhyolitePlugin {
    fn default() -> Self {
        Self {
            application_name: cstr!(b"Unnamed Application").to_owned(),
            application_version: Default::default(),
            engine_name: cstr!(b"Unnamed Engine").to_owned(),
            engine_version: Default::default(),
            api_version: Version::new(0, 1, 3, 0),
            physical_device_index: 0,
        }
    }
}
#[derive(Resource, Clone)]
pub struct VulkanEntry(Arc<ash::Entry>);
impl Deref for VulkanEntry {
    type Target = ash::Entry;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Default for VulkanEntry {
    fn default() -> Self {
        Self(Arc::new(unsafe { ash::Entry::load().unwrap() }))
    }
}

#[derive(Resource)]
struct DeviceExtensions {
    available_extensions: BTreeMap<CString, Version>,
    enabled_extensions: Vec<*const c_char>,
}
impl DeviceExtensions {
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
            enabled_extensions: Vec::new(),
        })
    }
}
unsafe impl Send for DeviceExtensions {}
unsafe impl Sync for DeviceExtensions {}

#[derive(Resource)]
struct InstanceExtensions {
    available_extensions: BTreeMap<CString, Version>,
    enabled_extensions: Vec<*const c_char>,
}
impl FromWorld for InstanceExtensions {
    fn from_world(world: &mut World) -> Self {
        if world.contains_resource::<Instance>() {
            panic!("Instance extensions may only be added before the instance was created");
        }
        let entry = world.get_resource_or_insert_with::<VulkanEntry>(VulkanEntry::default);
        let available_extensions = entry
            .enumerate_instance_extension_properties(None)
            .unwrap()
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
        Self {
            available_extensions,
            enabled_extensions: Vec::new(),
        }
    }
}
unsafe impl Send for InstanceExtensions {}
unsafe impl Sync for InstanceExtensions {}

#[derive(Resource)]
struct InstanceLayers {
    available_layers: BTreeMap<CString, LayerProperties>,
    enabled_layers: Vec<*const c_char>,
}
impl FromWorld for InstanceLayers {
    fn from_world(world: &mut World) -> Self {
        if world.contains_resource::<Instance>() {
            panic!("Instance layers may only be added before the instance was created");
        }
        let entry = world.get_resource_or_insert_with::<VulkanEntry>(VulkanEntry::default);
        let available_layers = entry
            .enumerate_instance_layer_properties()
            .unwrap()
            .into_iter()
            .map(|layer| {
                let str = CStr::from_bytes_until_nul(unsafe {
                    std::slice::from_raw_parts(
                        layer.layer_name.as_ptr() as *const u8,
                        layer.layer_name.len(),
                    )
                })
                .unwrap();
                (
                    str.to_owned(),
                    LayerProperties {
                        implementation_version: Version(layer.implementation_version),
                        spec_version: Version(layer.spec_version),
                        description: CStr::from_bytes_until_nul(unsafe {
                            std::slice::from_raw_parts(
                                layer.description.as_ptr() as *const u8,
                                layer.description.len(),
                            )
                        })
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_string(),
                    },
                )
            })
            .collect::<BTreeMap<CString, LayerProperties>>();
        Self {
            available_layers,
            enabled_layers: Vec::new(),
        }
    }
}
unsafe impl Send for InstanceLayers {}
unsafe impl Sync for InstanceLayers {}

impl Plugin for RhyolitePlugin {
    fn build(&self, app: &mut App) {
        let instance_extensions = app.world.remove_resource::<InstanceExtensions>();
        let instance_layers = app.world.remove_resource::<InstanceLayers>();
        let entry: &VulkanEntry = &app.world.get_resource_or_insert_with(VulkanEntry::default);
        let instance = Instance::create(
            entry.0.clone(),
            &crate::InstanceCreateInfo {
                enabled_extension_names: instance_extensions
                    .as_ref()
                    .map(|f| f.enabled_extensions.as_slice())
                    .unwrap_or(&[]),
                enabled_layer_names: instance_layers
                    .as_ref()
                    .map(|f| f.enabled_layers.as_slice())
                    .unwrap_or(&[]),
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
        let extensions = DeviceExtensions::new(&physical_device).unwrap();

        let queue_router = QueuesRouter::find_with_queue_family_properties(
            &physical_device.get_queue_family_properties(),
        );
        app.insert_resource(extensions)
            .insert_resource(instance)
            .insert_resource(physical_device)
            .insert_resource(properties)
            .insert_resource(features)
            .insert_resource(queue_router);

        // Add build pass
        app.get_schedule_mut(Update)
            .as_mut()
            .unwrap()
            .add_build_pass(RenderSystemPass::new())
            .before::<bevy_ecs::schedule::passes::AutoInsertApplyDeferredPass>();
        app.init_resource::<RenderResRegistry>();

        // Required features
        app.enable_feature::<vk::PhysicalDeviceVulkan12Features>(|f| &mut f.timeline_semaphore);
        app.enable_feature::<vk::PhysicalDeviceVulkan13Features>(|f| &mut f.synchronization2);
    }
    fn finish(&self, app: &mut App) {
        let extension_settings: DeviceExtensions =
            app.world.remove_resource::<DeviceExtensions>().unwrap();
        app.world
            .resource_mut::<PhysicalDeviceFeatures>()
            .finalize();
        let physical_device: &PhysicalDevice = app.world.resource();
        let features = app.world.resource::<PhysicalDeviceFeatures>();
        let queues_router = app.world.resource::<QueuesRouter>();
        let device = Device::create(
            physical_device.clone(),
            &queues_router.create_infos(),
            &extension_settings.enabled_extensions,
            features.pdevice_features2(),
        )
        .unwrap();
        app.insert_resource(device);
    }
}

pub trait RhyoliteApp {
    /// Called in the [Plugin::build] phase of device plugins.
    /// Device plugins must be added after [RhyolitePlugin].
    fn add_device_extension(&mut self, extension: &'static CStr) -> Option<Version>;

    /// Called in the [Plugin::build] phase of instance plugins.
    /// Instance plugins must be added after [RhyolitePlugin].
    fn add_instance_extension(&mut self, extension: &'static CStr) -> Option<Version>;

    /// Called in the [Plugin::build] phase of instance plugins.
    /// Instance plugins must be added after [RhyolitePlugin].
    fn add_instance_layer(&mut self, layer: &'static CStr) -> Option<LayerProperties>;

    /// Called in the [Plugin::build] phase of device plugins.
    /// Device plugins must be added after [RhyolitePlugin].
    fn enable_feature<T: Feature + Default>(
        &mut self,
        selector: impl FnMut(&mut T) -> &mut vk::Bool32,
    ) -> Option<()>;
}

impl RhyoliteApp for App {
    fn add_device_extension(&mut self, extension: &'static CStr) -> Option<Version> {
        let Some(mut extension_settings) = self.world.get_resource_mut::<DeviceExtensions>() else {
            panic!("Device extensions may only be added after the instance was created. Add RhyolitePlugin before all device plugins.")
        };
        if let Some(v) = extension_settings.available_extensions.get(extension) {
            let v = *v;
            extension_settings
                .enabled_extensions
                .push(extension.as_ptr());
            Some(v)
        } else {
            None
        }
    }
    fn add_instance_extension(&mut self, extension: &'static CStr) -> Option<Version> {
        let extension_settings = self.world.get_resource_mut::<InstanceExtensions>();
        let mut extension_settings = match extension_settings {
            Some(extension_settings) => extension_settings,
            None => {
                let extension_settings = InstanceExtensions::from_world(&mut self.world);
                self.world.insert_resource(extension_settings);
                self.world.resource_mut::<InstanceExtensions>()
            }
        };
        if let Some(v) = extension_settings.available_extensions.get(extension) {
            let v = *v;
            extension_settings
                .enabled_extensions
                .push(extension.as_ptr());
            Some(v)
        } else {
            None
        }
    }
    fn add_instance_layer(&mut self, layer: &'static CStr) -> Option<LayerProperties> {
        let layers = self.world.get_resource_mut::<InstanceLayers>();
        let mut layers = match layers {
            Some(layers) => layers,
            None => {
                let extension_settings = InstanceLayers::from_world(&mut self.world);
                self.world.insert_resource(extension_settings);
                self.world.resource_mut::<InstanceLayers>()
            }
        };
        if let Some(v) = layers.available_layers.get(layer) {
            let v = v.clone();
            layers.enabled_layers.push(layer.as_ptr());

            let vulkan_entry = self.world.resource::<VulkanEntry>();
            let additional_instance_extensions = vulkan_entry
                .enumerate_instance_extension_properties(Some(layer))
                .unwrap();

            let instance_extensions = self.world.get_resource_mut::<InstanceExtensions>();
            let mut instance_extensions = match instance_extensions {
                Some(instance_extensions) => instance_extensions,
                None => {
                    let instance_extensions = InstanceExtensions::from_world(&mut self.world);
                    self.world.insert_resource(instance_extensions);
                    self.world.resource_mut::<InstanceExtensions>()
                }
            };
            instance_extensions.available_extensions.extend(
                additional_instance_extensions.into_iter().map(|a| {
                    let name = unsafe {
                        CStr::from_bytes_until_nul(std::slice::from_raw_parts(
                            a.extension_name.as_ptr() as *const u8,
                            a.extension_name.len(),
                        ))
                    }
                    .unwrap();
                    let name = name.to_owned();
                    (name, Version(a.spec_version))
                }),
            );

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
