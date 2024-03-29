use ash::{
    extensions::khr,
    prelude::VkResult,
    vk::{self},
};
use bevy::ecs::prelude::*;
use bevy::{app::prelude::*, asset::AssetApp};
use std::{
    any::Any,
    collections::BTreeMap,
    ffi::{c_char, CStr, CString},
    ops::Deref,
    sync::Arc,
};

use crate::{
    ecs::{DefaultCommandPool, PerFrame, RenderSystemPass},
    extensions::{DeviceExtension, InstanceExtension, PromotedDeviceExtension},
    Device, Feature, Instance, PhysicalDevice, PhysicalDeviceFeatures, QueuesRouter, Version,
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
            api_version: Version::new(0, 1, 2, 0),
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
pub(crate) type InstanceMetaBuilder =
    Box<dyn FnOnce(&ash::Entry, &ash::Instance) -> Box<dyn Any + Send + Sync> + Send + Sync>;
pub(crate) type DeviceMetaBuilder = Box<
    dyn FnOnce(&ash::Instance, &mut ash::Device) -> Option<Box<dyn Any + Send + Sync>>
        + Send
        + Sync,
>;
#[derive(Resource)]
struct DeviceExtensions {
    available_extensions: BTreeMap<CString, Version>,
    enabled_extensions: Vec<*const c_char>,
    extension_builders: Vec<DeviceMetaBuilder>,
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
            extension_builders: Vec::new(),
        })
    }
}
unsafe impl Send for DeviceExtensions {}
unsafe impl Sync for DeviceExtensions {}

#[derive(Resource)]
struct InstanceExtensions {
    available_extensions: BTreeMap<CString, Version>,
    enabled_extensions: Vec<*const c_char>,
    extension_builders: Vec<InstanceMetaBuilder>,
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
            extension_builders: Vec::new(),
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
        #[allow(unused_mut)]
        let mut instance_create_flags = vk::InstanceCreateFlags::empty();
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            instance_create_flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
            app.add_instance_extension_named(ash::vk::KhrPortabilityEnumerationFn::name())
                .unwrap();
        }
        let mut instance_extensions = app.world.remove_resource::<InstanceExtensions>();
        let instance_layers = app.world.remove_resource::<InstanceLayers>();
        let entry: &VulkanEntry = &app.world.get_resource_or_insert_with(VulkanEntry::default);
        let meta_builders = instance_extensions
            .as_mut()
            .map(|a| std::mem::take(&mut a.extension_builders))
            .unwrap_or_default();
        let instance = Instance::create(
            entry.0.clone(),
            crate::InstanceCreateInfo {
                flags: instance_create_flags,
                enabled_extension_names: instance_extensions
                    .as_ref()
                    .map(|f| f.enabled_extensions.as_slice())
                    .unwrap_or(&[]),
                enabled_layer_names: instance_layers
                    .as_ref()
                    .map(|f| f.enabled_layers.as_slice())
                    .unwrap_or(&[]),
                api_version: self.api_version,
                engine_name: self.engine_name.as_c_str(),
                engine_version: self.engine_version,
                application_name: self.application_name.as_c_str(),
                application_version: self.application_version,
                meta_builders,
            },
        )
        .unwrap();
        let physical_device = instance
            .enumerate_physical_devices()
            .unwrap()
            .skip(self.physical_device_index)
            .next()
            .unwrap();
        tracing::info!(
            "Using {:?} {:?} with memory model {:?}",
            physical_device.properties().device_type,
            physical_device.properties().device_name(),
            physical_device.properties().memory_model
        );
        let features = PhysicalDeviceFeatures::new(physical_device.clone());
        let extensions = DeviceExtensions::new(&physical_device).unwrap();

        let queue_router = QueuesRouter::find_with_queue_family_properties(
            &physical_device.get_queue_family_properties(),
        );
        app.insert_resource(extensions)
            .insert_resource(instance)
            .insert_resource(physical_device)
            .insert_resource(features)
            .insert_resource(queue_router)
            .init_asset::<crate::shader::ShaderModule>()
            .init_asset::<crate::shader::loader::SpirvShaderSource>();
        // Add build pass
        app.get_schedule_mut(PostUpdate)
            .as_mut()
            .unwrap()
            .add_build_pass(RenderSystemPass::new())
            .before::<bevy::ecs::schedule::passes::AutoInsertApplyDeferredPass>();

        // Required features
        app.enable_feature::<vk::PhysicalDeviceVulkan12Features>(|f| &mut f.timeline_semaphore)
            .unwrap();
        app.enable_feature::<vk::PhysicalDeviceVulkan13Features>(|f| &mut f.synchronization2)
            .or_enable_in_extension::<vk::PhysicalDeviceSynchronization2FeaturesKHR>(|f| {
                &mut f.synchronization2
            })
            .unwrap();

        // Optional extensions
        app.add_device_extension::<khr::DeferredHostOperations>();

        // IF supported, must be enabled.
        app.add_device_extension_named(vk::KhrPortabilitySubsetFn::name());

        #[cfg(feature = "glsl")]
        app.add_plugins(crate::shader::loader::GlslPlugin);
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
            extension_settings.extension_builders,
        )
        .unwrap();
        app.insert_resource(device);

        // Add allocator
        app.world.init_resource::<crate::Allocator>();
        app.world
            .init_resource::<crate::buffer::staging::StagingBelt>();
        app.world.init_resource::<crate::pipeline::PipelineCache>();
        app.world
            .init_resource::<crate::DeferredOperationTaskPool>();
        app.init_asset_loader::<crate::shader::loader::SpirvLoader>();
    }
}

pub trait RhyoliteApp {
    fn add_device_meta(&mut self, builder: DeviceMetaBuilder);
    fn add_instance_meta(&mut self, builder: InstanceMetaBuilder);
    /// Called in the [Plugin::build] phase of device plugins.
    /// Device plugins must be added after [RhyolitePlugin].
    fn add_device_extension<T: DeviceExtension>(&mut self) -> Option<Version>;

    /// Called in the [Plugin::build] phase of instance plugins.
    /// Instance plugins must be added after [RhyolitePlugin].
    fn add_instance_extension<T: InstanceExtension>(&mut self) -> Option<Version>;

    /// Called in the [Plugin::build] phase of device plugins.
    /// Device plugins must be added after [RhyolitePlugin].
    fn add_device_extension_named(&mut self, extension: &'static CStr) -> Option<Version>;

    /// Called in the [Plugin::build] phase of instance plugins.
    /// Instance plugins must be added after [RhyolitePlugin].
    fn add_instance_extension_named(&mut self, extension: &'static CStr) -> Option<Version>;

    /// Called in the [Plugin::build] phase of instance plugins.
    /// Instance plugins must be added after [RhyolitePlugin].
    fn add_instance_layer(&mut self, layer: &'static CStr) -> Option<LayerProperties>;

    /// Called in the [Plugin::build] phase of device plugins.
    /// Device plugins must be added after [RhyolitePlugin].
    fn enable_feature<T: Feature + Default>(
        &mut self,
        selector: impl FnMut(&mut T) -> &mut vk::Bool32,
    ) -> FeatureEnableResult;
}

impl RhyoliteApp for App {
    fn add_device_meta(&mut self, builder: DeviceMetaBuilder) {
        let Some(mut extension_settings) = self.world.get_resource_mut::<DeviceExtensions>() else {
            panic!("Device metas may only be added after the instance was created. Add RhyolitePlugin before all device plugins.")
        };
        extension_settings.extension_builders.push(builder);
    }
    fn add_instance_meta(&mut self, builder: InstanceMetaBuilder) {
        let extension_settings = self.world.get_resource_mut::<InstanceExtensions>();
        let mut extension_settings = match extension_settings {
            Some(extension_settings) => extension_settings,
            None => {
                let extension_settings = InstanceExtensions::from_world(&mut self.world);
                self.world.insert_resource(extension_settings);
                self.world.resource_mut::<InstanceExtensions>()
            }
        };
        extension_settings.extension_builders.push(builder);
    }
    fn add_device_extension<T: DeviceExtension>(&mut self) -> Option<Version> {
        let Some(mut extension_settings) = self.world.get_resource_mut::<DeviceExtensions>() else {
            panic!("Device extensions may only be added after the instance was created. Add RhyolitePlugin before all device plugins.")
        };
        if let Some(v) = extension_settings.available_extensions.get(T::name()) {
            let v = *v;
            extension_settings
                .enabled_extensions
                .push(T::name().as_ptr());
            extension_settings
                .extension_builders
                .push(Box::new(|instance, device| {
                    if let Some(t) = T::new(instance, device) {
                        Some(Box::new(t))
                    } else {
                        None
                    }
                }));
            Some(v)
        } else {
            None
        }
    }

    fn add_instance_extension<T: InstanceExtension>(&mut self) -> Option<Version> {
        let extension_settings = self.world.get_resource_mut::<InstanceExtensions>();
        let mut extension_settings = match extension_settings {
            Some(extension_settings) => extension_settings,
            None => {
                let extension_settings = InstanceExtensions::from_world(&mut self.world);
                self.world.insert_resource(extension_settings);
                self.world.resource_mut::<InstanceExtensions>()
            }
        };
        if let Some(v) = extension_settings.available_extensions.get(T::name()) {
            let v = *v;
            extension_settings
                .enabled_extensions
                .push(T::name().as_ptr());

            extension_settings
                .extension_builders
                .push(Box::new(|entry, instance| {
                    Box::new(T::new(entry, instance))
                }));
            Some(v)
        } else {
            None
        }
    }
    fn add_device_extension_named(&mut self, extension: &'static CStr) -> Option<Version> {
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
    fn add_instance_extension_named(&mut self, extension: &'static CStr) -> Option<Version> {
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
    fn enable_feature<'a, T: Feature + Default>(
        &'a mut self,
        selector: impl FnMut(&mut T) -> &mut vk::Bool32,
    ) -> FeatureEnableResult<'a> {
        let mut features = self.world.resource_mut::<PhysicalDeviceFeatures>();
        if features.enable_feature::<T>(selector).is_none() {
            return FeatureEnableResult::NotFound { app: self };
        }
        FeatureEnableResult::Success
    }
}

pub enum FeatureEnableResult<'a> {
    Success,
    NotFound { app: &'a mut App },
}
impl<'a> FeatureEnableResult<'a> {
    pub fn or_enable_in_extension<F: Feature + Default>(
        &'a mut self,
        selector: impl FnMut(&mut F) -> &mut vk::Bool32,
    ) -> Self
    where
        F::Extension: PromotedDeviceExtension,
    {
        match self {
            FeatureEnableResult::Success => Self::Success,
            FeatureEnableResult::NotFound { app } => {
                if app.add_device_extension::<F::Extension>().is_none() {
                    return Self::NotFound { app: *app };
                }
                app.enable_feature(selector)
            }
        }
    }
    #[track_caller]
    pub fn unwrap(&self) {
        match self {
            FeatureEnableResult::Success => (),
            FeatureEnableResult::NotFound { .. } => {
                panic!("Feature not found")
            }
        }
    }
}
