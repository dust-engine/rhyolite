use crate::{cstr, plugin::InstanceMetaBuilder};
use ash::{prelude::VkResult, vk};
use bevy_ecs::system::Resource;
use bevy_utils::hashbrown::HashMap;
use std::{
    any::{Any, TypeId},
    ffi::{c_char, CStr},
    fmt::Debug,
    ops::Deref,
    sync::Arc,
};

#[derive(Clone, Resource)]
pub struct Instance(Arc<InstanceInner>);

struct InstanceInner {
    entry: Arc<ash::Entry>,
    instance: ash::Instance,
    metas: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

#[derive(Clone, Copy)]
pub struct Version(pub u32);
impl Version {
    pub fn new(variant: u32, major: u32, minor: u32, patch: u32) -> Self {
        let num = vk::make_api_version(variant, major, minor, patch);
        Self(num)
    }
    pub fn major(&self) -> u32 {
        vk::api_version_major(self.0)
    }
    pub fn minor(&self) -> u32 {
        vk::api_version_minor(self.0)
    }
    pub fn patch(&self) -> u32 {
        vk::api_version_patch(self.0)
    }
    pub fn variant(&self) -> u32 {
        vk::api_version_patch(self.0)
    }
    pub fn as_raw(&self) -> u32 {
        self.0
    }
}
impl Default for Version {
    fn default() -> Self {
        Self::new(0, 0, 1, 0)
    }
}
impl Debug for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Version({}.{}.{})",
            self.major(),
            self.minor(),
            self.patch()
        ))?;
        let variant = self.variant();
        if variant != 0 {
            f.write_fmt(format_args!(" variant {variant}"))?;
        }
        Ok(())
    }
}
impl From<Version> for String {
    fn from(value: Version) -> Self {
        format!(
            "{}.{}.{}:{}",
            value.major(),
            value.minor(),
            value.patch(),
            value.variant()
        )
    }
}

pub struct InstanceCreateInfo<'a> {
    pub application_name: &'a CStr,
    pub application_version: Version,
    pub engine_name: &'a CStr,
    pub engine_version: Version,
    pub api_version: Version,
    pub enabled_layer_names: &'a [*const c_char],
    pub enabled_extension_names: &'a [*const c_char],
    pub meta_builders: Vec<InstanceMetaBuilder>,
}

const DEFAULT_INSTANCE_EXTENSIONS: &[*const c_char] =
    &[ash::extensions::ext::DebugUtils::name().as_ptr()];
impl<'a> Default for InstanceCreateInfo<'a> {
    fn default() -> Self {
        Self {
            application_name: cstr!(b"Unnamed Application"),
            application_version: Default::default(),
            engine_name: cstr!(b"Unnamed Engine"),
            engine_version: Default::default(),
            api_version: Version::new(0, 1, 3, 0),
            enabled_layer_names: Default::default(),
            enabled_extension_names: DEFAULT_INSTANCE_EXTENSIONS,
            meta_builders: Default::default(),
        }
    }
}

impl Instance {
    pub fn create(entry: Arc<ash::Entry>, info: InstanceCreateInfo) -> VkResult<Self> {
        let application_info = vk::ApplicationInfo {
            p_application_name: info.application_name.as_ptr(),
            application_version: info.application_version.0,
            p_engine_name: info.engine_name.as_ptr(),
            engine_version: info.engine_version.0,
            api_version: info.api_version.0,
            ..Default::default()
        };
        let create_info = vk::InstanceCreateInfo {
            p_application_info: &application_info,
            enabled_layer_count: info.enabled_layer_names.len() as u32,
            pp_enabled_layer_names: info.enabled_layer_names.as_ptr(),
            enabled_extension_count: info.enabled_extension_names.len() as u32,
            pp_enabled_extension_names: info.enabled_extension_names.as_ptr(),
            ..Default::default()
        };
        // Safety: No Host Syncronization rules for vkCreateInstance.
        let instance = unsafe { entry.create_instance(&create_info, None)? };
        let metas: HashMap<TypeId, _> = info
            .meta_builders
            .into_iter()
            .map(|builder| {
                let item = builder(&entry, &instance);
                (item.type_id(), item)
            })
            .collect();
        Ok(Instance(Arc::new(InstanceInner {
            entry,
            instance,
            metas,
        })))
    }
    pub fn entry(&self) -> &Arc<ash::Entry> {
        &self.0.entry
    }
}

impl Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.0.instance
    }
}

impl Drop for InstanceInner {
    fn drop(&mut self) {
        tracing::info!(instance = ?self.instance.handle(), "drop instance");
        // Safety: Host Syncronization rule for vkDestroyInstance:
        // - Host access to instance must be externally synchronized.
        // - Host access to all VkPhysicalDevice objects enumerated from instance must be externally synchronized.
        // We have &mut self and therefore exclusive control on instance.
        // VkPhysicalDevice created from this Instance may not exist at this point,
        // because PhysicalDevice retains an Arc to Instance.
        // If there still exist a copy of PhysicalDevice, the Instance wouldn't be dropped.
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}
