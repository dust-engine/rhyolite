use crate::{
    cstr,
    extensions::{ExtensionNotFoundError, InstanceExtension},
    plugin::InstanceMetaBuilder,
};
use ash::{prelude::VkResult, vk};
use bevy::ecs::system::Resource;
use bevy::utils::hashbrown::HashMap;
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
    extensions: HashMap<&'static CStr, Option<Box<dyn Any + Send + Sync>>>,
    api_version: Version,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version(pub u32);
impl Version {
    pub const fn new(variant: u32, major: u32, minor: u32, patch: u32) -> Self {
        let num = vk::make_api_version(variant, major, minor, patch);
        Self(num)
    }
    pub const fn major(&self) -> u32 {
        vk::api_version_major(self.0)
    }
    pub const fn minor(&self) -> u32 {
        vk::api_version_minor(self.0)
    }
    pub const fn patch(&self) -> u32 {
        vk::api_version_patch(self.0)
    }
    pub const fn variant(&self) -> u32 {
        vk::api_version_patch(self.0)
    }
    pub const fn as_raw(&self) -> u32 {
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
    pub flags: vk::InstanceCreateFlags,
    pub application_name: &'a CStr,
    pub application_version: Version,
    pub engine_name: &'a CStr,
    pub engine_version: Version,
    pub api_version: Version,
    pub enabled_layer_names: &'a [*const c_char],
    pub enabled_extensions: HashMap<&'static CStr, Option<InstanceMetaBuilder>>,
}

impl<'a> Default for InstanceCreateInfo<'a> {
    fn default() -> Self {
        Self {
            flags: vk::InstanceCreateFlags::empty(),
            application_name: cstr!(b"Unnamed Application"),
            application_version: Default::default(),
            engine_name: cstr!(b"Unnamed Engine"),
            engine_version: Default::default(),
            api_version: Version::new(0, 1, 2, 0),
            enabled_layer_names: Default::default(),
            enabled_extensions: Default::default(),
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

        let enabled_extension_names = info
            .enabled_extensions
            .iter()
            .map(|(name, _)| name.as_ptr())
            .collect::<Vec<_>>();
        let create_info = vk::InstanceCreateInfo {
            p_application_info: &application_info,
            enabled_layer_count: info.enabled_layer_names.len() as u32,
            pp_enabled_layer_names: info.enabled_layer_names.as_ptr(),
            enabled_extension_count: enabled_extension_names.len() as u32,
            pp_enabled_extension_names: enabled_extension_names.as_ptr(),
            flags: info.flags,
            ..Default::default()
        };
        // Safety: No Host Syncronization rules for vkCreateInstance.
        let instance = unsafe { entry.create_instance(&create_info, None)? };
        let extensions: HashMap<&'static CStr, _> = info
            .enabled_extensions
            .into_iter()
            .map(|(name, builder)| {
                let item = builder.map(|builder| builder(&entry, &instance));
                (name, item)
            })
            .collect();
        Ok(Instance(Arc::new(InstanceInner {
            entry,
            instance,
            extensions,
            api_version: info.api_version,
        })))
    }
    pub fn entry(&self) -> &Arc<ash::Entry> {
        &self.0.entry
    }
    pub fn get_extension<T: InstanceExtension>(&self) -> Result<&T, ExtensionNotFoundError> {
        self.0
            .extensions
            .get(&T::name())
            .map(|item| {
                item.as_ref()
                    .expect("Instance extension does not have a function table.")
                    .downcast_ref::<T>()
                    .unwrap()
            })
            .ok_or(ExtensionNotFoundError)
    }
    pub fn extension<T: InstanceExtension>(&self) -> &T {
        self.get_extension::<T>().unwrap()
    }
    /// Returns the version of the Vulkan API used when creating the instance.
    pub fn api_version(&self) -> Version {
        self.0.api_version
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
        self.extensions.clear();
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
