use crate::DebugUtilsMessenger;
use ash::{prelude::VkResult, vk};
use std::{ops::Deref, sync::Arc};

pub struct Instance {
    entry: Arc<ash::Entry>,
    instance: ash::Instance,
    debug_utils: DebugUtilsMessenger,
}

impl Instance {
    pub fn create(entry: Arc<ash::Entry>, info: &vk::InstanceCreateInfo) -> VkResult<Self> {
        // Safety: No Host Syncronization rules for vkCreateInstance.
        let mut instance = unsafe { entry.create_instance(info, None)? };
        let debug_utils = DebugUtilsMessenger::new(&entry, &mut instance)?;
        Ok(Instance {
            entry,
            instance,
            debug_utils,
        })
    }
    pub fn entry(&self) -> &Arc<ash::Entry> {
        &self.entry
    }
    pub fn debug_utils(&self) -> &DebugUtilsMessenger {
        &self.debug_utils
    }
}

impl Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

impl Drop for Instance {
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
            self.debug_utils
                .debug_utils
                .destroy_debug_utils_messenger(self.debug_utils.messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}
