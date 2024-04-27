use ash::{prelude::VkResult, vk};
use bevy::app::Plugin;
use bevy::ecs::system::Resource;
use std::ffi::CStr;

use std::sync::RwLock;

use crate::commands::CommandRecorder;
use crate::plugin::RhyoliteApp;
use crate::Instance;
use ash::ext::debug_utils::Meta as DebugUtilsExt;

#[derive(Default)]
pub struct DebugUtilsPlugin;

impl Plugin for DebugUtilsPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        app.add_instance_extension::<DebugUtilsExt>().unwrap();
    }
    fn finish(&self, app: &mut bevy::app::App) {
        let instance: Instance = app.world.resource::<Instance>().clone();
        app.insert_resource(DebugUtilsMessenger::new(instance).unwrap());
    }
}

pub type DebugUtilsMessengerCallback = fn(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: &DebugUtilsMessengerCallbackData,
);

pub struct DebugUtilsMessengerCallbackData<'a> {
    /// Identifies the particular message ID that is associated with the provided message.
    /// If the message corresponds to a validation layer message, then this string may contain
    /// the portion of the Vulkan specification that is believed to have been violated.
    pub message_id_name: Option<&'a CStr>,
    /// The ID number of the triggering message. If the message corresponds to a validation layer
    /// message, then this number is related to the internal number associated with the message
    /// being triggered.
    pub message_id_number: i32,
    /// Details on the trigger conditions
    pub message: Option<&'a CStr>,
    pub queue_labels: &'a [vk::DebugUtilsLabelEXT<'a>],
    pub cmd_buf_labels: &'a [vk::DebugUtilsLabelEXT<'a>],
    pub objects: &'a [vk::DebugUtilsObjectNameInfoEXT<'a>],
}

#[derive(Resource)]
pub struct DebugUtilsMessenger(Box<DebugUtilsMessengerInner>);

struct DebugUtilsMessengerInner {
    instance: Instance,
    messenger: vk::DebugUtilsMessengerEXT,
    callbacks: RwLock<Vec<DebugUtilsMessengerCallback>>,
}
impl Drop for DebugUtilsMessenger {
    fn drop(&mut self) {
        unsafe {
            self.0
                .instance
                .extension::<DebugUtilsExt>()
                .destroy_debug_utils_messenger(self.0.messenger, None);
        }
    }
}

impl DebugUtilsMessenger {
    pub fn new(instance: Instance) -> VkResult<Self> {
        let mut this = Box::new(DebugUtilsMessengerInner {
            instance,
            messenger: vk::DebugUtilsMessengerEXT::default(),
            callbacks: RwLock::new(vec![default_callback]),
        });
        let messenger = unsafe {
            let p_user_data =
                this.as_mut() as *mut DebugUtilsMessengerInner as *mut std::ffi::c_void;
            // Safety:
            // The application must ensure that vkCreateDebugUtilsMessengerEXT is not executed in parallel
            // with any Vulkan command that is also called with instance or child of instance as the dispatchable argument.
            // We do this by taking a mutable reference to Instance.
            this.instance
                .extension::<DebugUtilsExt>()
                .create_debug_utils_messenger(
                    &vk::DebugUtilsMessengerCreateInfoEXT {
                        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                            | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                        message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                        pfn_user_callback: Some(debug_utils_callback),
                        // This is self-referencing: Self contains `vk::DebugUtilsMessengerEXT` which then
                        // contains a pointer to Self. It's fine because Self was boxed.
                        p_user_data,
                        ..Default::default()
                    },
                    None,
                )?
        };
        this.messenger = messenger;
        Ok(Self(this))
    }
    pub fn add_callback(&self, callback: DebugUtilsMessengerCallback) {
        let mut callbacks = self.0.callbacks.write().unwrap();
        callbacks.push(callback);
    }
}

unsafe extern "system" fn debug_utils_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let this: &DebugUtilsMessengerInner =
        &*(user_data as *mut DebugUtilsMessengerInner as *const DebugUtilsMessengerInner);
    let callback_data_raw = &*callback_data;
    let callback_data = DebugUtilsMessengerCallbackData {
        message_id_number: callback_data_raw.message_id_number,
        message_id_name: if callback_data_raw.p_message_id_name.is_null() {
            None
        } else {
            Some(CStr::from_ptr(callback_data_raw.p_message_id_name))
        },
        message: if callback_data_raw.p_message.is_null() {
            None
        } else {
            Some(CStr::from_ptr(callback_data_raw.p_message))
        },
        queue_labels: if callback_data_raw.queue_label_count == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(
                callback_data_raw.p_queue_labels,
                callback_data_raw.queue_label_count as usize,
            )
        },
        cmd_buf_labels: if callback_data_raw.queue_label_count == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(
                callback_data_raw.p_cmd_buf_labels,
                callback_data_raw.cmd_buf_label_count as usize,
            )
        },
        objects: if callback_data_raw.queue_label_count == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(
                callback_data_raw.p_objects,
                callback_data_raw.object_count as usize,
            )
        },
    };
    for callback in this.callbacks.read().unwrap().iter() {
        (callback)(severity, types, &callback_data)
    }
    // The callback returns a VkBool32, which is interpreted in a layer-specified manner.
    // The application should always return VK_FALSE. The VK_TRUE value is reserved for use in layer development.
    vk::FALSE
}

fn default_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: &DebugUtilsMessengerCallbackData,
) {
    use log::Level;
    let level = match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => Level::Debug,
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => Level::Info,
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => Level::Warn,
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => Level::Error,
        _ => Level::Trace,
    };

    if level == Level::Error {
        let bt = std::backtrace::Backtrace::capture();
        if bt.status() == std::backtrace::BacktraceStatus::Captured {
            println!("{}", bt);
        }
    }

    log::log!(
        target: "vulkan debug",
        level,
        message_type:? = types,
        message_id_number = callback_data.message_id_number,
        message_id_name = callback_data.message_id_name.and_then(|x| x.to_str().ok()).unwrap_or("Unknown Message Type");
        "{}",
        callback_data.message.and_then(|x| x.to_str().ok()).unwrap_or_default(),
    );
}

/// Vulkan Object that can be associated with a name and/or a tag.
pub trait DebugObject: crate::HasDevice {
    fn object_handle(&mut self) -> u64;
    const OBJECT_TYPE: vk::ObjectType;
    fn set_name(&mut self, cstr: &CStr) -> VkResult<()> {
        unsafe {
            let object_handle = self.object_handle();
            self.device()
                .get_extension::<DebugUtilsExt>()?
                .set_debug_utils_object_name(&vk::DebugUtilsObjectNameInfoEXT {
                    object_type: Self::OBJECT_TYPE,
                    object_handle,
                    p_object_name: cstr.as_ptr(),
                    ..Default::default()
                })
        }
    }
    fn with_name(mut self, name: &CStr) -> Self
    where
        Self: Sized,
    {
        self.set_name(name).ok();
        self
    }
    fn remove_name(&mut self) -> VkResult<()> {
        unsafe {
            let object_handle = self.object_handle();
            self.device()
                .get_extension::<DebugUtilsExt>()?
                .set_debug_utils_object_name(&vk::DebugUtilsObjectNameInfoEXT {
                    object_type: Self::OBJECT_TYPE,
                    object_handle,
                    p_object_name: std::ptr::null(),
                    ..Default::default()
                })
        }
    }
}

pub trait DebugCommands: CommandRecorder {
    fn begin_debug_label(&mut self, label: &CStr, color: [f32; 4]) {
        if self.device().get_extension::<DebugUtilsExt>().is_err() {
            return;
        }
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .extension::<DebugUtilsExt>()
                .cmd_begin_debug_utils_label(
                    cmd_buf,
                    &vk::DebugUtilsLabelEXT {
                        p_label_name: label.as_ptr(),
                        color,
                        ..Default::default()
                    },
                )
        }
    }
    fn end_debug_label(&mut self) {
        if self.device().get_extension::<DebugUtilsExt>().is_err() {
            return;
        }
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .extension::<DebugUtilsExt>()
                .cmd_end_debug_utils_label(cmd_buf)
        }
    }
    fn insert_debug_label(&mut self, label: &CStr, color: [f32; 4]) {
        if self.device().get_extension::<DebugUtilsExt>().is_err() {
            return;
        }
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .extension::<DebugUtilsExt>()
                .cmd_insert_debug_utils_label(
                    cmd_buf,
                    &vk::DebugUtilsLabelEXT {
                        p_label_name: label.as_ptr(),
                        color,
                        ..Default::default()
                    },
                )
        }
    }
}
impl<T> DebugCommands for T where T: CommandRecorder {}
