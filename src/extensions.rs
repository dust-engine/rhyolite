use ash::{
    vk::{self},
    Device,
};

pub trait Extension: ash::vk::ExtensionMeta {
    fn promote_device(device: &mut ash::Device, ext: &Self::Device);
}
impl<T> Extension for T
where
    T: ash::vk::ExtensionMeta,
{
    default fn promote_device(device: &mut ash::Device, ext: &Self::Device) {}
}
#[derive(Debug)]
pub struct ExtensionNotFoundError;
impl From<ExtensionNotFoundError> for ash::vk::Result {
    fn from(_: ExtensionNotFoundError) -> Self {
        ash::vk::Result::ERROR_EXTENSION_NOT_PRESENT
    }
}

impl Extension for ash::khr::dynamic_rendering::Meta {
    fn promote_device(device: &mut ash::Device, ext: &Self::Device) {
        let device = ExposedDevice::new(device);
        device.device_fn_1_3.cmd_begin_rendering = ext.fp().cmd_begin_rendering_khr;
        device.device_fn_1_3.cmd_end_rendering = ext.fp().cmd_end_rendering_khr;
    }
}

impl Extension for ash::khr::synchronization2::Meta {
    fn promote_device(device: &mut ash::Device, ext: &Self::Device) {
        let device = ExposedDevice::new(device);
        device.device_fn_1_3.cmd_pipeline_barrier2 = ext.fp().cmd_pipeline_barrier2_khr;
        device.device_fn_1_3.cmd_reset_event2 = ext.fp().cmd_reset_event2_khr;
        device.device_fn_1_3.cmd_set_event2 = ext.fp().cmd_set_event2_khr;
        device.device_fn_1_3.cmd_wait_events2 = ext.fp().cmd_wait_events2_khr;
        device.device_fn_1_3.cmd_write_timestamp2 = ext.fp().cmd_write_timestamp2_khr;
        device.device_fn_1_3.queue_submit2 = ext.fp().queue_submit2_khr;
    }
}
pub struct ExposedDevice {
    handle: vk::Device,
    device_fn_1_0: ash::DeviceFnV1_0,
    device_fn_1_1: ash::DeviceFnV1_1,
    device_fn_1_2: ash::DeviceFnV1_2,
    device_fn_1_3: ash::DeviceFnV1_3,
}
impl ExposedDevice {
    pub fn new(device: &mut Device) -> &mut Self {
        unsafe { std::mem::transmute(device) }
    }
}
