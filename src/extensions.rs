use ash::{
    vk::{self, DeviceExtension},
    Device, Entry, Instance,
};

use crate::Version;

pub trait PromotedDeviceExtension: ash::vk::DeviceExtension {
    fn promote(&self, device: &mut Device);
}

#[derive(Debug)]
pub struct ExtensionNotFoundError;
impl From<ExtensionNotFoundError> for ash::vk::Result {
    fn from(_: ExtensionNotFoundError) -> Self {
        ash::vk::Result::ERROR_EXTENSION_NOT_PRESENT
    }
}

impl PromotedDeviceExtension for ash::khr::dynamic_rendering::Device {
    fn promote(&self, device: &mut Device) {
        let device = ExposedDevice::new(device);
        device.device_fn_1_3.cmd_begin_rendering = self.fp().cmd_begin_rendering_khr;
        device.device_fn_1_3.cmd_end_rendering = self.fp().cmd_end_rendering_khr;
    }
}

impl PromotedDeviceExtension for ash::khr::synchronization2::Device {
    fn promote(&self, device: &mut Device) {
        let device = ExposedDevice::new(device);
        device.device_fn_1_3.cmd_pipeline_barrier2 = self.fp().cmd_pipeline_barrier2_khr;
        device.device_fn_1_3.cmd_reset_event2 = self.fp().cmd_reset_event2_khr;
        device.device_fn_1_3.cmd_set_event2 = self.fp().cmd_set_event2_khr;
        device.device_fn_1_3.cmd_wait_events2 = self.fp().cmd_wait_events2_khr;
        device.device_fn_1_3.cmd_write_timestamp2 = self.fp().cmd_write_timestamp2_khr;
        device.device_fn_1_3.queue_submit2 = self.fp().queue_submit2_khr;
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
