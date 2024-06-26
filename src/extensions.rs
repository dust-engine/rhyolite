use ash::{vk, Device};

pub trait Extension: ash::vk::ExtensionMeta {
    fn promote_device(device: &mut ash::Device, ext: &Self::Device);
}
impl<T> Extension for T
where
    T: ash::vk::ExtensionMeta,
{
    default fn promote_device(_device: &mut ash::Device, _ext: &Self::Device) {}
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

impl Extension for ash::khr::maintenance1::Meta {
    fn promote_device(device: &mut ash::Device, ext: &Self::Device) {
        let device = ExposedDevice::new(device);
        device.device_fn_1_1.trim_command_pool = ext.fp().trim_command_pool_khr;
    }
}

impl Extension for ash::khr::maintenance3::Meta {
    fn promote_device(device: &mut ash::Device, ext: &Self::Device) {
        let device = ExposedDevice::new(device);
        device.device_fn_1_1.get_descriptor_set_layout_support =
            ext.fp().get_descriptor_set_layout_support_khr;
    }
}

impl Extension for ash::khr::maintenance4::Meta {
    fn promote_device(device: &mut ash::Device, ext: &Self::Device) {
        let device = ExposedDevice::new(device);
        device.device_fn_1_3.get_device_buffer_memory_requirements =
            ext.fp().get_device_buffer_memory_requirements_khr;
        device.device_fn_1_3.get_device_image_memory_requirements =
            ext.fp().get_device_image_memory_requirements_khr;
        device
            .device_fn_1_3
            .get_device_image_sparse_memory_requirements =
            ext.fp().get_device_image_sparse_memory_requirements_khr;
    }
}

pub struct ExposedDevice {
    pub handle: vk::Device,
    pub device_fn_1_0: ash::DeviceFnV1_0,
    pub device_fn_1_1: ash::DeviceFnV1_1,
    pub device_fn_1_2: ash::DeviceFnV1_2,
    pub device_fn_1_3: ash::DeviceFnV1_3,
}
impl ExposedDevice {
    pub fn new(device: &mut Device) -> &mut Self {
        unsafe { std::mem::transmute(device) }
    }
}
