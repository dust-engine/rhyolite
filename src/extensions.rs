use ash::{Device, Entry, Instance, vk};

use crate::Feature;

pub trait InstanceExtension: Send + Sync + 'static {
    fn new(entry: &Entry, instance: &Instance) -> Self;
    fn name() -> &'static std::ffi::CStr;
}

pub trait DeviceExtension: Send + Sync + Sized + 'static {
    fn new(instance: &Instance, device: &mut Device) -> Option<Self>;
    fn name() -> &'static std::ffi::CStr;
}

#[derive(Debug)]
pub struct ExtensionNotFoundError;
impl From<ExtensionNotFoundError> for ash::vk::Result {
    fn from(_: ExtensionNotFoundError) -> Self {
        ash::vk::Result::ERROR_EXTENSION_NOT_PRESENT
    }
}

macro_rules! impl_device_extension {
    ($name:ident $( :: $Vb : ident )*) => {
        impl DeviceExtension for $name$(::$Vb)* {
            fn new(instance: &Instance, device: &mut Device) -> Option<Self> {
                Some($name$(::$Vb)*::new(instance, device))
            }
            fn name() -> &'static std::ffi::CStr {
                $name$(::$Vb)*::name()
            }
        }
    };
}
macro_rules! impl_instance_extension {
    ($name:ident $( :: $Vb : ident )*) => {
        impl InstanceExtension for $name$(::$Vb)* {
            fn new(entry: &Entry, instance: &Instance) -> Self {
                $name$(::$Vb)*::new(entry, instance)
            }
            fn name() -> &'static std::ffi::CStr {
                $name$(::$Vb)*::name()
            }
        }
    };
}
impl_device_extension!(ash::extensions::khr::AccelerationStructure);
impl_instance_extension!(ash::extensions::khr::AndroidSurface);
impl_device_extension!(ash::extensions::khr::BufferDeviceAddress);
impl_device_extension!(ash::extensions::khr::CopyCommands2);
impl_device_extension!(ash::extensions::khr::CreateRenderPass2);
impl_device_extension!(ash::extensions::khr::DeferredHostOperations);
impl_device_extension!(ash::extensions::khr::DeviceGroup);
//impl_instance_extension!(ash::extensions::khr::DeviceGroupCreation);
impl_instance_extension!(ash::extensions::khr::Display);
impl_device_extension!(ash::extensions::khr::DisplaySwapchain);
impl_device_extension!(ash::extensions::khr::DrawIndirectCount);
impl DeviceExtension for ash::extensions::khr::DynamicRendering {
    fn new(instance: &Instance, device: &mut Device) -> Option<Self> {
        let ext = ash::extensions::khr::DynamicRendering::new(instance, device);
        let device = ExposedDevice::new(device);
        device.device_fn_1_3.cmd_begin_rendering = ext.fp().cmd_begin_rendering_khr;
        device.device_fn_1_3.cmd_end_rendering = ext.fp().cmd_end_rendering_khr;
        Some(ext)
    }
    fn name() -> &'static std::ffi::CStr {
        ash::extensions::khr::DynamicRendering::name()
    }

}
impl_device_extension!(ash::extensions::khr::ExternalFenceFd);
impl_device_extension!(ash::extensions::khr::ExternalFenceWin32);
impl_device_extension!(ash::extensions::khr::ExternalMemoryFd);
impl_device_extension!(ash::extensions::khr::ExternalMemoryWin32);
impl_device_extension!(ash::extensions::khr::ExternalSemaphoreFd);
impl_device_extension!(ash::extensions::khr::ExternalSemaphoreWin32);
impl_device_extension!(ash::extensions::khr::GetMemoryRequirements2);
impl_instance_extension!(ash::extensions::khr::GetPhysicalDeviceProperties2);
impl_instance_extension!(ash::extensions::khr::GetSurfaceCapabilities2);
impl_device_extension!(ash::extensions::khr::Maintenance1);
impl_device_extension!(ash::extensions::khr::Maintenance3);
impl_device_extension!(ash::extensions::khr::Maintenance4);
impl_instance_extension!(ash::extensions::khr::PerformanceQuery);
impl_device_extension!(ash::extensions::khr::PipelineExecutableProperties);
impl_device_extension!(ash::extensions::khr::PresentWait);
impl_device_extension!(ash::extensions::khr::PushDescriptor);
impl_device_extension!(ash::extensions::khr::RayTracingPipeline);
impl_device_extension!(ash::extensions::khr::RayTracingMaintenance1);
impl_instance_extension!(ash::extensions::khr::Surface);
impl_device_extension!(ash::extensions::khr::Swapchain);
impl DeviceExtension for ash::extensions::khr::Synchronization2 {
    fn new(instance: &Instance, device: &mut Device) -> Option<Self> {
        let ext = ash::extensions::khr::Synchronization2::new(instance, device);
        let device = ExposedDevice::new(device);
        device.device_fn_1_3.cmd_pipeline_barrier2 = ext.fp().cmd_pipeline_barrier2_khr;
        device.device_fn_1_3.cmd_reset_event2 = ext.fp().cmd_reset_event2_khr;
        device.device_fn_1_3.cmd_set_event2 = ext.fp().cmd_set_event2_khr;
        device.device_fn_1_3.cmd_wait_events2 = ext.fp().cmd_wait_events2_khr;
        device.device_fn_1_3.cmd_write_timestamp2 = ext.fp().cmd_write_timestamp2_khr;
        device.device_fn_1_3.queue_submit2 = ext.fp().queue_submit2_khr;
        Some(ext)
    }
    fn name() -> &'static std::ffi::CStr {
        ash::extensions::khr::Synchronization2::name()
    }
}
impl_device_extension!(ash::extensions::khr::TimelineSemaphore);
impl_instance_extension!(ash::extensions::khr::WaylandSurface);
impl_instance_extension!(ash::extensions::khr::Win32Surface);
impl_instance_extension!(ash::extensions::khr::XcbSurface);
impl_instance_extension!(ash::extensions::khr::XlibSurface);

impl_instance_extension!(ash::extensions::ext::AcquireDrmDisplay);
impl_device_extension!(ash::extensions::ext::BufferDeviceAddress);
impl_instance_extension!(ash::extensions::ext::CalibratedTimestamps);
impl_instance_extension!(ash::extensions::ext::DebugUtils);
impl_device_extension!(ash::extensions::ext::DescriptorBuffer);
impl_device_extension!(ash::extensions::ext::ExtendedDynamicState);
impl_device_extension!(ash::extensions::ext::ExtendedDynamicState2);
impl_device_extension!(ash::extensions::ext::ExtendedDynamicState3);
impl_device_extension!(ash::extensions::ext::FullScreenExclusive);
impl_instance_extension!(ash::extensions::ext::HeadlessSurface);
impl_device_extension!(ash::extensions::ext::ImageCompressionControl);
impl_device_extension!(ash::extensions::ext::ImageDrmFormatModifier);
impl_device_extension!(ash::extensions::ext::MeshShader);
impl_instance_extension!(ash::extensions::ext::MetalSurface);
impl_device_extension!(ash::extensions::ext::PipelineProperties);
impl_device_extension!(ash::extensions::ext::PrivateData);
impl_instance_extension!(ash::extensions::ext::SampleLocations);
impl_device_extension!(ash::extensions::ext::ShaderObject);
impl_instance_extension!(ash::extensions::ext::ToolingInfo);

impl_instance_extension!(ash::extensions::nv::CoverageReductionMode);
impl_device_extension!(ash::extensions::nv::DeviceDiagnosticCheckpoints);
impl_device_extension!(ash::extensions::nv::MeshShader);
impl_device_extension!(ash::extensions::nv::RayTracing);


pub struct ExposedDevice {
    handle: vk::Device,
    device_fn_1_0: vk::DeviceFnV1_0,
    device_fn_1_1: vk::DeviceFnV1_1,
    device_fn_1_2: vk::DeviceFnV1_2,
    device_fn_1_3: vk::DeviceFnV1_3,
}
impl ExposedDevice {
    pub fn new(device: &mut Device) -> &mut Self {
        unsafe {
            std::mem::transmute(device)
        }
    }
}

