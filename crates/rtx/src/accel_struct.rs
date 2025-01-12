use rhyolite::{
    ash::{khr::acceleration_structure::Meta as AccelerationStructureExt, prelude::VkResult, vk},
    buffer::{Buffer, BufferLike},
    cstr,
    debug::DebugObject,
    utils::AsVkHandle,
    Allocator, Device, HasDevice,
};

pub struct AccelStruct {
    buffer: Buffer,
    pub(crate) raw: vk::AccelerationStructureKHR,
    pub(crate) flags: vk::BuildAccelerationStructureFlagsKHR,
    pub(crate) device_address: vk::DeviceAddress,
}
impl Drop for AccelStruct {
    fn drop(&mut self) {
        unsafe {
            self.buffer
                .device()
                .extension::<AccelerationStructureExt>()
                .destroy_acceleration_structure(self.raw, None);
        }
    }
}
impl HasDevice for AccelStruct {
    fn device(&self) -> &Device {
        self.buffer.device()
    }
}
impl AsVkHandle for AccelStruct {
    fn vk_handle(&self) -> vk::AccelerationStructureKHR {
        self.raw
    }
    type Handle = vk::AccelerationStructureKHR;
}
impl AccelStruct {
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }
    pub fn size(&self) -> vk::DeviceSize {
        self.buffer.size()
    }
    pub fn new_blas(allocator: Allocator, size: vk::DeviceSize) -> VkResult<Self> {
        Self::new(
            allocator,
            size,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        )
    }
    pub fn new_tlas(allocator: Allocator, size: vk::DeviceSize) -> VkResult<Self> {
        Self::new(allocator, size, vk::AccelerationStructureTypeKHR::TOP_LEVEL)
    }
    pub fn new(
        allocator: Allocator,
        size: vk::DeviceSize,
        ty: vk::AccelerationStructureTypeKHR,
    ) -> VkResult<Self> {
        let mut buffer = Buffer::new_resource(
            allocator,
            size,
            1,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;
        let name = if ty == vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL {
            cstr!("BLAS backing buffer")
        } else {
            cstr!("TLAS backing buffer")
        };
        buffer.set_name(name).ok();
        unsafe {
            let raw = buffer
                .device()
                .extension::<AccelerationStructureExt>()
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR {
                        ty,
                        size,
                        buffer: buffer.raw_buffer(),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            let device_address = buffer
                .device()
                .extension::<AccelerationStructureExt>()
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR {
                        acceleration_structure: raw,
                        ..Default::default()
                    },
                );
            Ok(Self {
                buffer,
                raw,
                flags: vk::BuildAccelerationStructureFlagsKHR::empty(),
                device_address,
            })
        }
    }
    pub fn raw(&self) -> vk::AccelerationStructureKHR {
        self.raw
    }
}
