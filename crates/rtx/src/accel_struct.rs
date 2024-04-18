use rhyolite::{
    ash::{extensions::khr, prelude::VkResult, vk},
    cstr,
    debug::DebugObject,
    Allocator, Buffer, BufferLike, HasDevice,
};

pub struct AccelStruct {
    buffer: Buffer,
    pub(crate) raw: vk::AccelerationStructureKHR,
    pub(crate) flags: vk::BuildAccelerationStructureFlagsKHR,
    device_address: vk::DeviceAddress,
}
impl Drop for AccelStruct {
    fn drop(&mut self) {
        unsafe {
            self.buffer
                .device()
                .extension::<khr::AccelerationStructure>()
                .destroy_acceleration_structure(self.raw, None);
        }
    }
}
impl AccelStruct {
    pub fn new_blas(allocator: Allocator, size: vk::DeviceSize) -> VkResult<Self> {
        Self::new(
            allocator,
            size,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        )
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
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        )?;
        let name = if ty == vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL {
            cstr!("BLAS backing buffer")
        } else {
            cstr!("TLAS backing buffer")
        };
        buffer.set_name(name).unwrap();
        unsafe {
            let raw = buffer
                .device()
                .extension::<khr::AccelerationStructure>()
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
                .extension::<khr::AccelerationStructure>()
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
}
