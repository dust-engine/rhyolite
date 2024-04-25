use rhyolite::{
    ash::{khr::acceleration_structure::Meta as AccelerationStructureExt, prelude::VkResult, vk},
    cstr,
    debug::DebugObject,
    Allocator, Buffer, BufferLike, Device, HasDevice,
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
impl DebugObject for AccelStruct {
    fn object_handle(&mut self) -> u64 {
        use rhyolite::ash::vk::Handle;
        self.raw.as_raw()
    }

    const OBJECT_TYPE: vk::ObjectType = vk::ObjectType::ACCELERATION_STRUCTURE_KHR;
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
}
