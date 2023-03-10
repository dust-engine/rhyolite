use std::sync::Arc;

use ash::{prelude::VkResult, vk};

use crate::{Allocator, BufferLike, Device, HasDevice, ResidentBuffer};

pub mod blas;
pub mod build;

pub struct AccelerationStructure {
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
    raw: vk::AccelerationStructureKHR,
    device: Arc<Device>,
    device_address: vk::DeviceAddress,
    ty: AccelerationStructureType,
    buffer: ResidentBuffer,
}

impl AccelerationStructure {
    pub fn raw(&self) -> vk::AccelerationStructureKHR {
        self.raw
    }
    pub fn new_blas_aabb(allocator: &Allocator, size: vk::DeviceSize) -> VkResult<Self> {
        Self::new(
            allocator,
            size,
            AccelerationStructureType::BottomLevelAABB,
            vk::AccelerationStructureCreateFlagsKHR::empty(),
        )
    }
    pub fn new_blas_triangle(allocator: &Allocator, size: vk::DeviceSize) -> VkResult<Self> {
        Self::new(
            allocator,
            size,
            AccelerationStructureType::BottomLevelTriangle,
            vk::AccelerationStructureCreateFlagsKHR::empty(),
        )
    }
    pub fn new_tlas(allocator: &Allocator, size: vk::DeviceSize) -> VkResult<Self> {
        Self::new(
            allocator,
            size,
            AccelerationStructureType::TopLevel,
            vk::AccelerationStructureCreateFlagsKHR::empty(),
        )
    }
    fn new(
        allocator: &Allocator,
        size: vk::DeviceSize,
        ty: AccelerationStructureType,
        create_flags: vk::AccelerationStructureCreateFlagsKHR,
    ) -> VkResult<Self> {
        let backing_buffer = allocator
            .create_device_buffer_uninit(
                size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            )
            .unwrap();
        let accel_struct = unsafe {
            allocator
                .device()
                .accel_struct_loader()
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR {
                        create_flags,
                        buffer: backing_buffer.raw_buffer(),
                        offset: 0,
                        size,
                        ty: ty.into(),
                        ..Default::default()
                    },
                    None,
                )
        }?;
        let device_address = unsafe {
            allocator
                .device()
                .accel_struct_loader()
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR {
                        acceleration_structure: accel_struct,
                        ..Default::default()
                    },
                )
        };
        Ok(Self {
            flags: vk::BuildAccelerationStructureFlagsKHR::empty(),
            raw: accel_struct,
            device: allocator.device().clone(),
            device_address,
            ty,
            buffer: backing_buffer,
        })
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        unsafe {
            self.device
                .accel_struct_loader()
                .destroy_acceleration_structure(self.raw, None)
        }
    }
}

#[derive(Clone, Copy)]
pub enum AccelerationStructureType {
    TopLevel,
    BottomLevelAABB,
    BottomLevelTriangle,
}
impl From<AccelerationStructureType> for vk::AccelerationStructureTypeKHR {
    fn from(value: AccelerationStructureType) -> Self {
        match value {
            AccelerationStructureType::TopLevel => vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            AccelerationStructureType::BottomLevelAABB
            | AccelerationStructureType::BottomLevelTriangle => {
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL
            }
        }
    }
}
