use ash::vk;

use crate::{BufferLike, HasDevice};

pub trait BufferExt: BufferLike + Sized {
    fn with_device_address(self) -> DeviceAddressBuffer<Self> {
        let device_address = unsafe {
            self.device()
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo {
                    buffer: self.raw_buffer(),
                    ..Default::default()
                })
        };
        DeviceAddressBuffer {
            device_address: device_address + self.offset(),
            buffer: self,
        }
    }
}

pub struct DeviceAddressBuffer<B: BufferLike> {
    buffer: B,
    device_address: vk::DeviceAddress,
}
impl<B: BufferLike> HasDevice for DeviceAddressBuffer<B> {
    fn device(&self) -> &crate::Device {
        self.buffer.device()
    }
}
impl<B: BufferLike> BufferLike for DeviceAddressBuffer<B> {
    fn raw_buffer(&self) -> vk::Buffer {
        self.buffer.raw_buffer()
    }
    fn offset(&self) -> vk::DeviceSize {
        self.buffer.offset()
    }
    fn size(&self) -> vk::DeviceSize {
        self.buffer.size()
    }
}
