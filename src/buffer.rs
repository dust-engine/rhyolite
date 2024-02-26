use ash::vk;

pub trait BufferLike {
    fn raw_buffer(&self) -> vk::Buffer;
    fn offset(&self) -> vk::DeviceSize {
        0
    }
    fn size(&self) -> vk::DeviceSize;
    fn device_address(&self) -> vk::DeviceAddress;
    /// If the buffer is host visible and mapped, this function returns the host-side address.
    fn as_mut_ptr(&mut self) -> Option<*mut u8>;
}

impl BufferLike for vk::Buffer {
    fn raw_buffer(&self) -> vk::Buffer {
        *self
    }
    fn size(&self) -> vk::DeviceSize {
        vk::WHOLE_SIZE
    }
    fn device_address(&self) -> vk::DeviceAddress {
        panic!()
    }
    fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        panic!()
    }
}
