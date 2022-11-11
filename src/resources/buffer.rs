use std::sync::Arc;

use ash::vk;
use async_ash_macro::gpu;
use futures_util::Future;

use crate::{
    future::{GPUFuture, GPUTaskContext},
    Device,
};

pub trait BufferLike {
    fn raw_image(&self) -> vk::Image;
}

pub struct Buffer {
    device: Arc<Device>,
    image: vk::Image,
}

impl BufferLike for Buffer {
    fn raw_image(&self) -> vk::Image {
        self.image
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image(self.image, None);
        }
    }
}

pub struct BufferSlice<'a, B: BufferLike> {
    buffer: &'a B,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

trait BufferExt {
    fn copy_buffer<T: BufferLike>(&self, other: &mut T) -> impl GPUFuture<Output = ()> {
        gpu! {}
    }
}

impl<T: BufferLike> BufferExt for T {}

struct CopyBufferFuture {}

impl Future for CopyBufferFuture {
    type Output = ();

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let cx = cx.get();
        std::task::Poll::Ready(())
    }
}

impl GPUFuture for CopyBufferFuture {
    fn priority(&self) -> u64 {
        1
    }
}
