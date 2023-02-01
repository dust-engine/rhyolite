use ash::extensions::khr;
use ash::prelude::VkResult;
use ash::vk;

use std::future::Future;
use std::sync::{Arc, Mutex};
use std::{ops::Deref, pin::Pin};

use crate::future::QueueOperation;
use crate::queue;
use crate::{
    future::{GPUFuture},
    Device, ImageLike,
};

pub struct SwapchainLoader {
    loader: khr::Swapchain,
    device: Arc<Device>,
}

impl crate::HasDevice for SwapchainLoader {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl Deref for SwapchainLoader {
    type Target = khr::Swapchain;

    fn deref(&self) -> &Self::Target {
        &self.loader
    }
}

impl SwapchainLoader {
    pub fn new(device: Arc<Device>) -> Self {
        let loader = khr::Swapchain::new(device.instance(), &device);
        Self { loader, device }
    }
}

struct SwapchainInner {
    loader: Arc<SwapchainLoader>,
    swapchain: vk::SwapchainKHR,
}
impl Drop for SwapchainInner {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}

pub struct Swapchain {
    inner: Arc<SwapchainInner>,

    /// swapchain.acquire_next_image requires binary semaphore, and it signals one and only one semaphore.
    /// Due to these constraints, it make sense for the binary semaphore to be owned by the swapchain.
    images: Vec<vk::Image>,

    generation: u64,
}

/// Unsafe APIs for Swapchain
impl Swapchain {
    /// # Safety
    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCreateSwapchainKHR.html>
    pub unsafe fn create(
        loader: Arc<SwapchainLoader>,
        info: &vk::SwapchainCreateInfoKHR,
    ) -> VkResult<Self> {
        let swapchain = loader.create_swapchain(info, None)?;
        let images = loader.get_swapchain_images(swapchain)?;
        Ok(Self {
            inner: Arc::new(SwapchainInner {
                loader,
                swapchain,
            }),
            images,
            generation: 0,
        })
    }
    /// Returns (image_index, suboptimal)
    /// Semaphore must be binary semaphore
    /// # Safety
    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkAcquireNextImageKHR.html>
    unsafe fn acquire_next_image_raw(
        &mut self,
        timeout_ns: u64,
        semaphore: vk::Semaphore,
        fence: vk::Fence,
    ) -> VkResult<(u32, bool)> {
        // Requires exclusive access to swapchain
        self.inner.loader
            .acquire_next_image(self.inner.swapchain, timeout_ns, semaphore, fence)
    }

    // Returns: Suboptimal
    pub unsafe fn queue_present_raw(
        &mut self,
        queue: vk::Queue,
        wait_semaphores: &[vk::Semaphore],
        image_indice: u32,
    ) -> VkResult<bool> {
        let suboptimal = self.inner.loader.queue_present(
            queue,
            &vk::PresentInfoKHR {
                wait_semaphore_count: wait_semaphores.len() as u32,
                p_wait_semaphores: wait_semaphores.as_ptr(),
                swapchain_count: 1,
                p_swapchains: &self.inner.swapchain,
                p_image_indices: &image_indice,
                p_results: std::ptr::null_mut(), // Applications that do not need per-swapchain results can use NULL for pResults.
                ..Default::default()
            },
        );
        suboptimal
    }
}

pub struct SwapchainImage {
    swapchain: Arc<SwapchainInner>,
    image: vk::Image,
    indice: u32,
    suboptimal: bool,
    generation: u64,
}

impl ImageLike for SwapchainImage {
    fn raw_image(&self) -> vk::Image {
        self.image
    }
}


fn use_swapchain() {
    
}
