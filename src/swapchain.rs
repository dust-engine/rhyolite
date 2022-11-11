use ash::extensions::khr;
use ash::prelude::VkResult;
use ash::vk;

use std::future::Future;
use std::sync::{Arc, Mutex};
use std::{ops::Deref, pin::Pin};

use crate::{
    future::{GPUContext, GPUFuture},
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

pub struct Swapchain {
    pub(crate) loader: Arc<SwapchainLoader>,
    pub(crate) swapchain: Mutex<vk::SwapchainKHR>,
    images: Vec<vk::Image>,
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
            loader,
            swapchain: Mutex::new(swapchain),
            images,
        })
    }
    /// Returns (image_index, suboptimal)
    /// Semaphore must be binary semaphore
    /// # Safety
    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkAcquireNextImageKHR.html>
    unsafe fn acquire_next_image_raw(
        &self,
        timeout_ns: u64,
        semaphore: vk::Semaphore,
        fence: vk::Fence,
    ) -> VkResult<(u32, bool)> {
        let swapchain = self.swapchain.lock().unwrap();
        // Requires exclusive access to swapchain
        self.loader
            .acquire_next_image(*swapchain, timeout_ns, semaphore, fence)
    }

    // Returns: Suboptimal
    pub unsafe fn queue_present_raw(
        &self,
        queue: vk::Queue,
        wait_semaphores: &[vk::Semaphore],
        image_indice: u32,
    ) -> VkResult<bool> {
        let swapchain = self.swapchain.lock().unwrap();
        let swapchain_handle = *swapchain;
        let suboptimal = self.loader.queue_present(
            queue,
            &vk::PresentInfoKHR {
                wait_semaphore_count: wait_semaphores.len() as u32,
                p_wait_semaphores: wait_semaphores.as_ptr(),
                swapchain_count: 1,
                p_swapchains: &swapchain_handle,
                p_image_indices: &image_indice,
                p_results: std::ptr::null_mut(), // Applications that do not need per-swapchain results can use NULL for pResults.
                ..Default::default()
            },
        );
        drop(swapchain);
        suboptimal
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        let swapchain = self.swapchain.lock().unwrap();
        unsafe {
            self.loader.destroy_swapchain(*swapchain, None);
        }
    }
}

pub struct SwapchainImage {
    swapchain: Arc<Swapchain>,
    image: vk::Image,
    indice: u32,
    suboptimal: bool,
}

impl ImageLike for SwapchainImage {
    fn raw_image(&self) -> vk::Image {
        self.image
    }
}

struct SwapchainAcquireFuture {
    swapchain: Arc<Swapchain>,
    timeout_ns: u64,
    task: Option<blocking::Task<VkResult<(u32, bool)>>>,
}

impl Future for SwapchainAcquireFuture {
    type Output = VkResult<SwapchainImage>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.task.is_none() {
            let swapchain = self.swapchain.clone();
            let timeout_ns = self.timeout_ns;
            self.task = Some(blocking::unblock(move || unsafe {
                swapchain.acquire_next_image_raw(
                    timeout_ns,
                    vk::Semaphore::null(),
                    vk::Fence::null(),
                )
            }));
        }

        let task = self.task.as_mut().unwrap();
        let task = Pin::new(task);
        match task.poll(cx) {
            std::task::Poll::Ready(result) => {
                std::task::Poll::Ready(result.map(|(indice, suboptimal)| SwapchainImage {
                    swapchain: self.swapchain.clone(),
                    image: self.swapchain.images[indice as usize],
                    indice,
                    suboptimal,
                }))
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}
impl GPUFuture for SwapchainAcquireFuture {
    fn schedule(&self, cx: &mut GPUContext) {


        

    }
}

impl Swapchain {
    pub fn acquire_next_image(
        self: &Arc<Self>,
        timeout_ns: u64,
    ) -> impl GPUFuture<Output = VkResult<SwapchainImage>> {
        SwapchainAcquireFuture {
            swapchain: self.clone(),
            timeout_ns,
            task: None,
        }
    }
}

impl crate::Queue {
    /// Takes a future of SwapchainImage
    pub fn present(&mut self, image: SwapchainImage) -> VkResult<bool> {
        let suboptimal = unsafe {
            image
                .swapchain
                .queue_present_raw(self.queue, todo!(), image.indice)?
        };
        Ok(suboptimal)
    }
}
