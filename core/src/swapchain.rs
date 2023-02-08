use ash::extensions::khr;
use ash::prelude::VkResult;
use ash::vk;
use pin_project::pin_project;

use std::future::Future;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::{ops::Deref, pin::Pin};

use crate::future::{Access, Res};
use crate::{
    Device, ImageLike, QueueFuture, QueueFuturePoll, QueueMask, QueueRef,
    QueueSubmissionContextSemaphoreWait, QueueSubmissionType,
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

pub struct SwapchainInner {
    loader: Arc<SwapchainLoader>,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    generation: u64,
}

pub struct Swapchain {
    inner: Arc<SwapchainInner>,
}

impl Drop for SwapchainInner {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}

/// Unsafe APIs for Swapchain
impl Swapchain {
    /// # Safety
    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCreateSwapchainKHR.html>
    pub fn create(
        loader: Arc<SwapchainLoader>,
        info: &vk::SwapchainCreateInfoKHR,
    ) -> VkResult<Self> {
        unsafe {
            let swapchain = loader.create_swapchain(info, None)?;
            let images = loader.get_swapchain_images(swapchain)?;
            let inner = SwapchainInner {
                loader,
                swapchain,
                images,
                generation: 0,
            };
            Ok(Self {
                inner: Arc::new(inner),
            })
        }
    }

    pub fn recreate(&mut self, info: &vk::SwapchainCreateInfoKHR) -> VkResult<()> {
        unsafe {
            let swapchain = self.inner.loader.create_swapchain(info, None)?;
            let images = self.inner.loader.get_swapchain_images(swapchain)?;
            let inner = SwapchainInner {
                loader: self.inner.loader.clone(),
                swapchain,
                images,
                generation: self.inner.generation.wrapping_add(1),
            };
            self.inner = Arc::new(inner);
        }
        Ok(())
    }

    pub fn acquire_next_image(&mut self, semaphore: vk::Semaphore) -> AcquireFuture {
        let (image_indice, suboptimal) = unsafe {
            self.inner.loader.acquire_next_image(
                self.inner.swapchain,
                !0,
                semaphore,
                vk::Fence::null(),
            )
        }
        .unwrap();
        let image = self.inner.images[image_indice as usize];
        let swapchain_image = SwapchainImage {
            swapchain: self.inner.swapchain,
            image,
            indice: image_indice,
            suboptimal,
            generation: self.inner.generation,
        };
        AcquireFuture {
            image: Some(swapchain_image),
            semaphore,
        }
    }
}

pub struct SwapchainImage {
    swapchain: vk::SwapchainKHR,
    image: vk::Image,
    indice: u32,
    suboptimal: bool,
    generation: u64,
}
impl Drop for SwapchainImage {
    fn drop(&mut self) {
        panic!("SwapchainImage must be returned to the OS by calling Present!")
    }
}

impl ImageLike for SwapchainImage {
    fn raw_image(&self) -> vk::Image {
        self.image
    }

    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        }
    }
}

#[pin_project]
pub struct PresentFuture {
    queue: QueueRef,
    prev_queue: QueueMask,
    swapchain: Vec<Res<SwapchainImage>>,
}

impl QueueFuture for PresentFuture {
    type Output = ();

    type RecycledState = ();

    type RetainedState = ();

    fn init(
        self: Pin<&mut Self>,
        ctx: &mut crate::SubmissionContext,
        recycled_state: &mut Self::RecycledState,
        prev_queue: crate::QueueMask,
    ) {
        let mut this = self.project();
        *this.prev_queue = prev_queue;
    }

    fn record(
        self: Pin<&mut Self>,
        ctx: &mut crate::SubmissionContext,
        recycled_state: &mut Self::RecycledState,
    ) -> QueueFuturePoll<Self::Output> {
        let this = self.project();

        let queue = {
            let mut mask = QueueMask::empty();
            mask.set_queue(*this.queue);
            mask
        };

        if !this.prev_queue.is_empty() {
            *this.prev_queue = QueueMask::empty();
            return QueueFuturePoll::Semaphore;
        }
        for swapchain in this.swapchain.iter() {
            let tracking = swapchain.tracking_info.borrow_mut();

            // If we consider the queue present operation as a read, then we only need to syncronize with previous writes.
            ctx.queues[tracking.queue_index.0 as usize]
                .signals
                .insert((tracking.current_stage_access.write_stages, true));
            ctx.queues[this.queue.0 as usize].waits.push(
                QueueSubmissionContextSemaphoreWait::WaitForSignal {
                    dst_stages: vk::PipelineStageFlags2::empty(),
                    queue: tracking.queue_index,
                    src_stages: tracking.current_stage_access.write_stages,
                },
            );
        }
        assert!(matches!(
            ctx.submission[this.queue.0 as usize],
            QueueSubmissionType::Unknown
        ));
        ctx.submission[this.queue.0 as usize] = QueueSubmissionType::Present(
            this.swapchain
                .iter()
                .map(|a| (a.inner.swapchain, a.inner.indice))
                .collect(),
        );
        QueueFuturePoll::Ready {
            next_queue: QueueMask::empty(),
            output: (),
        }
    }

    fn dispose(self) -> Self::RetainedState {
        ()
    }
}

#[pin_project]

pub struct AcquireFuture {
    image: Option<SwapchainImage>,
    semaphore: vk::Semaphore,
}
impl QueueFuture for AcquireFuture {
    type Output = Res<SwapchainImage>;

    type RecycledState = ();

    type RetainedState = ();

    fn init(
        self: Pin<&mut Self>,
        ctx: &mut crate::SubmissionContext,
        recycled_state: &mut Self::RecycledState,
        prev_queue: QueueMask,
    ) {
    }

    fn record(
        self: Pin<&mut Self>,
        ctx: &mut crate::SubmissionContext,
        recycled_state: &mut Self::RecycledState,
    ) -> QueueFuturePoll<Self::Output> {
        let this = self.project();
        let mut output = Res::new(this.image.take().unwrap());
        {
            let mut tracking = output.tracking_info.borrow_mut();
            tracking.untracked_semaphore = Some(*this.semaphore);
        }
        QueueFuturePoll::Ready {
            next_queue: QueueMask::empty(),
            output,
        }
    }

    fn dispose(self) -> Self::RetainedState {}
}
