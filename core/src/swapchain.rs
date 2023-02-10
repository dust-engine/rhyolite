use ash::extensions::khr;
use ash::prelude::VkResult;
use ash::vk;
use pin_project::pin_project;

use std::future::Future;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::{ops::Deref, pin::Pin};

use crate::future::{Access, Res, ResImage};
use crate::{
    Device, ImageLike, PhysicalDevice, QueueFuture, QueueFuturePoll, QueueMask, QueueRef,
    QueueSubmissionContextSemaphoreWait, QueueSubmissionType, Surface,
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
    device: Arc<Device>,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    generation: u64,

    surface: Arc<Surface>,
    extent: vk::Extent2D,
    layer_count: u32,
}

pub struct Swapchain {
    inner: Arc<SwapchainInner>,
}

impl Drop for SwapchainInner {
    fn drop(&mut self) {
        unsafe {
            self.device
                .swapchain_loader()
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

pub struct SwapchainCreateInfo<'a> {
    pub flags: vk::SwapchainCreateFlagsKHR,
    pub min_image_count: u32,
    pub image_format: vk::Format,
    pub image_color_space: vk::ColorSpaceKHR,
    pub image_extent: vk::Extent2D,
    pub image_array_layers: u32,
    pub image_usage: vk::ImageUsageFlags,
    pub image_sharing_mode: vk::SharingMode,
    pub queue_family_indices: &'a [u32],
    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
    pub present_mode: vk::PresentModeKHR,
    pub clipped: bool,
}

pub fn color_space_area(color_space: vk::ColorSpaceKHR) -> f32 {
    match color_space {
        vk::ColorSpaceKHR::SRGB_NONLINEAR => 0.112,
        vk::ColorSpaceKHR::EXTENDED_SRGB_NONLINEAR_EXT => 0.112,
        vk::ColorSpaceKHR::ADOBERGB_LINEAR_EXT => 0.151,
        vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT => 0.152,
        vk::ColorSpaceKHR::DISPLAY_P3_LINEAR_EXT => 0.152,
        vk::ColorSpaceKHR::DCI_P3_NONLINEAR_EXT => 0.5,
        vk::ColorSpaceKHR::BT709_LINEAR_EXT => 0.112,
        vk::ColorSpaceKHR::BT709_NONLINEAR_EXT => 0.112,
        vk::ColorSpaceKHR::BT2020_LINEAR_EXT => 0.212,
        vk::ColorSpaceKHR::HDR10_ST2084_EXT => 0.212,
        vk::ColorSpaceKHR::HDR10_HLG_EXT => 0.212,
        vk::ColorSpaceKHR::DOLBYVISION_EXT => 0.212,
        vk::ColorSpaceKHR::PASS_THROUGH_EXT => 0.0,
        vk::ColorSpaceKHR::DISPLAY_NATIVE_AMD => 1.0,
        _ => 0.0,
    }
}

impl<'a> SwapchainCreateInfo<'a> {
    pub fn pick(
        surface: &Surface,
        pdevice: &PhysicalDevice,
        usage: vk::ImageUsageFlags,
    ) -> VkResult<Self> {
        let formats = surface
            .pick_format(pdevice, usage)?
            .ok_or(vk::Result::ERROR_FORMAT_NOT_SUPPORTED)?;
        Ok(Self {
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            min_image_count: 3,
            image_format: formats.format,
            image_color_space: formats.color_space,
            image_extent: Default::default(),
            image_array_layers: 1,
            image_usage: usage,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_indices: &[],
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: vk::PresentModeKHR::FIFO,
            clipped: true,
        })
    }
}

/// Unsafe APIs for Swapchain
impl Swapchain {
    /// # Safety
    /// <https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCreateSwapchainKHR.html>
    pub fn create(
        device: Arc<Device>,
        surface: Arc<Surface>,
        info: SwapchainCreateInfo,
    ) -> VkResult<Self> {
        unsafe {
            let info = vk::SwapchainCreateInfoKHR {
                flags: info.flags,
                surface: surface.surface,
                min_image_count: info.min_image_count,
                image_format: info.image_format,
                image_color_space: info.image_color_space,
                image_extent: info.image_extent,
                image_array_layers: info.image_array_layers,
                image_usage: info.image_usage,
                image_sharing_mode: info.image_sharing_mode,
                queue_family_index_count: info.queue_family_indices.len() as u32,
                p_queue_family_indices: info.queue_family_indices.as_ptr(),
                pre_transform: info.pre_transform,
                composite_alpha: info.composite_alpha,
                present_mode: info.present_mode,
                clipped: info.clipped.into(),
                ..Default::default()
            };
            let swapchain = device.swapchain_loader().create_swapchain(&info, None)?;
            let images = device.swapchain_loader().get_swapchain_images(swapchain)?;
            let inner = SwapchainInner {
                device,
                surface,
                swapchain,
                images,
                generation: 0,
                extent: info.image_extent,
                layer_count: info.image_array_layers,
            };
            Ok(Self {
                inner: Arc::new(inner),
            })
        }
    }

    pub fn recreate(&mut self, info: SwapchainCreateInfo) -> VkResult<()> {
        unsafe {
            let info = vk::SwapchainCreateInfoKHR {
                flags: info.flags,
                surface: self.inner.surface.surface,
                min_image_count: info.min_image_count,
                image_format: info.image_format,
                image_color_space: info.image_color_space,
                image_extent: info.image_extent,
                image_array_layers: info.image_array_layers,
                image_usage: info.image_usage,
                image_sharing_mode: info.image_sharing_mode,
                queue_family_index_count: info.queue_family_indices.len() as u32,
                p_queue_family_indices: info.queue_family_indices.as_ptr(),
                pre_transform: info.pre_transform,
                composite_alpha: info.composite_alpha,
                present_mode: info.present_mode,
                clipped: info.clipped.into(),
                old_swapchain: self.inner.swapchain,
                ..Default::default()
            };
            let swapchain = self
                .inner
                .device
                .swapchain_loader()
                .create_swapchain(&info, None)?;
            let images = self
                .inner
                .device
                .swapchain_loader()
                .get_swapchain_images(swapchain)?;
            let inner = SwapchainInner {
                device: self.inner.device.clone(),
                surface: self.inner.surface.clone(),
                swapchain,
                images,
                generation: self.inner.generation.wrapping_add(1),
                extent: info.image_extent,
                layer_count: info.image_array_layers,
            };
            self.inner = Arc::new(inner);
        }
        Ok(())
    }

    pub fn acquire_next_image(&mut self, semaphore: vk::Semaphore) -> AcquireFuture {
        let (image_indice, suboptimal) = unsafe {
            self.inner.device.swapchain_loader().acquire_next_image(
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
            extent: self.inner.extent,
            layer_count: self.inner.layer_count,
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
    extent: vk::Extent2D,
    layer_count: u32,
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
            level_count: 1,
            base_array_layer: 0,
            layer_count: self.layer_count,
        }
    }
    fn extent(&self) -> vk::Extent3D {
        vk::Extent3D {
            width: self.extent.width,
            height: self.extent.height,
            depth: 1,
        }
    }
}

#[pin_project]
pub struct PresentFuture {
    queue: QueueRef,
    prev_queue: QueueMask,
    swapchain: Vec<ResImage<SwapchainImage>>,
}

impl ResImage<SwapchainImage> {
    pub fn present(self) -> PresentFuture {
        PresentFuture {
            queue: QueueRef::null(),
            prev_queue: QueueMask::empty(),
            swapchain: vec![self],
        }
    }
}

impl QueueFuture for PresentFuture {
    type Output = ();

    type RecycledState = ();

    type RetainedState = Vec<ResImage<SwapchainImage>>;

    fn init(
        self: Pin<&mut Self>,
        ctx: &mut crate::SubmissionContext,
        recycled_state: &mut Self::RecycledState,
        prev_queue: crate::QueueMask,
    ) {
        let this = self.project();
        *this.prev_queue = prev_queue;

        if this.queue.is_null() {
            let mut iter = prev_queue.iter();
            if let Some(inherited_queue) = iter.next() {
                *this.queue = inherited_queue;
                assert!(
                    iter.next().is_none(),
                    "Cannot use derived queue when the future depends on more than one queues"
                );
            } else {
                // Default to the first queue, if the queue does not have predecessor.
                *this.queue = QueueRef(0);
            }
        }
    }

    fn record(
        self: Pin<&mut Self>,
        ctx: &mut crate::SubmissionContext,
        recycled_state: &mut Self::RecycledState,
    ) -> QueueFuturePoll<Self::Output> {
        let this = self.project();

        if !this.prev_queue.is_empty() {
            *this.prev_queue = QueueMask::empty();
            return QueueFuturePoll::Semaphore;
        }
        for swapchain in this.swapchain.iter() {
            let tracking = swapchain.res.tracking_info.borrow_mut();

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
                .map(|a| (a.res.inner.swapchain, a.res.inner.indice))
                .collect(),
        );
        QueueFuturePoll::Ready {
            next_queue: QueueMask::empty(),
            output: (),
        }
    }

    fn dispose(self) -> Self::RetainedState {
        self.swapchain
    }
}

#[pin_project]

pub struct AcquireFuture {
    image: Option<SwapchainImage>,
    semaphore: vk::Semaphore,
}
impl QueueFuture for AcquireFuture {
    type Output = ResImage<SwapchainImage>;

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
        let output = ResImage::new(this.image.take().unwrap(), vk::ImageLayout::UNDEFINED);
        {
            let mut tracking = output.res.tracking_info.borrow_mut();
            tracking.untracked_semaphore = Some(*this.semaphore);
            tracking.current_stage_access = Access {
                read_stages: vk::PipelineStageFlags2::ALL_COMMANDS,
                write_stages: vk::PipelineStageFlags2::ALL_COMMANDS,
                ..Default::default()
            };
        }
        QueueFuturePoll::Ready {
            next_queue: QueueMask::empty(),
            output,
        }
    }

    fn dispose(self) -> Self::RetainedState {}
}
