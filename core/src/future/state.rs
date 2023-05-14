use std::cell::UnsafeCell;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;

pub fn use_state<T>(
    this: &mut Option<T>,
    init: impl FnOnce() -> T,
    update: impl FnOnce(&mut T),
) -> &mut T {
    if let Some(inner) = this {
        update(inner)
    }
    this.get_or_insert_with(init)
}

struct SharedDeviceStateInner<T> {
    item: T,
    tracking_feedback: UnsafeCell<TrackingFeedback>, // TODO: remove Mutex
    fetched: AtomicBool,
}
unsafe impl<T> Send for SharedDeviceStateInner<T> {}
unsafe impl<T> Sync for SharedDeviceStateInner<T> {}

pub struct SharedDeviceStateHostContainer<T>(Arc<SharedDeviceStateInner<T>>);

/// Indicates shared resource access on the device-side only.
/// It is safe to implement ImageLike and BufferLike, because
/// it is implied that at any given time, only one submission
/// may hold a SharedDeviceState<T>.
pub struct SharedDeviceState<T>(Arc<SharedDeviceStateInner<T>>);
impl<T> RenderData for SharedDeviceState<T> {
    fn tracking_feedback(&mut self, feedback: &super::TrackingFeedback) {
        assert!(self.0.fetched.load(std::sync::atomic::Ordering::Relaxed));
        self.0
            .fetched
            .store(false, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            *self.0.tracking_feedback.get() = feedback.clone();
        }
    }
}
impl<T> SharedDeviceStateInner<T> {
    pub fn get_tracking_feedback(&self) -> &super::TrackingFeedback {
        assert!(self.fetched.load(std::sync::atomic::Ordering::Relaxed));
        unsafe { &*self.tracking_feedback.get() }
    }
}
impl<T: HasDevice> HasDevice for SharedDeviceState<T> {
    fn device(&self) -> &Arc<crate::Device> {
        self.0.item.device()
    }
}
impl<T: ImageLike> ImageLike for SharedDeviceState<T> {
    fn raw_image(&self) -> ash::vk::Image {
        self.0.item.raw_image()
    }

    fn subresource_range(&self) -> ash::vk::ImageSubresourceRange {
        self.0.item.subresource_range()
    }

    fn extent(&self) -> ash::vk::Extent3D {
        self.0.item.extent()
    }

    fn format(&self) -> ash::vk::Format {
        self.0.item.format()
    }
    fn offset(&self) -> ash::vk::Offset3D {
        self.0.item.offset()
    }
}
impl<T: BufferLike> BufferLike for SharedDeviceState<T> {
    fn raw_buffer(&self) -> ash::vk::Buffer {
        self.0.item.raw_buffer()
    }

    fn size(&self) -> ash::vk::DeviceSize {
        self.0.item.size()
    }
    fn offset(&self) -> ash::vk::DeviceSize {
        self.0.item.offset()
    }
    fn device_address(&self) -> ash::vk::DeviceAddress {
        self.0.item.device_address()
    }
}
impl<T: ImageViewLike> ImageViewLike for SharedDeviceState<T> {
    fn raw_image_view(&self) -> ash::vk::ImageView {
        self.0.item.raw_image_view()
    }
}

/// Creates a resource to be used by multiple frames. Generally this is applicable to resources
/// that stay entirely on the device side.
/// If `should_update` returns false, returns a reference to the current object.
/// If `should_update` returns true, calls `create` to create a new object.
/// The old object will be dropped when its reference count drops to zero.
pub fn use_shared_state<T>(
    this: &mut Option<SharedDeviceStateHostContainer<T>>,
    create: impl FnOnce(Option<&T>) -> T,
    should_update: impl FnOnce(&T) -> bool,
) -> RenderRes<SharedDeviceState<T>> {
    if let Some(inner) = this {
        let inner = &mut inner.0;
        if should_update(&inner.item) {
            *inner = Arc::new(SharedDeviceStateInner {
                item: create(Some(&inner.item)),
                tracking_feedback: Default::default(),
                fetched: AtomicBool::new(true),
            });
            RenderRes::new(SharedDeviceState(inner.clone()))
        } else {
            inner
                .fetched
                .store(true, std::sync::atomic::Ordering::Relaxed);
            RenderRes::with_feedback(
                SharedDeviceState(inner.clone()),
                &inner.get_tracking_feedback(),
            )
        }
    } else {
        let item = Arc::new(SharedDeviceStateInner {
            item: create(None),
            tracking_feedback: Default::default(),
            fetched: AtomicBool::new(true),
        });
        *this = Some(SharedDeviceStateHostContainer(item));
        RenderRes::new(SharedDeviceState(this.as_ref().unwrap().0.clone()))
    }
}

pub fn use_shared_image<T>(
    this: &mut Option<SharedDeviceStateHostContainer<T>>,
    create: impl FnOnce(Option<&T>) -> (T, vk::ImageLayout),
    should_update: impl FnOnce(&T) -> bool,
) -> RenderImage<SharedDeviceState<T>> {
    if let Some(inner) = this {
        let inner = &mut inner.0;
        if should_update(&inner.item) {
            let (item, layout) = create(Some(&inner.item));
            *inner = Arc::new(SharedDeviceStateInner {
                item,
                tracking_feedback: Default::default(),
                fetched: AtomicBool::new(true),
            });
            RenderImage::new(SharedDeviceState(inner.clone()), layout)
        } else {
            inner
                .fetched
                .store(true, std::sync::atomic::Ordering::Relaxed);
            RenderImage::with_feedback(
                SharedDeviceState(inner.clone()),
                &inner.get_tracking_feedback(),
            )
        }
    } else {
        let (item, layout) = create(None);
        let item = Arc::new(SharedDeviceStateInner {
            item,
            tracking_feedback: Default::default(),
            fetched: AtomicBool::new(true),
        });
        *this = Some(SharedDeviceStateHostContainer(item));
        RenderImage::new(SharedDeviceState(this.as_ref().unwrap().0.clone()), layout)
    }
}
/*
pub fn use_shared_state_initialized<'a, T, Fut: GPUCommandFuture<Output = RenderRes<T>>>(
    this: &'a mut Option<SharedDeviceStateHostContainer<T>>,
    create: impl FnOnce(Option<&T>) -> Fut + 'a,
    should_update: impl FnOnce(&T) -> bool + 'a,
) -> impl GPUCommandFuture<Output = RenderRes<SharedDeviceState<T>>> + 'a
where
    Fut::RetainedState: 'a,
    Fut::RecycledState: 'a,
{
    commands! { move
        if let Some(inner) = this {
            let inner = &mut inner.0;
            if should_update(&inner.item) {
                let res = create(Some(&inner.item)).await.map(|a| Arc::new(a));
                *inner = res.inner().clone();
                res.map(|a| SharedDeviceState(a))
            } else {
                RenderRes::new(SharedDeviceState(inner.clone()))
            }
        } else {
            let res = create(None).await.map(|a| Arc::new(a));
            *this = Some(SharedDeviceStateHostContainer(res.inner().clone()));
            res.map(|a| SharedDeviceState(a))
        }
    }
}
*/
/// Returns (new, old)
pub fn use_shared_state_with_old<T>(
    this: &mut Option<SharedDeviceStateHostContainer<T>>,
    create: impl FnOnce(Option<&T>) -> T,
    should_update: impl FnOnce(&T) -> bool,
) -> (
    RenderRes<SharedDeviceState<T>>,
    Option<RenderRes<SharedDeviceState<T>>>,
) {
    if let Some(inner) = this {
        let inner = &mut inner.0;
        if should_update(&inner.item) {
            let old = std::mem::replace(
                inner,
                Arc::new(SharedDeviceStateInner {
                    item: create(Some(&inner.item)),
                    tracking_feedback: Default::default(),
                    fetched: AtomicBool::new(true),
                }),
            );
            let old =
                RenderRes::with_feedback(SharedDeviceState(old), inner.get_tracking_feedback());
            (RenderRes::new(SharedDeviceState(inner.clone())), Some(old))
        } else {
            inner
                .fetched
                .store(true, std::sync::atomic::Ordering::Relaxed);
            (
                RenderRes::with_feedback(
                    SharedDeviceState(inner.clone()),
                    inner.get_tracking_feedback(),
                ),
                None,
            )
        }
    } else {
        let item = Arc::new(SharedDeviceStateInner {
            item: create(None),
            tracking_feedback: Default::default(),
            fetched: AtomicBool::new(true),
        });
        *this = Some(SharedDeviceStateHostContainer(item));
        (
            RenderRes::new(SharedDeviceState(this.as_ref().unwrap().0.clone())),
            None,
        )
    }
}

use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;

use ash::vk;

use crate::BufferLike;
use crate::HasDevice;
use crate::ImageLike;
use crate::ImageViewLike;

use super::RenderData;
use super::RenderImage;
use super::RenderRes;
use super::TrackingFeedback;
// probably needs a mpsc channel.
pub struct PerFrameState<T> {
    receiver: crossbeam_channel::Receiver<T>,
    sender: crossbeam_channel::Sender<T>,

    /// Total number of items stored inside this container,
    /// including owned items and pending (taken) items.
    total_items: AtomicUsize,
}
impl<T> PerFrameState<T> {
    fn try_recv(&self) -> Option<T> {
        match self.receiver.try_recv() {
            Ok(item) => Some(item),
            Err(crossbeam_channel::TryRecvError::Empty) => None,
            Err(crossbeam_channel::TryRecvError::Disconnected) => panic!(),
        }
    }
    fn recv(&self) -> T {
        match self.receiver.recv() {
            Ok(item) => item,
            Err(crossbeam_channel::RecvError) => panic!(),
        }
    }
    pub fn use_state(&self, create: impl FnOnce() -> T) -> PerFrameContainer<T> {
        let (item, reused) = self.try_recv().map(|item| (item, true)).unwrap_or_else(|| {
            self.total_items
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            (create(), false)
        });
        PerFrameContainer {
            sender: self.sender.clone(),
            item: Some(item),
            reused,
        }
    }

    pub fn use_blocking(
        // &mut self ensures we can't run into a race condition where two threads call use_blocking at the same time
        // in which case we might exceed the max_items limitation
        &mut self,
        max_items: usize,
        create: impl FnOnce() -> T,
    ) -> PerFrameContainer<T> {
        if *self.total_items.get_mut() < max_items {
            self.use_state(create)
        } else {
            // When this branch was selected, we're acquiring images faster than we can process them.
            // This is usually due to a max_frame_in_flight number that is smaller than the swapchain image count.
            // When max_frame_in_flight >= swapchain image count, this branch usually shouldn't be triggered.
            let item = self.recv();

            PerFrameContainer {
                sender: self.sender.clone(),
                item: Some(item),
                reused: true,
            }
        }
    }
}
// Safety: We do not expose &self.receiver or &self.sender to the outside.
unsafe impl<T> Sync for PerFrameState<T> {}
impl<T> Default for PerFrameState<T> {
    fn default() -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded();
        Self {
            receiver,
            sender,
            total_items: AtomicUsize::new(0),
        }
    }
}
pub struct PerFrameContainer<T> {
    sender: crossbeam_channel::Sender<T>,
    item: Option<T>,
    reused: bool,
}
impl<T: RenderData> RenderData for PerFrameContainer<T> {}
impl<T> PerFrameContainer<T> {
    pub fn reuse(mut self, reuse: impl FnOnce(&mut T)) -> Self {
        if self.reused {
            reuse(&mut self);
        }
        self
    }
}
unsafe impl<T> Sync for PerFrameContainer<T> {}
impl<T> Deref for PerFrameContainer<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.item.as_ref().unwrap()
    }
}
impl<T> DerefMut for PerFrameContainer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item.as_mut().unwrap()
    }
}
impl<T> Drop for PerFrameContainer<T> {
    fn drop(&mut self) {
        self.sender.send(self.item.take().unwrap()).ok();
    }
}
impl<T: HasDevice> HasDevice for PerFrameContainer<T> {
    fn device(&self) -> &Arc<crate::Device> {
        self.item.as_ref().unwrap().device()
    }
}
impl<T: ImageLike> ImageLike for PerFrameContainer<T> {
    fn raw_image(&self) -> ash::vk::Image {
        self.item.as_ref().unwrap().raw_image()
    }

    fn subresource_range(&self) -> ash::vk::ImageSubresourceRange {
        self.item.as_ref().unwrap().subresource_range()
    }

    fn extent(&self) -> ash::vk::Extent3D {
        self.item.as_ref().unwrap().extent()
    }

    fn format(&self) -> ash::vk::Format {
        self.item.as_ref().unwrap().format()
    }
    fn offset(&self) -> ash::vk::Offset3D {
        self.item.as_ref().unwrap().offset()
    }
}
impl<T: BufferLike> BufferLike for PerFrameContainer<T> {
    fn raw_buffer(&self) -> ash::vk::Buffer {
        self.item.as_ref().unwrap().raw_buffer()
    }

    fn size(&self) -> ash::vk::DeviceSize {
        self.item.as_ref().unwrap().size()
    }
    fn offset(&self) -> ash::vk::DeviceSize {
        self.item.as_ref().unwrap().offset()
    }
    fn device_address(&self) -> ash::vk::DeviceAddress {
        self.item.as_ref().unwrap().device_address()
    }
}
impl<T: ImageViewLike> ImageViewLike for PerFrameContainer<T> {
    fn raw_image_view(&self) -> ash::vk::ImageView {
        self.item.as_ref().unwrap().raw_image_view()
    }
}

pub fn use_per_frame_state<T>(
    this: &mut PerFrameState<T>,
    create: impl FnOnce() -> T,
) -> PerFrameContainer<T> {
    this.use_state(create)
}
pub fn use_per_frame_state_blocking<T>(
    this: &mut PerFrameState<T>,
    max_items: usize,
    create: impl FnOnce() -> T,
) -> PerFrameContainer<T> {
    this.use_blocking(max_items, create)
}
