use std::{
    collections::BTreeMap,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{atomic::AtomicU64, Arc},
};

use crate::{
    swapchain::SwapchainImage, sync::TimelineSemaphore, Device, HasDevice, QueueConfiguration,
    QueueInner, QueueSelector,
};
use ash::{
    prelude::VkResult,
    vk::{self},
};
use bevy::{
    ecs::{component::ComponentId, system::SystemParam},
    prelude::{FromWorld, Mut, Resource, World},
};

#[derive(Resource)]
pub struct CommandPool {
    device: Device,
    pub(crate) raw: vk::CommandPool,
    queue_family_index: u32,
    flags: vk::CommandPoolCreateFlags,
    pub(crate) generation: u64,

    semaphore_signals: BTreeMap<vk::CommandBuffer, (Arc<TimelineSemaphore>, u64)>,

    semaphore_signal_raws: Vec<vk::Semaphore>,
    semaphore_signal_values: Vec<u64>,
}
impl HasDevice for CommandPool {
    fn device(&self) -> &Device {
        &self.device
    }
}
struct CommandPoolTimeline {
    semaphore: Arc<TimelineSemaphore>,
    index: usize,
}
impl Drop for CommandPool {
    fn drop(&mut self) {
        // Wait for semaphore signals to ensure that there's no pending command buffers.
        self.wait_for_completion();
        unsafe {
            self.device.destroy_command_pool(self.raw, None);
        }
    }
}

/// ## Usages
/// ### Reset the whole command pool
/// ```rust
/// use ash::vk;
/// let device = rhyolite::create_system_default_device(unsafe { ash::Entry::load().unwrap() });
/// let command_pool = rhyolite::command::CommandPool::new(device.clone(), 0, vk::CommandPoolCreateFlags::TRANSIENT);
///
/// let timeline = rhyolite::command::Timeline::new(device.clone());
/// ```
impl CommandPool {
    fn wait_for_completion(&mut self) {
        if self.semaphore_signals.is_empty() {
            return;
        }
        self.semaphore_signal_raws
            .extend(self.semaphore_signals.values().map(|x| x.0.raw()));
        self.semaphore_signal_values
            .extend(self.semaphore_signals.values().map(|x| x.1));
        unsafe {
            self.device
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&self.semaphore_signal_raws)
                        .values(&self.semaphore_signal_values),
                    !0,
                )
                .unwrap();
        }
        self.semaphore_signal_raws.clear();
        self.semaphore_signal_values.clear();
        self.semaphore_signals.clear();
    }
    pub fn new(
        device: Device,
        queue_family_index: u32,
        flags: vk::CommandPoolCreateFlags,
    ) -> VkResult<Self> {
        unsafe {
            let raw = device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    queue_family_index,
                    flags,
                    ..Default::default()
                },
                None,
            )?;
            Ok(Self {
                raw,
                device,
                queue_family_index,
                flags,
                generation: 0,
                semaphore_signals: BTreeMap::new(),
                semaphore_signal_raws: Vec::new(),
                semaphore_signal_values: Vec::new(),
            })
        }
    }

    /// Wait for all outstanding command buffers to complete. Then, reset the command pool.
    ///
    /// Safety:
    /// ## VUID-vkResetCommandPool-commandPool-00040
    /// All VkCommandBuffer objects allocated from commandPool must not be in the pending state
    ///
    /// We enforce this by awaiting on the timeline semaphores signaled.
    pub fn reset_pool_blocked(&mut self, release_resources: bool) {
        let reset_flags = if release_resources {
            vk::CommandPoolResetFlags::RELEASE_RESOURCES
        } else {
            vk::CommandPoolResetFlags::empty()
        };
        self.wait_for_completion();
        unsafe {
            self.device
                .reset_command_pool(self.raw, reset_flags)
                .unwrap();
        }
        self.generation += 1;
    }
    pub fn end(
        &mut self,
        command_buffer: CommandBuffer<states::Recording>,
    ) -> CommandBuffer<states::Executable> {
        assert_eq!(command_buffer.pool, self.raw);
        assert_eq!(command_buffer.generation, self.generation, "Command pool has already been reset, and this command buffer is now invaild. Call `update` to restart encoding.");

        self.semaphore_signals.insert(
            command_buffer.raw,
            (
                command_buffer.timeline_semaphore.clone(),
                command_buffer.signal_value,
            ),
        );
        unsafe {
            self.device.end_command_buffer(command_buffer.raw).unwrap();
            command_buffer.state_transition(states::Executable)
        }
    }
    pub unsafe fn allocate_raw(&mut self) -> VkResult<vk::CommandBuffer> {
        let mut buffer = vk::CommandBuffer::null();
        (self.device.fp_v1_0().allocate_command_buffers)(
            self.device.handle(),
            &vk::CommandBufferAllocateInfo {
                command_pool: self.raw,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            },
            &mut buffer,
        )
        .result()?;
        Ok(buffer)
    }
    pub fn allocate(
        &mut self,
        on_timeline: &Timeline,
        flags: vk::CommandBufferUsageFlags,
    ) -> VkResult<CommandBuffer<states::Recording>> {
        unsafe {
            let raw = self.allocate_raw()?;
            self.device.begin_command_buffer(
                raw,
                &vk::CommandBufferBeginInfo {
                    flags,
                    ..Default::default()
                },
            )?;

            let command_buffer = CommandBuffer {
                raw,
                pool: self.raw,
                flags,
                queue_family_index: self.queue_family_index,
                timeline_semaphore: on_timeline.semaphore.clone(),
                signal_value: on_timeline.wait_value() + 1,
                generation: self.generation,
                _marker: PhantomData,
            };
            Ok(command_buffer)
        }
    }

    /// This is used in combination with [`CommandPool::reset_blocked`].
    /// We enforce that [`CommandPool::reset_blocked`] has been called by checking the generation.
    pub fn update<T>(
        &mut self,
        mut command_buffer: CommandBuffer<T>,
        on_timeline: &Timeline,
        flags: vk::CommandBufferUsageFlags,
    ) -> CommandBuffer<states::Recording> {
        assert!(
            self.generation > command_buffer.generation,
            "Must reset the command pool before updating individual command buffers"
        );
        assert_eq!(command_buffer.pool, self.raw);
        command_buffer.generation = self.generation;
        if !Arc::ptr_eq(&on_timeline.semaphore, &command_buffer.timeline_semaphore) {
            command_buffer.timeline_semaphore = on_timeline.semaphore.clone();
        }
        command_buffer.signal_value = on_timeline.increment() + 1;

        unsafe {
            self.device
                .begin_command_buffer(
                    command_buffer.raw,
                    &vk::CommandBufferBeginInfo {
                        flags,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        command_buffer.state_transition(states::Recording)
    }
    pub fn free<T: states::NonPending>(&mut self, buffer: CommandBuffer<T>) {
        self.try_remove_semaphore(&buffer);
        assert_eq!(buffer.pool, self.raw);
        unsafe {
            self.device.free_command_buffers(self.raw, &[buffer.raw]);
        }
        buffer.force_drop();
    }
    pub fn free_many<T: states::NonPending>(
        &mut self,
        buffers: impl Iterator<Item = CommandBuffer<T>>,
    ) {
        let handles: Vec<_> = buffers
            .map(|x| {
                assert_eq!(x.pool, self.raw);

                self.try_remove_semaphore(&x);

                let buf = x.raw;

                x.force_drop();
                buf
            })
            .collect();
        unsafe {
            self.device.free_command_buffers(self.raw, &handles);
        }
    }
    fn try_remove_semaphore<T: states::NonPending>(&mut self, command_buffer: &CommandBuffer<T>) {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<states::Completed>()
            || std::any::TypeId::of::<T>() == std::any::TypeId::of::<states::Executable>()
        {
            if self.generation == command_buffer.generation {
                self.semaphore_signals.remove(&command_buffer.raw).unwrap();
            }
            // if generation is not the same, the command pool has already been resetted. no need to remove semaphore in this case.
        }
    }

    pub unsafe fn reset_command_buffer_raw(
        &mut self,
        command_buffer: vk::CommandBuffer,
        release_resources: bool,
    ) -> VkResult<()> {
        let reset_flags = if release_resources {
            vk::CommandBufferResetFlags::RELEASE_RESOURCES
        } else {
            vk::CommandBufferResetFlags::empty()
        };
        self.device
            .reset_command_buffer(command_buffer, reset_flags)
    }
    pub fn reset_command_buffer<T: states::NonPending>(
        &mut self,
        mut command_buffer: CommandBuffer<T>,
        release_resources: bool,
        next_timeline: &Timeline,
    ) -> CommandBuffer<states::Recording> {
        assert!(self
            .flags
            .contains(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER));
        self.try_remove_semaphore(&command_buffer);
        assert_eq!(command_buffer.pool, self.raw);
        if !Arc::ptr_eq(&next_timeline.semaphore, &command_buffer.timeline_semaphore) {
            command_buffer.timeline_semaphore = next_timeline.semaphore.clone();
        }
        command_buffer.signal_value = next_timeline.wait_value() + 1;
        command_buffer.generation = self.generation;
        unsafe {
            self.reset_command_buffer_raw(command_buffer.raw, release_resources)
                .unwrap();
            self.device
                .begin_command_buffer(
                    command_buffer.raw,
                    &vk::CommandBufferBeginInfo {
                        flags: command_buffer.flags,
                        ..Default::default()
                    },
                )
                .unwrap();
            command_buffer.state_transition(states::Recording)
        }
    }
}

pub mod states {
    pub trait NonPending: 'static {}

    pub struct Recording;
    pub struct Pending;
    pub struct Executable;
    pub struct Completed;
    impl NonPending for Recording {}
    impl NonPending for Executable {}
    impl NonPending for Completed {}
}
pub struct CommandBuffer<STATE: 'static> {
    pub(crate) raw: vk::CommandBuffer,
    pub(crate) pool: vk::CommandPool,
    queue_family_index: u32,
    flags: vk::CommandBufferUsageFlags,
    pub(crate) timeline_semaphore: Arc<TimelineSemaphore>,
    // When the command buffer was executed on the GPU, this shall be signaled
    pub(crate) signal_value: u64,

    /// The generation of the command pool when this was last allocated / reset
    pub(crate) generation: u64,
    _marker: PhantomData<STATE>,
}
impl<STATE> CommandBuffer<STATE> {
    fn state_transition<NEXT>(self, _next: NEXT) -> CommandBuffer<NEXT> {
        unsafe { std::mem::transmute(self) }
    }
    fn force_drop(self) {
        self.state_transition(()); // this should be free to drop
    }
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }
}
impl CommandBuffer<states::Pending> {
    pub fn wait_for_completion(self) -> CommandBuffer<states::Completed> {
        self.timeline_semaphore
            .wait_blocked(self.signal_value, !0)
            .unwrap();
        self.state_transition(states::Completed)
    }
}
impl<T: 'static> Drop for CommandBuffer<T> {
    fn drop(&mut self) {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<states::Executable>() {
            self.timeline_semaphore.signal(self.signal_value);
            tracing::warn!("CommandBuffer dropped without being submitted");
        }
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<states::Recording>()
            || std::any::TypeId::of::<T>() == std::any::TypeId::of::<states::Completed>()
        {
            tracing::warn!("CommandBuffer dropped without being freed");
        }
        // Notably, we do not show a warning if the command buffer is pending.
        // It's very common to drop pending command buffers when the application shuts down.
        // As long as we wait for all command buffers to complete before destroying the command pool, we're good.
    }
}
// How do we handle simutaneous use and one time submit?

// Each queue system will have one of this.
// It gets incremented during queue submit.
#[derive(Debug)]
pub struct Timeline {
    pub(crate) semaphore: Arc<TimelineSemaphore>,
    wait_value: AtomicU64,
}
impl FromWorld for Timeline {
    fn from_world(world: &mut World) -> Self {
        Self {
            semaphore: Arc::new(
                TimelineSemaphore::new(world.resource::<Device>().clone(), 0).unwrap(),
            ),
            wait_value: AtomicU64::new(0),
        }
    }
}
impl Timeline {
    pub fn new(device: Device) -> VkResult<Self> {
        Ok(Self {
            semaphore: Arc::new(TimelineSemaphore::new(device, 0)?),
            wait_value: AtomicU64::new(0),
        })
    }
    pub fn increment(&self) -> u64 {
        self.wait_value
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
    pub fn wait_value(&self) -> u64 {
        self.wait_value.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn wait_blocked(&self, timeout: u64) -> VkResult<()> {
        self.semaphore.wait_blocked(self.wait_value(), timeout)
    }
}

#[repr(transparent)]
pub struct QueueDependency<'a>(pub(crate) vk::SemaphoreSubmitInfo<'a>);

impl QueueInner {
    pub fn submit_one(
        &mut self,
        command_buffer: CommandBuffer<states::Executable>,
        dependencies: &[QueueDependency],
    ) -> VkResult<CommandBuffer<states::Pending>> {
        // TODO: emit syncronization barrier for command buffer future ctx.
        unsafe {
            self.device.queue_submit2(
                self.queue,
                &[vk::SubmitInfo2 {
                    ..Default::default()
                }
                .command_buffer_infos(&[vk::CommandBufferSubmitInfo {
                    command_buffer: command_buffer.raw,
                    ..Default::default()
                }])
                .wait_semaphore_infos(std::mem::transmute(dependencies))
                .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo {
                    semaphore: command_buffer.timeline_semaphore.raw(),
                    value: command_buffer.signal_value,
                    // We signal on ALL_COMMANDS because
                    // 1. Timeline semaphore.
                    // 2. Most impl probably cannot take advantage of any other flags.
                    stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                    ..Default::default()
                }])],
                vk::Fence::null(),
            )?;
        }
        Ok(command_buffer.state_transition(states::Pending))
    }

    pub fn submit_one_and_present(
        &mut self,
        command_buffer: CommandBuffer<states::Executable>,
        dependencies: &[QueueDependency],
        swapchain_image: &SwapchainImage,
    ) -> VkResult<CommandBuffer<states::Pending>> {
        // TODO: emit syncronization barrier for command buffer future ctx.

        unsafe {
            self.device.queue_submit2(
                self.queue,
                &[vk::SubmitInfo2 {
                    ..Default::default()
                }
                .command_buffer_infos(&[vk::CommandBufferSubmitInfo {
                    command_buffer: command_buffer.raw,
                    ..Default::default()
                }])
                .wait_semaphore_infos(std::mem::transmute(dependencies))
                .signal_semaphore_infos(&[
                    vk::SemaphoreSubmitInfo {
                        semaphore: command_buffer.timeline_semaphore.raw(),
                        value: command_buffer.signal_value,
                        // We signal on ALL_COMMANDS because
                        // 1. Timeline semaphore.
                        // 2. Most impl probably cannot take advantage of any other flags.
                        stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                        ..Default::default()
                    },
                    vk::SemaphoreSubmitInfo {
                        semaphore: swapchain_image.inner.as_ref().unwrap().present_semaphore,
                        value: 0,
                        stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                        ..Default::default()
                    },
                ])],
                vk::Fence::null(),
            )?;
        }
        Ok(command_buffer.state_transition(states::Pending))
    }
}

// TODO: for buffers without one time submit, allow one to retarget them onto a different timeline.
// TODO: for buffers with simutaneous use, allow one to submit simutaneously.

/// Shared command pool. One for each queue family.
/// Good for small amount of command encoding. Use parallel encoding if an incredibly large amount of
/// commands need to be encoded.
pub struct SharedCommandPool<'a, Q: QueueSelector> {
    command_pool: Mut<'a, CommandPool>,
    _marker: PhantomData<Q>,
}
impl<'a, Q: QueueSelector> Deref for SharedCommandPool<'a, Q> {
    type Target = CommandPool;
    fn deref(&self) -> &Self::Target {
        &*self.command_pool
    }
}
impl<'a, Q: QueueSelector> DerefMut for SharedCommandPool<'a, Q> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.command_pool
    }
}

unsafe impl<'a, Q: QueueSelector> SystemParam for SharedCommandPool<'a, Q> {
    type State = ComponentId;

    type Item<'world, 'state> = SharedCommandPool<'world, Q>;

    fn init_state(
        world: &mut bevy::prelude::World,
        system_meta: &mut bevy::ecs::system::SystemMeta,
    ) -> Self::State {
        let config = world.resource::<QueueConfiguration>();
        let component_id = Q::shared_command_pool_component_id(config);

        let combined_access = system_meta.component_access_set().combined_access();
        if combined_access.has_resource_write(component_id) {
            panic!(
                "error[B0002]: ResMut<{}> in system {} conflicts with a previous ResMut<{0}> access. Consider removing the duplicate access. See: https://bevyengine.org/learn/errors/#b0002",
                std::any::type_name::<Self>(), system_meta.name());
        } else if combined_access.has_resource_read(component_id) {
            panic!(
                "error[B0002]: ResMut<{}> in system {} conflicts with a previous Res<{0}> access. Consider removing the duplicate access. See: https://bevyengine.org/learn/errors/#b0002",
                std::any::type_name::<Self>(), system_meta.name());
        }
        unsafe {
            system_meta
                .component_access_set_mut()
                .add_unfiltered_resource_write(component_id);

            let archetype_component_id = world.storages().resources.get(component_id).unwrap().id();
            system_meta
                .archetype_component_access_mut()
                .add_resource_write(archetype_component_id);
        }
        component_id
    }

    unsafe fn get_param<'world, 'state>(
        component_id: &'state mut Self::State,
        system_meta: &bevy::ecs::system::SystemMeta,
        world: bevy::ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        _change_tick: bevy::ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        let value = world
            .get_resource_mut_by_id(*component_id)
            .unwrap_or_else(|| {
                panic!(
                    "Resource requested by {} does not exist: {}",
                    system_meta.name(),
                    std::any::type_name::<Q>()
                )
            });
        SharedCommandPool {
            command_pool: value.with_type::<CommandPool>(),
            _marker: PhantomData,
        }
    }
}
