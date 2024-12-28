use std::{
    any::Any,
    borrow::Cow,
    ops::{Deref, DerefMut},
    pin::Pin,
    ptr::NonNull,
    sync::Arc,
    usize,
};

use crate::{
    command::{states, CommandBuffer, CommandPool, QueueDependency, Timeline},
    future::{GPUFutureBlock, GPUFutureBlockReturnValue, GPUFutureContext},
    utils::RingBuffer,
    Device, HasDevice, QueueConfiguration, QueueInner, QueueSelector,
};
use ash::{prelude::VkResult, vk};
use bevy::{
    ecs::{
        archetype::ArchetypeComponentId,
        component::{ComponentId, Tick},
        query::Access,
        schedule::InternedSystemSet,
        system::{System, SystemMeta, SystemParam},
        world::{unsafe_world_cell::UnsafeWorldCell, DeferredWorld},
    },
    prelude::{In, IntoSystem, Mut, Resource, SystemInput, World},
};

#[derive(Clone, Debug)]
pub struct TimelineDependencies {
    pub this: Arc<Timeline>,
    pub dependencies: Vec<(Arc<Timeline>, vk::PipelineStageFlags2)>,
}
impl Drop for TimelineDependencies {
    fn drop(&mut self) {
        // Whenever there's a dependency relationship between multiple semaphores,
        // wait for the work to be completed before dropping the dependency semaphores.
        self.this.wait_blocked(!0).unwrap();
    }
}

/// Used as In<RenderSystemCtx<Returned>> for render systems.
pub struct RenderSystemCtx<Returned = ()> {
    returned_value: Option<Returned>,
}
impl<Returned> SystemInput for RenderSystemCtx<Returned> {
    type Param<'i> = RenderSystemCtx<Returned>;
    type Inner<'i> = RenderSystemCtx<Returned>;

    fn wrap(this: Self::Inner<'_>) -> Self::Param<'_> {
        this
    }
}
impl<Returned> RenderSystemCtx<Returned> {
    pub fn take(&mut self) -> Option<Returned> {
        self.returned_value.take()
    }
}
struct RenderSystemFrame<Returned, Retained> {
    returned: Returned,
    retained: Retained,
}
/// A wrapper for a system that records command buffers.
pub struct RenderSystem<
    F: GPUFutureBlock + Send + Sync,
    T: bevy::ecs::system::System<In = RenderSystemCtx<F::Returned>, Out = F>,
> {
    inner: T,
    future: Option<F>,
    frames: RingBuffer<RenderSystemFrame<F::Returned, F::Retained>, 3>,
    shared_state_component_id: ComponentId,

    /// If `queue_selector` is None, this must be valid upon initialization.
    queue_component_id: ComponentId,

    /// If non-None, `queue_component_id` will be null upon initialization, and this function
    /// will be called to figure out the queue dynamically.
    queue_selector: Option<fn(&QueueConfiguration) -> ComponentId>,

    component_access: Access<ComponentId>,
    archetype_component_access: Access<ArchetypeComponentId>,
}

/// Shared across all render systems within a single submission.
#[derive(Resource)]
pub(super) struct RenderSystemSharedState {
    /// The submission will take the command buffer and push it here.
    /// The prelude system is responsible for popping command buffers from this queue and await on them.
    pending_command_buffers: RingBuffer<CommandBuffer<states::Pending>, 3>,

    /// The prelude system will allocate and populate this command buffer.
    recording_command_buffer: Option<CommandBuffer<states::Recording>>,
    ctx: GPUFutureContext,
    timeline: Arc<Timeline>,
    command_pool: CommandPool,
}
impl RenderSystemSharedState {
    pub(super) fn new(device: Device, queue_family_index: u32, timeline: Arc<Timeline>) -> Self {
        Self {
            ctx: GPUFutureContext::new(
                device.clone(),
                vk::CommandBuffer::null(),
                queue_family_index,
            ),
            timeline,
            recording_command_buffer: None,
            pending_command_buffers: RingBuffer::new(),
            command_pool: CommandPool::new(
                device,
                queue_family_index,
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                    | vk::CommandPoolCreateFlags::TRANSIENT,
            )
            .unwrap(),
        }
    }
}

pub(super) struct RenderSystemIdentifierConfig {
    pub(super) queue_component_id: ComponentId,
    pub(super) is_standalone: bool,
}
pub(super) struct RenderSystemInputConfig {
    pub(super) shared_state_component_id: ComponentId,
    pub(super) shared_state_archetype_component_id: ArchetypeComponentId,
    pub(super) queue: ComponentId,
}

// command buffer ownership.
// obviously in the simple case, ownership is easy. eveerything owned by the RenderSystemSharedState.
// we also want command buffer affinity. so command buffers should return at a predictable fasion.

impl<
        F: GPUFutureBlock + Send + Sync + 'static,
        T: bevy::ecs::system::System<In = RenderSystemCtx<F::Returned>, Out = F>,
    > bevy::ecs::system::System for RenderSystem<F, T>
where
    F::Retained: Send + Sync,
    F::Returned: Send + Sync,
{
    type In = ();

    type Out = ();

    fn default_system_sets(&self) -> Vec<InternedSystemSet> {
        self.inner.default_system_sets()
    }

    fn configurate(&mut self, config: &mut dyn Any) {
        if let Some(config) = config.downcast_mut::<Option<RenderSystemIdentifierConfig>>() {
            assert!(config.is_none());
            *config = Some(RenderSystemIdentifierConfig {
                queue_component_id: self.queue_component_id,
                is_standalone: false,
            });
            return;
        }

        let Some(config) = config.downcast_mut::<RenderSystemInputConfig>() else {
            return;
        };
        self.shared_state_component_id = config.shared_state_component_id;
        assert_eq!(config.queue, self.queue_component_id);
        {
            // Add component access for the shared state
            self.component_access
                .add_resource_write(config.shared_state_component_id);
            self.archetype_component_access
                .add_resource_write(config.shared_state_archetype_component_id);
        }
    }

    fn name(&self) -> Cow<'static, str> {
        self.inner.name()
    }

    fn component_access(&self) -> &Access<ComponentId> {
        &self.component_access
    }

    fn archetype_component_access(&self) -> &Access<ArchetypeComponentId> {
        &self.archetype_component_access
    }

    fn is_send(&self) -> bool {
        self.inner.is_send()
    }

    fn is_exclusive(&self) -> bool {
        self.inner.is_exclusive()
    }

    fn has_deferred(&self) -> bool {
        self.inner.has_deferred()
    }

    unsafe fn run_unsafe(&mut self, _: Self::In, world: UnsafeWorldCell) -> Self::Out {
        if self.future.is_none() {
            // This is the first time that this has run this frame.
            let frame = self.frames.pop_if_full();
            let returned_value = frame.map(|frame| {
                // (We don't need to wait for anything anymore, since the leading system already does that for us?)
                drop(frame.retained);
                frame.returned
            });
            let ctx = RenderSystemCtx { returned_value };

            self.future = Some(self.inner.run_unsafe(ctx, world));
        }
        let pinned_future = unsafe {
            // Safety: This assumes that the system will be boxed up eventually.
            Pin::new_unchecked(self.future.as_mut().unwrap())
        };
        let mut shared_state = world
            .get_resource_mut_by_id(self.shared_state_component_id)
            .unwrap()
            .with_type::<RenderSystemSharedState>();

        // `shared_state.recording_command_buffer` should be populated by the prelude system and taken by the submisison system.
        shared_state.ctx.command_buffer =
            shared_state.recording_command_buffer.as_mut().unwrap().raw;

        match crate::future::gpu_future_poll(pinned_future, &mut shared_state.ctx) {
            std::task::Poll::Ready(GPUFutureBlockReturnValue {
                retained_values,
                output,
            }) => {
                self.frames.push(RenderSystemFrame {
                    returned: output,
                    retained: retained_values,
                });
                self.future = None;
            }
            std::task::Poll::Pending => {
                shared_state.command_pool.device().cmd_pipeline_barrier2(
                    shared_state.ctx.command_buffer,
                    &vk::DependencyInfo::default()
                        .image_memory_barriers(&shared_state.ctx.image_barrier)
                        .memory_barriers(&[shared_state.ctx.memory_barrier]),
                );
                shared_state.ctx.image_barrier.clear();
                shared_state.ctx.memory_barrier = Default::default();
            }
        }
    }

    fn yielded(&self) -> bool {
        self.future.is_some()
    }

    fn apply_deferred(&mut self, world: &mut World) {
        self.inner.apply_deferred(world);
    }

    fn queue_deferred(&mut self, world: DeferredWorld) {
        self.inner.queue_deferred(world);
    }

    fn initialize(&mut self, world: &mut World) {
        if let Some(selector) = self.queue_selector.take() {
            assert_eq!(self.queue_component_id.index(), usize::MAX);
            let queue_config: &QueueConfiguration = world.resource();
            self.queue_component_id = selector(queue_config);
        } else {
            assert_ne!(self.queue_component_id.index(), usize::MAX);
        }

        self.inner.initialize(world);

        self.component_access.extend(self.inner.component_access());
    }

    fn update_archetype_component_access(&mut self, world: UnsafeWorldCell) {
        self.inner.update_archetype_component_access(world);

        self.archetype_component_access
            .extend(self.inner.archetype_component_access());
    }

    fn check_change_tick(&mut self, change_tick: Tick) {
        self.inner.check_change_tick(change_tick);
    }

    fn get_last_run(&self) -> Tick {
        self.inner.get_last_run()
    }

    fn set_last_run(&mut self, last_run: Tick) {
        self.inner.set_last_run(last_run);
    }

    unsafe fn validate_param_unsafe(&mut self, world: UnsafeWorldCell) -> bool {
        self.inner.validate_param_unsafe(world)
    }
}

/// Used by the prelude system and submission system to access the render systems shared state
pub(super) struct RenderSystemSharedStateSystemParam<'w>(Mut<'w, RenderSystemSharedState>);
impl<'w> Deref for RenderSystemSharedStateSystemParam<'w> {
    type Target = RenderSystemSharedState;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}
impl<'w> DerefMut for RenderSystemSharedStateSystemParam<'w> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.deref_mut()
    }
}
unsafe impl<'w> SystemParam for RenderSystemSharedStateSystemParam<'w> {
    type State = ComponentId;

    type Item<'world, 'state> = RenderSystemSharedStateSystemParam<'world>;

    fn init_state(_world: &mut World, _system_meta: &mut SystemMeta) -> Self::State {
        ComponentId::new(usize::MAX)
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        _system_meta: &SystemMeta,
        world: UnsafeWorldCell<'world>,
        _change_tick: Tick,
    ) -> Self::Item<'world, 'state> {
        let shared_state = world
            .get_resource_mut_by_id(*state)
            .unwrap()
            .with_type::<RenderSystemSharedState>();
        RenderSystemSharedStateSystemParam(shared_state)
    }
}

/// A wrapper for a system that submits queue operations.
pub struct QueueSystem<T: bevy::ecs::system::System<In = QueueSystemCtx, Out = ()>> {
    inner: T,
    timeline_dependency: Option<TimelineDependencies>,
    component_access: Access<ComponentId>,
    archetype_component_access: Access<ArchetypeComponentId>,
    queue_component_id: ComponentId,
    queue_selector: Option<fn(&QueueConfiguration) -> ComponentId>,
}
pub struct QueueSystemCtx {
    queue: NonNull<QueueInner>,
    dependencies: *const TimelineDependencies,
}
impl SystemInput for QueueSystemCtx {
    type Param<'i> = QueueSystemCtx;
    type Inner<'i> = QueueSystemCtx;

    fn wrap(this: Self::Inner<'_>) -> Self::Param<'_> {
        this
    }
}
impl QueueSystemCtx {
    pub fn family_index(&self) -> u32 {
        let queue_inner = unsafe { self.queue.as_ref() };
        queue_inner.queue_family
    }
    pub fn dependencies(&self) -> &TimelineDependencies {
        unsafe { &*self.dependencies }
    }
    pub fn raw_queue(&self) -> vk::Queue {
        let queue_inner = unsafe { self.queue.as_ref() };
        queue_inner.queue
    }
    pub fn submit_one(
        &mut self,
        command_buffer: CommandBuffer<states::Executable>,
    ) -> VkResult<CommandBuffer<states::Pending>> {
        let queue_inner = unsafe { self.queue.as_mut() };
        let dependencies = unsafe { &*self.dependencies };
        let mut waits = dependencies
            .dependencies
            .iter()
            .map(|(semaphore, stages)| {
                QueueDependency(vk::SemaphoreSubmitInfo {
                    semaphore: semaphore.semaphore.raw(),
                    value: semaphore.wait_value(),
                    stage_mask: *stages,
                    _marker: std::marker::PhantomData,
                    ..Default::default()
                })
            })
            .collect::<Vec<_>>();
        assert!(Arc::ptr_eq(
            &command_buffer.timeline_semaphore,
            &dependencies.this.semaphore
        ));
        waits.push(QueueDependency(vk::SemaphoreSubmitInfo {
            semaphore: dependencies.this.semaphore.raw(),
            value: dependencies.this.wait_value(),
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            _marker: std::marker::PhantomData,
            ..Default::default()
        }));
        queue_inner.submit_one(command_buffer, &waits)
    }
}
impl<T: bevy::ecs::system::System<In = QueueSystemCtx, Out = ()>> System for QueueSystem<T> {
    type In = ();

    type Out = ();

    fn default_system_sets(&self) -> Vec<InternedSystemSet> {
        self.inner.default_system_sets()
    }

    fn name(&self) -> Cow<'static, str> {
        self.inner.name()
    }

    fn component_access(&self) -> &Access<ComponentId> {
        &self.component_access
    }

    fn archetype_component_access(&self) -> &Access<ArchetypeComponentId> {
        &self.archetype_component_access
    }

    fn is_send(&self) -> bool {
        self.inner.is_send()
    }

    fn is_exclusive(&self) -> bool {
        self.inner.is_exclusive()
    }

    fn has_deferred(&self) -> bool {
        self.inner.has_deferred()
    }

    unsafe fn run_unsafe(&mut self, _input: (), world: UnsafeWorldCell) -> Self::Out {
        let input = QueueSystemCtx {
            queue: unsafe {
                NonNull::new_unchecked(
                    &mut *world
                        .get_resource_mut_by_id(self.queue_component_id)
                        .unwrap()
                        .with_type(),
                )
            },
            dependencies: self.timeline_dependency.as_ref().unwrap(),
        };
        self.inner.run_unsafe(input, world)
    }

    fn apply_deferred(&mut self, world: &mut World) {
        self.inner.apply_deferred(world);
    }

    fn queue_deferred(&mut self, world: DeferredWorld) {
        self.inner.queue_deferred(world);
    }

    fn initialize(&mut self, world: &mut World) {
        if let Some(selector) = self.queue_selector.take() {
            assert_eq!(self.queue_component_id.index(), usize::MAX);
            let queue_config: &QueueConfiguration = world.resource();
            self.queue_component_id = selector(queue_config);
        } else {
            assert_ne!(self.queue_component_id.index(), usize::MAX);
        }
        {
            // Add component access for the queue
            self.component_access
                .add_resource_write(self.queue_component_id);
            let archetype_component_id = world
                .storages()
                .resources
                .get(self.queue_component_id)
                .unwrap()
                .id();
            self.archetype_component_access
                .add_resource_write(archetype_component_id);
        }

        self.inner.initialize(world);

        self.component_access.extend(self.inner.component_access());
    }

    fn update_archetype_component_access(&mut self, world: UnsafeWorldCell) {
        self.inner.update_archetype_component_access(world);

        self.archetype_component_access
            .extend(self.inner.archetype_component_access());
    }

    fn check_change_tick(&mut self, change_tick: Tick) {
        self.inner.check_change_tick(change_tick);
    }

    fn get_last_run(&self) -> Tick {
        self.inner.get_last_run()
    }

    fn set_last_run(&mut self, last_run: Tick) {
        self.inner.set_last_run(last_run);
    }
    fn configurate(&mut self, config: &mut dyn Any) {
        if let Some(config) = config.downcast_mut::<Option<RenderSystemIdentifierConfig>>() {
            assert!(config.is_none());
            *config = Some(RenderSystemIdentifierConfig {
                queue_component_id: self.queue_component_id,
                is_standalone: true,
            });
            return;
        }

        if let Some(config) = config.downcast_mut::<TimelineDependencies>() {
            self.timeline_dependency = Some(config.clone());
            return;
        }
        self.inner.configurate(config); // So that the wrapped system may have `RenderSystemInputConfig`
    }

    unsafe fn validate_param_unsafe(&mut self, world: UnsafeWorldCell) -> bool {
        self.inner.validate_param_unsafe(world)
    }
}

/// Prelude system runs before every render system in the queue node.
/// It shall:
/// - Allocate, or reuse and reset command buffer
/// - Wait for command buffer completion
pub(super) fn prelude_system(mut shared: RenderSystemSharedStateSystemParam) {
    let shared = &mut *shared;
    assert!(shared.recording_command_buffer.is_none());
    if let Some(reused_command_buffer) = shared.pending_command_buffers.pop_if_full() {
        let reused_command_buffer = reused_command_buffer.wait_for_completion();
        let reused_command_buffer = shared.command_pool.reset_command_buffer(
            reused_command_buffer,
            false,
            &shared.timeline,
        );
        shared.recording_command_buffer = Some(reused_command_buffer);
    } else {
        let reused_command_buffer = shared
            .command_pool
            .allocate(
                &shared.timeline,
                vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            )
            .unwrap();
        shared.recording_command_buffer = Some(reused_command_buffer);
    }
}

/// Submission system runs after every render system in the queue node. It also runs after every render system in the next queue node.
pub(super) fn submission_system(
    mut queue: QueueSystemCtx,
    mut shared: RenderSystemSharedStateSystemParam,
    //queue_submission_ctx: (), // this gives you the semaphores from the schedule build pass and identify the system as a queue system.
) {
    let command_buffer = shared.recording_command_buffer.take().unwrap();
    let command_buffer = shared.command_pool.end(command_buffer);
    let command_buffer = queue.submit_one(command_buffer).unwrap();
    shared.pending_command_buffers.push(command_buffer);
    queue.dependencies().this.increment();
}

/// An extension trait for [`IntoSystem`] allowing the user to turn a system that returns a GPUFuture into
/// a regular system that can be added to the App.
pub trait IntoRenderSystem<Out, Marker> {
    fn into_render_system<Q: QueueSelector>(self) -> impl System<In = (), Out = ()>;
    fn with_queue(self, queue: ComponentId) -> impl System<In = (), Out = ()>;
}

pub struct MarkerA;
impl<Out: GPUFutureBlock, Marker, T: IntoSystem<RenderSystemCtx<Out::Returned>, Out, Marker>>
    IntoRenderSystem<Out, (Marker, MarkerA)> for T
where
    T: Send + Sync + 'static,
    Out: Send + Sync + 'static,
{
    fn with_queue(self, queue: ComponentId) -> impl System<In = (), Out = ()> {
        RenderSystem::<Out, T::System> {
            inner: IntoSystem::into_system(self),
            future: None,
            frames: RingBuffer::new(),
            shared_state_component_id: ComponentId::new(usize::MAX),
            queue_component_id: queue,
            queue_selector: None,
            component_access: Default::default(),
            archetype_component_access: Default::default(),
        }
    }
    fn into_render_system<Q: QueueSelector>(self) -> impl System<In = (), Out = ()> {
        RenderSystem::<Out, T::System> {
            inner: IntoSystem::into_system(self),
            future: None,
            frames: RingBuffer::new(),
            shared_state_component_id: ComponentId::new(usize::MAX),
            queue_component_id: ComponentId::new(usize::MAX),
            queue_selector: Some(Q::component_id),
            component_access: Default::default(),
            archetype_component_access: Default::default(),
        }
    }
}

fn null<T>(_ctx: RenderSystemCtx<T>) {}
pub struct MarkerB;
impl<Out: GPUFutureBlock, Marker, T: IntoSystem<(), Out, Marker>>
    IntoRenderSystem<Out, (Marker, MarkerB)> for T
where
    T: Send + Sync + 'static,
    Out: Send + Sync + 'static,
{
    fn with_queue(self, queue: ComponentId) -> impl System<In = (), Out = ()> {
        let self_system = IntoSystem::into_system(self);
        let null_system = IntoSystem::into_system(null);
        let self_system_name = self_system.name();
        RenderSystem::<_, _> {
            inner: bevy::ecs::system::PipeSystem::new(null_system, self_system, self_system_name),
            future: None,
            frames: RingBuffer::new(),
            shared_state_component_id: ComponentId::new(usize::MAX),
            queue_component_id: queue,
            queue_selector: None,
            component_access: Default::default(),
            archetype_component_access: Default::default(),
        }
    }
    fn into_render_system<Q: QueueSelector>(self) -> impl System<In = (), Out = ()> {
        let self_system = IntoSystem::into_system(self);
        let null_system = IntoSystem::into_system(null);
        let self_system_name = self_system.name();
        RenderSystem::<_, _> {
            inner: bevy::ecs::system::PipeSystem::new(null_system, self_system, self_system_name),
            future: None,
            frames: RingBuffer::new(),
            shared_state_component_id: ComponentId::new(usize::MAX),
            queue_component_id: ComponentId::new(usize::MAX),
            queue_selector: Some(Q::component_id),
            component_access: Default::default(),
            archetype_component_access: Default::default(),
        }
    }
}

pub struct MarkerC;
impl<Marker, T: IntoSystem<QueueSystemCtx, (), Marker>> IntoRenderSystem<(), (Marker, MarkerC)>
    for T
where
    T: Send + Sync + 'static,
{
    fn into_render_system<Q: QueueSelector>(self) -> impl System<In = (), Out = ()> {
        QueueSystem::<T::System> {
            inner: IntoSystem::into_system(self),
            queue_component_id: ComponentId::new(usize::MAX),
            queue_selector: Some(Q::component_id),
            component_access: Default::default(),
            archetype_component_access: Default::default(),
            timeline_dependency: None,
        }
    }
    fn with_queue(self, queue: ComponentId) -> impl System<In = (), Out = ()> {
        QueueSystem::<T::System> {
            inner: IntoSystem::into_system(self),
            queue_component_id: queue,
            queue_selector: None,
            component_access: Default::default(),
            archetype_component_access: Default::default(),
            timeline_dependency: None,
        }
    }
}
