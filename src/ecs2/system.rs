use std::{
    any::Any,
    borrow::Cow,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    pin::Pin,
};

use ash::vk;
use bevy::{
    ecs::{
        archetype::ArchetypeComponentId,
        component::{ComponentId, Tick},
        query::Access,
        system::{System, SystemMeta, SystemParam},
        world::{unsafe_world_cell::UnsafeWorldCell, DeferredWorld},
    },
    prelude::{In, IntoSystem, Mut, World},
    utils::ConfigMap,
};

use crate::{
    command::{states, CommandBuffer, CommandPool, Timeline},
    future::{GPUFutureBlock, GPUFutureBlockReturnValue, GPUFutureContext},
    utils::RingBuffer,
    Device, QueueConfiguration, QueueSelector,
};

/// Used as In<RenderSystemCtx<Returned>> for render systems.
pub struct RenderSystemCtx<Returned = ()> {
    returned_value: Option<Returned>,
}
struct RenderSystemFrame<Returned, Retained> {
    returned: Returned,
    retained: Retained,
}
/// A wrapper for a system that records command buffers.
pub struct RenderSystem<
    Q: QueueSelector,
    F: GPUFutureBlock + Send + Sync,
    T: bevy::ecs::system::System<In = RenderSystemCtx<F::Returned>, Out = F>,
> {
    inner: T,
    future: Option<F>,
    frames: RingBuffer<RenderSystemFrame<F::Returned, F::Retained>, 3>,
    shared_state_component_id: ComponentId,
    queue_component_id: ComponentId,

    component_access: Access<ComponentId>,
    archetype_component_access: Access<ArchetypeComponentId>,
    _marker: PhantomData<Q>,
}

/// Shared across all render systems within a single submission.
pub(super) struct RenderSystemSharedState {
    /// The submission will take the command buffer and push it here.
    /// The prelude system is responsible for popping command buffers from this queue and await on them.
    pending_command_buffers: RingBuffer<CommandBuffer<states::Pending>, 3>,

    /// The prelude system will allocate and populate this command buffer.
    recording_command_buffer: Option<CommandBuffer<states::Recording>>,
    ctx: GPUFutureContext,
    timeline: Timeline,
    command_pool: CommandPool,
}
impl RenderSystemSharedState {
    pub(super) fn new(device: Device, queue_family_index: u32) -> Self {
        Self {
            ctx: GPUFutureContext::new(device.clone(), vk::CommandBuffer::null()),
            timeline: Timeline::new(device.clone()).unwrap(),
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
    pub(super) shared_state: ComponentId,
}

// command buffer ownership.
// obviously in the simple case, ownership is easy. eveerything owned by the RenderSystemSharedState.
// we also want command buffer affinity. so command buffers should return at a predictable fasion.

impl<
        Q: QueueSelector,
        F: GPUFutureBlock + Send + Sync + 'static,
        T: bevy::ecs::system::System<In = RenderSystemCtx<F::Returned>, Out = F>,
    > bevy::ecs::system::System for RenderSystem<Q, F, T>
where
    F::Retained: Send + Sync,
    F::Returned: Send + Sync,
{
    type In = ();

    type Out = ();

    fn default_configs(&mut self, config: &mut ConfigMap) {
        // Inseret this config to identify the current system as a render config.
        config.insert(RenderSystemIdentifierConfig {
            queue_component_id: self.queue_component_id,
            is_standalone: false,
        });
    }

    fn configurate(&mut self, config: &mut dyn Any, world: &mut World) {
        let Some(config) = config.downcast_mut::<RenderSystemInputConfig>() else {
            return;
        };
        self.shared_state_component_id = config.shared_state;
        {
            // Add component access for the shared state
            self.component_access
                .add_write(self.shared_state_component_id);
            let archetype_component_id = world
                .get_resource_archetype_component_id(self.shared_state_component_id)
                .unwrap();
            self.archetype_component_access
                .add_write(archetype_component_id);
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
                // TODO: signal events, and wait for events.
                // how about this: between system sets we use
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
        let queue_config: &QueueConfiguration = world.resource();
        self.queue_component_id = Q::component_id(queue_config);
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
        ComponentId::new(0)
    }
    fn configurate(
        state: &mut Self::State,
        config: &mut dyn Any,
        meta: &mut SystemMeta,
        world: &mut World,
    ) {
        let Some(config) = config.downcast_mut::<RenderSystemInputConfig>() else {
            return;
        };
        *state = config.shared_state;
        {
            // Add component access for the shared state
            meta.component_access_set
                .add_unfiltered_write(config.shared_state);
            let archetype_component_id = world
                .get_resource_archetype_component_id(config.shared_state)
                .unwrap();
            meta.archetype_component_access
                .add_write(archetype_component_id);
        }
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
    //mut queue: Queue,
    mut shared: RenderSystemSharedStateSystemParam,
    //queue_submission_ctx: (), // this gives you the semaphores from the schedule build pass and identify the system as a queue system.
) {
    let command_buffer = shared.recording_command_buffer.take().unwrap();
    let command_buffer = shared.command_pool.end(command_buffer);
    /*
    queue.submit_one(command_buffer, &[
        QueueDependency {

        }
    ]).unwrap();
    */
}

/// An extension trait for [`IntoSystem`]
pub trait IntoRenderSystem<Out, Marker> {
    fn into_render_system<Q: QueueSelector>(self) -> impl System<In = (), Out = ()>;
}

pub struct MarkerA;
impl<Out: GPUFutureBlock, Marker, T: IntoSystem<RenderSystemCtx<Out::Returned>, Out, Marker>>
    IntoRenderSystem<Out, (Marker, MarkerA)> for T
where
    T: Send + Sync + 'static,
    Out: Send + Sync + 'static,
{
    fn into_render_system<Q: QueueSelector>(self) -> impl System<In = (), Out = ()> {
        RenderSystem::<Q, Out, T::System> {
            inner: IntoSystem::into_system(self),
            future: None,
            frames: RingBuffer::new(),
            shared_state_component_id: ComponentId::new(0),
            queue_component_id: ComponentId::new(0),
            component_access: Default::default(),
            archetype_component_access: Default::default(),
            _marker: PhantomData,
        }
    }
}
pub struct MarkerB;
impl<Out: GPUFutureBlock, Marker, T: IntoSystem<(), Out, Marker>>
    IntoRenderSystem<Out, (Marker, MarkerB)> for T
where
    T: Send + Sync + 'static,
    Out: Send + Sync + 'static,
{
    fn into_render_system<Q: QueueSelector>(self) -> impl System<In = (), Out = ()> {
        fn null<T>(In(_ctx): In<RenderSystemCtx<T>>) {}
        let self_system = IntoSystem::into_system(self);
        let null_system = IntoSystem::into_system(null);
        let self_system_name = self_system.name();
        RenderSystem::<Q, _, _> {
            inner: bevy::ecs::system::PipeSystem::new(null_system, self_system, self_system_name),
            future: None,
            frames: RingBuffer::new(),
            shared_state_component_id: ComponentId::new(0),
            queue_component_id: ComponentId::new(0),
            component_access: Default::default(),
            archetype_component_access: Default::default(),
            _marker: PhantomData,
        }
    }
}
