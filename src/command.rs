use std::{marker::PhantomData, ops::{Deref, DerefMut}, sync::{atomic::AtomicU64, Arc}};

use ash::{prelude::VkResult, vk::{self, Handle, SwapchainKHR}};
use bevy::{ecs::{component::ComponentId, system::SystemParam}, prelude::{FromWorld, World}};

use crate::{future::GPUFutureContext, semaphore::{SemaphoreDeferredValue, TimelineSemaphore}, swapchain::SwapchainImage, Device, HasDevice, Queue, QueueConfiguration, QueueSelector};

pub struct CommandPool {
    device: Device,
    raw: vk::CommandPool,
    queue_family_index: u32,
    flags: vk::CommandPoolCreateFlags,
    num_outstanding_buffers: u32,
}
impl Drop for CommandPool {
    fn drop(&mut self) {
        assert!(self.num_outstanding_buffers == 0, "Cannot drop the command pool because there are still outstanding command buffers allocated from this pool.");
        unsafe  {
            self.device.destroy_command_pool(self.raw, None);
        }
    }
}

pub struct CommandEncoder<'a> {
    pool: &'a mut CommandPool,
    pub(crate) command_buffer: CommandBuffer,
}
pub struct CommandBuffer {
    raw: vk::CommandBuffer,
    flags: vk::CommandBufferUsageFlags,
    pub(crate) timeline_semaphore: Arc<TimelineSemaphore>,
    pub(crate) wait_value: u64,
    pub(crate) future_ctx: GPUFutureContext,
    drop_guard: CommandBufferDropGuard,
}
struct CommandBufferDropGuard;
impl Drop for CommandBufferDropGuard {
    fn drop(&mut self) {
        panic!("CommandBuffer must be returned to the CommandPool!");
    }
}


impl CommandPool {
    pub fn new(device: Device, queue_family_index: u32, flags: vk::CommandPoolCreateFlags) -> VkResult<Self> {
        unsafe {
            let raw = device.create_command_pool(&vk::CommandPoolCreateInfo {
                queue_family_index,
                ..Default::default()
            }, None)?;
            Ok(Self {
                raw,
                device,
                num_outstanding_buffers: 0,
                queue_family_index,
                flags,
            })
        }
    }
    pub fn start_encoding(&mut self, on_timeline: &Timeline, flags: vk::CommandBufferUsageFlags) -> VkResult<CommandEncoder> {
        self.num_outstanding_buffers += 1;
        let mut raw = vk::CommandBuffer::null();
        unsafe  {
            (self.device.fp_v1_0().allocate_command_buffers)(self.device.handle(), &vk::CommandBufferAllocateInfo {
                command_pool: self.raw,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            }, &mut raw).result()?;
            self.device.begin_command_buffer(raw, &vk::CommandBufferBeginInfo {
                flags,
                ..Default::default()
            })?;
        }
        let command_buffer = CommandBuffer {
            raw,
            flags,
            timeline_semaphore: on_timeline.semaphore.clone(),
            wait_value: on_timeline.wait_value.load(std::sync::atomic::Ordering::Relaxed),
            future_ctx: GPUFutureContext::new(self.device.clone(), raw),
            drop_guard: CommandBufferDropGuard,
        };

        on_timeline.increment();
        Ok(CommandEncoder {
            pool: self,
            command_buffer,
        })
    }
    pub fn restart_encoding(&mut self, mut command_buffer: CommandBuffer, on_timeline: &mut Timeline, release_resources: bool) -> VkResult<CommandEncoder> {
        assert!(self.flags.contains(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER));
        let flags = if release_resources {
            vk::CommandBufferResetFlags::RELEASE_RESOURCES
        } else {
            vk::CommandBufferResetFlags::empty()
        };
        unsafe {
            self.device.reset_command_buffer(command_buffer.raw, flags)?;
        }
        command_buffer.future_ctx.clear();
        
        let encoder = CommandEncoder {
            pool: self,
            command_buffer,
        };
        Ok(encoder)
    }
    pub fn recycle(&mut self, buffer: CommandBuffer) {
        unsafe {
            self.device.free_command_buffers(self.raw, &[buffer.raw]);
            self.num_outstanding_buffers -= 1;

            std::mem::forget(buffer.drop_guard);
        }
    }
    pub fn recycle_many(&mut self, buffers: impl Iterator<Item = CommandBuffer>) {
        let handles: Vec<_> = buffers.map(|x| {
            let buf = x.raw;
            std::mem::forget(x.drop_guard);
            buf
        }).collect();
        unsafe {
            self.device.free_command_buffers(self.raw, &handles);
            self.num_outstanding_buffers -= handles.len() as u32;
        }
    }
}


impl<'a> CommandEncoder<'a> {
    pub fn end(self) -> VkResult<CommandBuffer> {
        unsafe {
            self.pool.device.end_command_buffer(self.command_buffer.raw)?;
        }
        Ok(self.command_buffer)
    }
    pub fn reset(&mut self, release_resources: bool) -> VkResult<()> {
        assert!(self.pool.flags.contains(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER));
        let flags = if release_resources {
            vk::CommandBufferResetFlags::RELEASE_RESOURCES
        } else {
            vk::CommandBufferResetFlags::empty()
        };
        unsafe {
            self.pool.device.reset_command_buffer(self.command_buffer.raw, flags)?;
        }
        Ok(())
    }
}


// Each queue system will have one of this.
// It gets incremented during queue submit.
pub struct Timeline {
    semaphore: Arc<TimelineSemaphore>,
    wait_value: AtomicU64,
}
impl FromWorld for Timeline {
    fn from_world(world: &mut World) -> Self {
        Self {
            semaphore: Arc::new(
                TimelineSemaphore::new(world.resource::<Device>().clone(), 0).unwrap()),
            wait_value: AtomicU64::new(0),
        }
    }
}
impl Timeline {
    pub fn increment(&self) {
        self.wait_value.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    pub fn blocking_stages(&self, stages: vk::PipelineStageFlags2) -> QueueDependency {
        QueueDependency(vk::SemaphoreSubmitInfo {
            stage_mask: stages,
            semaphore: self.semaphore.raw(),
            value: self.wait_value.load(std::sync::atomic::Ordering::Relaxed),
            ..Default::default()
        })
    }
}

#[repr(transparent)]
pub struct QueueDependency<'a>(pub(crate) vk::SemaphoreSubmitInfo<'a>);

impl<'a, T: QueueSelector> Queue<'a, T> {
    pub fn submit_one(&mut self, command_buffer: CommandBuffer, dependencies: &[QueueDependency]) -> VkResult<SemaphoreDeferredValue<CommandBuffer>> {
        // TODO: emit syncronization barrier for command buffer future ctx.
        unsafe {
            self.device.queue_submit2(self.queue, &[
                vk::SubmitInfo2 {
                    ..Default::default()
                }.command_buffer_infos(&[
                    vk::CommandBufferSubmitInfo {
                        command_buffer: command_buffer.raw,
                        ..Default::default()
                    }
                ])
                .wait_semaphore_infos(std::mem::transmute(dependencies))
                .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo {
                    semaphore: command_buffer.timeline_semaphore.raw(),
                    value: command_buffer.wait_value + 1,
                    // We signal on ALL_COMMANDS because
                    // 1. Timeline semaphore.
                    // 2. Most impl probably cannot take advantage of any other flags.
                    stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                    ..Default::default()
                }])
            ], vk::Fence::null())?;
        }
        Ok(SemaphoreDeferredValue::new(command_buffer.timeline_semaphore.clone(), command_buffer.wait_value + 1, command_buffer))
    }

    /// The submitted command buffer should transfer ownership of the swapchain image into the current queue,
    /// and also do image format transition.
    pub fn submit_one_and_present(&mut self, command_buffer: CommandBuffer, dependencies: &[QueueDependency], swapchain_image: &SwapchainImage) -> VkResult<SemaphoreDeferredValue<CommandBuffer>> {
        // TODO: emit syncronization barrier for command buffer future ctx.

        unsafe {
            self.device.queue_submit2(self.queue, &[
                vk::SubmitInfo2 {
                    ..Default::default()
                }.command_buffer_infos(&[
                    vk::CommandBufferSubmitInfo {
                        command_buffer: command_buffer.raw,
                        ..Default::default()
                    }
                ])
                .wait_semaphore_infos(std::mem::transmute(dependencies))
                .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo {
                    semaphore: command_buffer.timeline_semaphore.raw(),
                    value: command_buffer.wait_value + 1,
                    // We signal on ALL_COMMANDS because
                    // 1. Timeline semaphore.
                    // 2. Most impl probably cannot take advantage of any other flags.
                    stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                    ..Default::default()
                }, vk::SemaphoreSubmitInfo {
                    semaphore: swapchain_image.inner.as_ref().unwrap().present_semaphore,
                    value: 0,
                    stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                    ..Default::default()
                }])
            ], vk::Fence::null())?;
        }
        Ok(SemaphoreDeferredValue::new(command_buffer.timeline_semaphore.clone(), command_buffer.wait_value + 1, command_buffer))
    }
}


/// Shared command pool. One for each queue family.
/// Good for small amount of command encoding. Use parallel encoding if an incredibly large amount of
/// commands need to be encoded.
pub struct SharedCommandPool<'a, Q: QueueSelector> {
    command_pool: &'a mut CommandPool,
    _marker: PhantomData<Q>
}
impl<'a, Q: QueueSelector> Deref for SharedCommandPool<'a, Q> {
    type Target = CommandPool;
    fn deref(&self) -> &Self::Target {
        self.command_pool
    }
}
impl<'a, Q: QueueSelector> DerefMut for SharedCommandPool<'a, Q> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.command_pool
    }
}
unsafe impl<'a, Q: QueueSelector> SystemParam for SharedCommandPool<'a, Q> {
    type State = ComponentId;

    type Item<'world, 'state> = SharedCommandPool<'world, Q>;

    fn init_state(world: &mut bevy::prelude::World, system_meta: &mut bevy::ecs::system::SystemMeta) -> Self::State {
        let config = world.resource::<QueueConfiguration>();
        let component_id = Q::shared_command_pool_component_id(config);

        let combined_access = system_meta.component_access_set.combined_access();
        if combined_access.has_write(component_id) {
            panic!(
                "error[B0002]: ResMut<{}> in system {} conflicts with a previous ResMut<{0}> access. Consider removing the duplicate access. See: https://bevyengine.org/learn/errors/#b0002",
                std::any::type_name::<Self>(), system_meta.name);
        } else if combined_access.has_read(component_id) {
            panic!(
                "error[B0002]: ResMut<{}> in system {} conflicts with a previous Res<{0}> access. Consider removing the duplicate access. See: https://bevyengine.org/learn/errors/#b0002",
                std::any::type_name::<Self>(), system_meta.name);
        }
        system_meta
            .component_access_set
            .add_unfiltered_write(component_id);

        let archetype_component_id = world
            .get_resource_archetype_component_id(component_id)
            .unwrap();
        system_meta
            .archetype_component_access
            .add_write(archetype_component_id);
        component_id
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &bevy::ecs::system::SystemMeta,
        world: bevy::ecs::world::unsafe_world_cell::UnsafeWorldCell<'world>,
        change_tick: bevy::ecs::component::Tick,
    ) -> Self::Item<'world, 'state> {
        let value = world
            .get_resource_mut_by_id(*state)
            .unwrap_or_else(|| {
                panic!(
                    "Resource requested by {} does not exist: {}",
                    system_meta.name,
                    std::any::type_name::<Q>()
                )
            });
        SharedCommandPool {
            command_pool: value.value.deref_mut::<CommandPool>(),
            _marker: PhantomData
        }
    }
}
