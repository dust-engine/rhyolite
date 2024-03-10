use std::cell::UnsafeCell;

use std::sync::Arc;

use ash::vk;

use bevy::ecs::system::{In, IntoSystem};
use bevy::ecs::{
    schedule::{IntoSystemConfigs, SystemConfigs},
    system::BoxedSystem,
};

use crate::commands::ResourceTransitionCommands;
use crate::{queue::QueueType, Access};

use super::RenderSystemPass;

pub struct RenderSystemConfig {
    /// The render system must be assigned onto a queue supporting these feature flags.
    pub queue: QueueType,
    pub force_binary_semaphore: bool,
    pub is_queue_op: bool,
    pub barrier_producer_config: Option<RenderSystemBarrierProducerConfig>,
}
pub struct RenderSystemBarrierProducerConfig {
    pub barrier_producer: BoxedBarrierProducer,
    /// This is a pointer from Arc::<BarrierProducerCell<O>>::into_raw()
    pub barrier_producer_output_cell: *const (),
    pub barrier_producer_output_type: std::any::TypeId,
}
unsafe impl Send for RenderSystemBarrierProducerConfig {}
unsafe impl Sync for RenderSystemBarrierProducerConfig {}
impl Default for RenderSystemConfig {
    fn default() -> Self {
        Self {
            queue: QueueType::Graphics,
            force_binary_semaphore: false,
            is_queue_op: false,
            barrier_producer_config: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueueSystemDependencyConfig {
    pub wait: Access,
    pub signal: Access,
}
impl Default for QueueSystemDependencyConfig {
    fn default() -> Self {
        Self {
            wait: Access {
                stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                access: vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ,
            },
            signal: Access {
                stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                access: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            },
        }
    }
}

/// GAT limitation. All references contained herein are valid for the duration of the system call.
pub struct Barriers {
    pub(crate) dependency_flags: *mut vk::DependencyFlags,
    pub(crate) image_barriers: *mut Vec<vk::ImageMemoryBarrier2>,
    pub(crate) buffer_barriers: *mut Vec<vk::BufferMemoryBarrier2>,
    pub(crate) global_barriers: *mut vk::MemoryBarrier2,
    pub(crate) dropped: *mut bool,
}
impl Drop for Barriers {
    fn drop(&mut self) {
        unsafe {
            *self.dropped = true;
        }
    }
}
impl ResourceTransitionCommands for Barriers {
    fn add_global_barrier(&mut self, barrier: vk::MemoryBarrier2) -> &mut Self {
        let current = unsafe { &mut *self.global_barriers };
        current.src_stage_mask |= barrier.src_stage_mask;
        current.dst_stage_mask |= barrier.dst_stage_mask;
        current.src_access_mask |= barrier.src_access_mask;
        current.dst_access_mask |= barrier.dst_access_mask;
        self
    }

    fn add_image_barrier(&mut self, barrier: vk::ImageMemoryBarrier2) -> &mut Self {
        let current = unsafe { &mut *self.image_barriers };
        current.push(barrier);
        self
    }

    fn add_buffer_barrier(&mut self, barrier: vk::BufferMemoryBarrier2) -> &mut Self {
        let current = unsafe { &mut *self.buffer_barriers };
        current.push(barrier);
        self
    }

    fn set_dependency_flags(&mut self, flags: vk::DependencyFlags) -> &mut Self {
        unsafe {
            *self.dependency_flags |= flags;
        }
        self
    }
}

pub type BoxedBarrierProducer = BoxedSystem<Barriers, ()>;
pub struct BarrierProducerCell<O>(pub(crate) UnsafeCell<Option<O>>);
unsafe impl<O> Send for BarrierProducerCell<O> {}
unsafe impl<O> Sync for BarrierProducerCell<O> {}

pub trait IntoRenderSystemConfigs<Marker>: IntoSystemConfigs<Marker>
where
    Self: Sized,
{
    /// Ensure that this render system will be assigned to a queue supporting the specified queue flags
    /// while minimizing the amount of overhead associated with semaphore syncronizations.
    /// Should be called for most smaller render systems in-between heavier operations.
    fn on_queue(self, queue_type: QueueType) -> SystemConfigs {
        self.with_option::<RenderSystemPass>(|entry| {
            let config = entry.or_default();
            config.queue = queue_type;
        })
    }
    fn with_barriers<M, O: 'static, T: IntoSystem<Barriers, O, M>>(
        self,
        barriers: T,
    ) -> SystemConfigs {
        let cell: Option<Arc<BarrierProducerCell<O>>> = if std::mem::size_of::<O>() == 0 {
            None
        } else {
            Some(Arc::new(BarrierProducerCell(UnsafeCell::new(None))))
        };
        let cell2 = cell.clone();
        let system = barriers.pipe(move |In(input): In<O>| unsafe {
            // Cell is only ever written to in the barrier producer and read from in the render system.
            let Some(cell) = &cell else {
                return;
            };
            let ptr = cell.0.get().as_mut().unwrap();
            ptr.replace(input);
        });
        let mut barriers: Option<RenderSystemBarrierProducerConfig> =
            Some(RenderSystemBarrierProducerConfig {
                barrier_producer: Box::new(system),
                barrier_producer_output_cell: cell2
                    .map(Arc::into_raw)
                    .map(|ptr| ptr as *const ())
                    .unwrap_or(std::ptr::null()),
                barrier_producer_output_type: std::any::TypeId::of::<O>(),
            });
        self.with_option::<RenderSystemPass>(move |entry| {
            let config = entry.or_default();
            if barriers.is_none() {
                unimplemented!();
                // TODO: allow collective barriers for multiple systems
            }
            config.barrier_producer_config = barriers.take();
        })
    }
}

impl<Marker, T> IntoRenderSystemConfigs<Marker> for T where T: IntoSystemConfigs<Marker> {}
