use ash::vk;
use bevy_app::{App, Update};
use bevy_ecs::{schedule::{Condition, IntoSystemConfigs, IntoSystemSet, SystemConfig, SystemConfigs, SystemSet}, system::{IntoSystem, ReadOnlySystem}};

use crate::queue::QueueType;

use super::RenderSystemPass;

pub struct RenderSystemConfig {
    /// The render system must be assigned onto a queue supporting these feature flags.
    pub queue: QueueType,
    pub force_binary_semaphore: bool,
    pub is_queue_op: bool,
    pub barrier_producer: Option<BoxedBarrierProducer>
}
impl Default for RenderSystemConfig {
    fn default() -> Self {
        Self {
            queue: QueueType::Graphics,
            force_binary_semaphore: false,
            is_queue_op: false,
            barrier_producer: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Access {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
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
    pub(crate) image_barriers: *mut Vec<vk::ImageMemoryBarrier2>,
    pub(crate) buffer_barriers: *mut Vec<vk::BufferMemoryBarrier2>,
    pub(crate) memory_barriers: *mut Vec<vk::MemoryBarrier2>
}

pub type BoxedBarrierProducer = Box<dyn ReadOnlySystem<In = Barriers, Out = ()>>;

pub trait RenderSystem {
    fn system(&self) -> SystemConfigs;
    fn barriers(&self) -> BoxedBarrierProducer;
}

pub struct RenderSystemConfigs(SystemConfigs);

pub trait IntoRenderSystemConfigs
where
    Self: Sized,
{
    /// Convert into a [`SystemConfigs`].
    fn into_configs(self) -> SystemConfigs;

    /// Add these systems to the provided `set`.
    #[track_caller]
    fn in_set(self, set: impl SystemSet) -> RenderSystemConfigs {
        RenderSystemConfigs(self.into_configs().in_set(set))
    }
    fn before<M>(self, set: impl IntoSystemSet<M>) -> RenderSystemConfigs {
        RenderSystemConfigs(self.into_configs().before(set))
    }
    fn after<M>(self, set: impl IntoSystemSet<M>) -> RenderSystemConfigs {
        RenderSystemConfigs(self.into_configs().after(set))
    }
    fn distributive_run_if<M>(self, condition: impl Condition<M> + Clone) -> RenderSystemConfigs {
        RenderSystemConfigs(self.into_configs().distributive_run_if(condition))
    }
    fn run_if<M>(self, condition: impl Condition<M>) -> RenderSystemConfigs {
        RenderSystemConfigs(self.into_configs().run_if(condition))
    }
    fn ambiguous_with<M>(self, set: impl IntoSystemSet<M>) -> RenderSystemConfigs {
        RenderSystemConfigs(self.into_configs().ambiguous_with(set))
    }
    fn ambiguous_with_all(self) -> RenderSystemConfigs {
        RenderSystemConfigs(self.into_configs().ambiguous_with_all())
    }
    fn on_queue<M>(self, queue_type: QueueType) -> RenderSystemConfigs {
        let this = self.into_configs().with_option::<RenderSystemPass>(|entry| {
            let config = entry.or_default();
            config.queue = queue_type;
        });
        RenderSystemConfigs(this)
    }
}

impl<T: RenderSystem> IntoRenderSystemConfigs for T {
    fn into_configs(self) -> SystemConfigs {
        self.system().with_option::<RenderSystemPass>(|entry| {
            let config = entry.or_default();
            config.barrier_producer = Some(self.barriers());
        })
    }
}
impl IntoRenderSystemConfigs for RenderSystemConfigs {
    fn into_configs(self) -> SystemConfigs {
        self.0
    }
}

pub trait RenderApp {
    fn add_render_system(&mut self, render_system: impl IntoRenderSystemConfigs) -> &mut App;
}
impl RenderApp for App {
    fn add_render_system(&mut self, render_system: impl IntoRenderSystemConfigs) -> &mut App {
        self.add_systems(Update, render_system.into_configs());
        self
    }

}