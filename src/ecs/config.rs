use ash::vk;
use bevy_ecs::schedule::{IntoSystemConfigs, SystemConfigs};

use crate::queue::QueueType;

use super::RenderSystemPass;

#[derive(Debug, Clone)]
pub struct RenderSystemConfig {
    /// The render system must be assigned onto a queue supporting these feature flags.
    pub queue: QueueType,
    pub force_binary_semaphore: bool,
    pub is_queue_op: bool,
}
impl Default for RenderSystemConfig {
    fn default() -> Self {
        Self {
            queue: QueueType::Graphics,
            force_binary_semaphore: false,
            is_queue_op: false,
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

pub trait IntoRenderSystemConfig<Marker>: IntoSystemConfigs<Marker>
where
    Self: Sized,
{
    /// Ensure that this render system will be assigned to a queue supporting the specified queue flags
    /// while minimizing the amount of overhead associated with semaphore syncronizations.
    /// Should be called for most smaller render systems in-between heavier operations.
    fn on_queue<M>(self, queue_type: QueueType) -> SystemConfigs {
        self.with_option::<RenderSystemPass>(|entry| {
            let config = entry.or_default();
            config.queue = queue_type;
        })
    }
}
impl<Marker, T: IntoSystemConfigs<Marker>> IntoRenderSystemConfig<Marker> for T {}
