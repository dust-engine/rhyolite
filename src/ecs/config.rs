use bevy_ecs::schedule::{IntoSystemConfigs, SystemConfigs};

use crate::queue::{QueueRef, QueueType};

use super::RenderSystemPass;

#[derive(Debug, Clone)]
pub struct RenderSystemConfig {
    /// The render system must be assigned onto a queue supporting these feature flags.
    pub queue: QueueAssignment,
}

#[derive(Debug, Clone)]
pub enum QueueAssignment {
    /// The render system will be assigned to a queue supporting the specified queue flags
    /// while minimizing the amount of overhead associated with semaphore syncronization.
    ///
    /// Should be selected for most smaller operations in-between heavier operations.
    ///
    /// For example, in a 'compute' -> 'transfer' -> 'graphics' dependency chain, all nodes
    /// will be merged into the same node 'graphics'.
    MinOverhead(QueueType),
    /// The render system will be assigned to a queue supporting the specified queue flags
    /// while maximizing asyncronous execution.
    ///
    /// Should be done rarely and for heavy weight operations.
    ///
    /// For example, in a 'transfer' -> 'graphics' -> 'compute' dependency chain, all nodes
    /// will be assigned to separate queues.
    MaxAsync(QueueType),
    /// Manually select the queue to use.
    Manual(QueueRef),
}

pub trait IntoRenderSystemConfig<Marker>: IntoSystemConfigs<Marker>
where
    Self: Sized,
{
    /// Ensure that this render system will be assigned to a queue supporting the specified queue flags
    /// while minimizing the amount of overhead associated with semaphore syncronizations.
    /// Should be called for most smaller render systems in-between heavier operations.
    fn on_queue<M>(self, queue_type: QueueType) -> SystemConfigs {
        self.with_option::<RenderSystemPass>(RenderSystemConfig {
            queue: QueueAssignment::MinOverhead(queue_type),
        })
    }
    /// Assign this render system to a queue supporting the specified queue flags
    /// while maximizing opportunities for asyncronous execution.
    ///
    /// Should be called for heavy weight operations only.
    fn on_async_queue<M>(self, queue: QueueType) -> SystemConfigs {
        self.with_option::<RenderSystemPass>(RenderSystemConfig {
            queue: QueueAssignment::MaxAsync(queue),
        })
    }
    /// Assign this render system to a specific queue. The caller is responsible for ensuring
    /// that the queue supports the features required.
    fn on_specific_queue<M>(self, queue: QueueRef) -> SystemConfigs {
        self.with_option::<RenderSystemPass>(RenderSystemConfig {
            queue: QueueAssignment::Manual(queue),
        })
    }
}
impl<Marker, T: IntoSystemConfigs<Marker>> IntoRenderSystemConfig<Marker> for T {}
