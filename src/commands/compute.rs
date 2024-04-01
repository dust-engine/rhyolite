use ash::vk;
use bevy::math::UVec3;

use crate::{ecs::queue_cap::IsComputeQueueCap, BufferLike};

use super::CommandRecorder;

pub trait ComputeCommands: Sized + CommandRecorder {
    /// Dispatch compute work items
    fn dispatch(&mut self, dimensions: UVec3) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .cmd_dispatch(cmd_buf, dimensions.x, dimensions.y, dimensions.z);
        }
    }
    /// Dispatch compute work items with indirect parameters
    fn dispatch_indirect(&mut self, args: &impl BufferLike) {
        assert!(args.size() as usize >= std::mem::size_of::<vk::DispatchIndirectCommand>());
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .cmd_dispatch_indirect(cmd_buf, args.raw_buffer(), args.offset());
        }
    }
    /// Dispatch compute work items with non-zero base values for the workgroup IDs
    fn dispatch_base(&mut self, base: UVec3, dimensions: UVec3) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device().cmd_dispatch_base(
                cmd_buf,
                base.x,
                base.y,
                base.z,
                dimensions.x,
                dimensions.y,
                dimensions.z,
            );
        }
    }
}

impl<T> ComputeCommands for T
where
    T: CommandRecorder,
    (): IsComputeQueueCap<{ T::QUEUE_CAP }>,
{
}
