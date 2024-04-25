use ash::vk::{self};
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
    fn build_acceleration_structure<'a>(
        &mut self,
        infos: &[vk::AccelerationStructureBuildGeometryInfoKHR],
        build_range_infos: impl Iterator<Item = &'a [vk::AccelerationStructureBuildRangeInfoKHR]>,
    ) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            let ptrs = build_range_infos
                .zip(infos.iter())
                .map(|(ranges, info)| {
                    assert_eq!(ranges.len() as u32, info.geometry_count);
                    ranges.as_ptr()
                })
                .collect::<Vec<_>>();
            assert_eq!(infos.len(), ptrs.len());
            (self
                .device()
                .extension::<ash::khr::acceleration_structure::Meta>()
                .fp()
                .cmd_build_acceleration_structures_khr)(
                cmd_buf,
                infos.len() as u32,
                infos.as_ptr(),
                ptrs.as_ptr(),
            );
        }
    }

    fn build_acceleration_structure_indirect(
        &mut self,
        infos: &[vk::AccelerationStructureBuildGeometryInfoKHR],
        indirect_device_addresses: &[vk::DeviceAddress],
        indirect_strides: &[u32],
        max_primitive_counts: &[&[u32]],
    ) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .extension::<ash::khr::acceleration_structure::Meta>()
                .cmd_build_acceleration_structures_indirect(
                    cmd_buf,
                    infos,
                    indirect_device_addresses,
                    indirect_strides,
                    max_primitive_counts,
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
