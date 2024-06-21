use ash::vk::{self};
use bevy::math::UVec3;

use crate::{dispose::RenderObject, ecs::queue_cap::IsComputeQueueCap, BufferLike, QueryPool};

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

    fn copy_acceleration_structure(&mut self, info: &vk::CopyAccelerationStructureInfoKHR<'_>) {
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .extension::<ash::khr::acceleration_structure::Meta>()
                .cmd_copy_acceleration_structure(cmd_buf, info);
        }
    }

    fn write_acceleration_structures_properties(
        &mut self,
        accel_structs: &[vk::AccelerationStructureKHR],
        pool: &mut RenderObject<QueryPool>,
        first_query: u32,
    ) {
        let pool = pool.use_on(self.semaphore_signal());
        debug_assert!(first_query + accel_structs.len() as u32 <= pool.count());
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device()
                .extension::<ash::khr::acceleration_structure::Meta>()
                .cmd_write_acceleration_structures_properties(
                    cmd_buf,
                    accel_structs,
                    pool.query_type(),
                    pool.raw(),
                    first_query,
                );
        }
    }

    fn reset_query_pool(
        &mut self,
        pool: &mut RenderObject<QueryPool>,
        range: impl std::ops::RangeBounds<u32>,
    ) {
        let pool = pool.use_on(self.semaphore_signal());

        let first_query = match range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let last_query = match range.end_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n - 1,
            std::ops::Bound::Unbounded => pool.count() - 1,
        };
        unsafe {
            let cmd_buf = self.cmd_buf();
            self.device().cmd_reset_query_pool(
                cmd_buf,
                pool.raw(),
                first_query,
                last_query + 1 - first_query,
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
