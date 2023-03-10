use std::{alloc::Layout, ops::Range, sync::Arc};

use super::AccelerationStructure;
use crate::HasDevice;
use crate::{future::GPUCommandFuture, Allocator, BufferLike, ResidentBuffer};
use ash::vk;
use pin_project::pin_project;

pub struct AccelerationStructureBuild {
    pub accel_struct: AccelerationStructure,
    pub build_size: vk::AccelerationStructureBuildSizesInfoKHR,
    pub geometries: Box<[(Arc<ResidentBuffer>, usize, vk::GeometryFlagsKHR, u32)]>, // data, stride, flags, num_primitives
    pub primitive_datasize: usize,
}

/// Builds many acceleration structures in batch.
pub struct AccelerationStructureBatchBuilder {
    allocator: Allocator,
    builds: Vec<AccelerationStructureBuild>,
}

impl AccelerationStructureBatchBuilder {
    pub fn new(allocator: Allocator) -> Self {
        Self {
            allocator,
            builds: Vec::new(),
        }
    }
    pub fn add_build(&mut self, item: AccelerationStructureBuild) {
        self.builds.push(item)
    }
    // build on the device.
    /// Instead of calling VkCmdBuildAccelerationStructure multiple times, it calls VkCmdBuildAccelerationStructure
    /// in batch mode, once for BLAS and once for TLAS, with a pipeline barrier inbetween.  
    pub fn build(self) -> BLASBuildFuture {
        // Calculate the total number of geometries
        let total_num_geometries = self.builds.iter().map(|build| build.geometries.len()).sum();
        let mut geometries: Vec<vk::AccelerationStructureGeometryKHR> =
            Vec::with_capacity(total_num_geometries);
        let mut build_ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR> =
            Vec::with_capacity(total_num_geometries);
        let mut build_range_ptrs: Box<[*const vk::AccelerationStructureBuildRangeInfoKHR]> =
            Vec::with_capacity(self.builds.len()).into_boxed_slice();

        let scratch_buffer_alignment =
            self.allocator
                .device()
                .physical_device()
                .properties()
                .acceleration_structure
                .min_acceleration_structure_scratch_offset_alignment as usize;

        let mut scratch_buffer_total: u64 = 0;
        for build in self.builds.iter() {
            scratch_buffer_total += Layout::from_size_align(
                build.build_size.build_scratch_size as usize,
                scratch_buffer_alignment,
            )
            .unwrap()
            .pad_to_align()
            .size() as u64;
        }
        let scratch_buffer = self
            .allocator
            .create_device_buffer_uninit(
                scratch_buffer_total,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .unwrap();
        let scratch_buffer_device_address = scratch_buffer.device_address();

        // Create build infos
        let mut current_scratch_buffer_device_address = scratch_buffer_device_address;

        let build_infos = self
            .builds
            .iter()
            .enumerate()
            .map(|(i, as_build)| {
                build_range_ptrs[i] = unsafe { build_ranges.as_ptr().add(build_ranges.len()) };

                // Add geometries
                build_ranges.extend(as_build.geometries.iter().map(|(_, _, _, a)| {
                    vk::AccelerationStructureBuildRangeInfoKHR {
                        primitive_count: *a,
                        primitive_offset: 0,
                        first_vertex: 0,
                        transform_offset: 0,
                    }
                }));

                let geometry_range: Range<usize> =
                    geometries.len()..(geometries.len() + as_build.geometries.len());
                // Insert geometries
                geometries.extend(aabbs_to_geometry_infos(&as_build.geometries));
                vk::AccelerationStructureBuildGeometryInfoKHR {
                    ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                    flags: as_build.accel_struct.flags,
                    mode: vk::BuildAccelerationStructureModeKHR::BUILD,
                    dst_acceleration_structure: as_build.accel_struct.raw,
                    geometry_count: as_build.geometries.len() as u32,
                    p_geometries: unsafe { geometries.as_ptr().add(geometry_range.start) },
                    scratch_data: {
                        let d = vk::DeviceOrHostAddressKHR {
                            device_address: current_scratch_buffer_device_address,
                        };
                        current_scratch_buffer_device_address += Layout::from_size_align(
                            as_build.build_size.build_scratch_size as usize,
                            scratch_buffer_alignment,
                        )
                        .unwrap()
                        .pad_to_align()
                        .size()
                            as u64;
                        d
                    },
                    ..Default::default()
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        assert_eq!(geometries.len(), total_num_geometries);
        assert_eq!(build_ranges.len(), total_num_geometries);
        BLASBuildFuture {
            geometries: geometries.into_boxed_slice(),
            build_infos,
            build_range_infos: build_ranges.into_boxed_slice(),
            build_range_ptrs,
        }
    }
}

#[pin_project]
pub struct BLASBuildFuture {
    geometries: Box<[vk::AccelerationStructureGeometryKHR]>,
    build_infos: Box<[vk::AccelerationStructureBuildGeometryInfoKHR]>,
    build_range_infos: Box<[vk::AccelerationStructureBuildRangeInfoKHR]>,
    build_range_ptrs: Box<[*const vk::AccelerationStructureBuildRangeInfoKHR]>,
}

impl GPUCommandFuture for BLASBuildFuture {
    type Output = ();

    type RetainedState = ();

    type RecycledState = ();

    fn record(
        self: std::pin::Pin<&mut Self>,
        ctx: &mut crate::future::CommandBufferRecordContext,
        recycled_state: &mut Self::RecycledState,
    ) -> std::task::Poll<(Self::Output, Self::RetainedState)> {
        let this = self.project();
        assert_eq!(this.build_range_ptrs.len(), this.build_infos.len());
        ctx.record(|ctx, cmd_buffer| unsafe {
            (ctx.device()
                .accel_struct_loader()
                .fp()
                .cmd_build_acceleration_structures_khr)(
                cmd_buffer,
                this.build_infos.len() as u32,
                this.build_infos.as_ptr(),
                this.build_range_ptrs.as_ptr(),
            )
        });
        std::task::Poll::Ready(((), ()))
    }

    fn context(self: std::pin::Pin<&mut Self>, ctx: &mut crate::future::StageContext) {}
}

fn aabbs_to_geometry_infos(
    geometries: &[(Arc<ResidentBuffer>, usize, vk::GeometryFlagsKHR, u32)],
) -> impl IntoIterator<Item = vk::AccelerationStructureGeometryKHR> + '_ {
    geometries.iter().map(
        |(data, stride, flags, _)| vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::AABBS,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                aabbs: vk::AccelerationStructureGeometryAabbsDataKHR {
                    data: vk::DeviceOrHostAddressConstKHR {
                        device_address: data.device_address(),
                    },
                    stride: *stride as u64,
                    ..Default::default()
                },
            },
            flags: *flags,
            ..Default::default()
        },
    )
}
