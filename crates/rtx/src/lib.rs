#![feature(let_chains)]
#![feature(alloc_layout_extra)]
#![feature(impl_trait_in_assoc_type)]
#![feature(type_alias_impl_trait)]
#![feature(associated_type_defaults)]
#![feature(generic_const_exprs)]

mod accel_struct;
mod blas;
mod pipeline;
mod sbt;
mod tlas;

pub use accel_struct::*;
pub use blas::*;
pub use pipeline::*;
use rhyolite::ash::vk;
pub use sbt::*;
pub use tlas::*;

use bevy::app::{App, Plugin};
pub struct RtxPlugin;
impl Plugin for RtxPlugin {
    fn build(&self, app: &mut App) {
        use bevy::utils::tracing;
        use rhyolite::{ash::khr, RhyoliteApp};
        app.add_device_extension::<khr::acceleration_structure::Meta>()
            .unwrap();
        app.add_device_extension::<khr::ray_tracing_pipeline::Meta>()
            .unwrap();
        app.add_device_extension::<khr::pipeline_library::Meta>()
            .ok();
        app.enable_feature::<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>(|f| {
            &mut f.acceleration_structure
        })
        .unwrap();
        app.enable_feature::<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>(|f| {
            &mut f.ray_tracing_pipeline
        })
        .unwrap();

        app.enable_feature::<vk::PhysicalDeviceBufferDeviceAddressFeatures>(|f| {
            &mut f.buffer_device_address
        })
        .unwrap();
        if app
            .enable_feature::<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>(|f| {
                &mut f.acceleration_structure_host_commands
            })
            .exists()
        {
            tracing::info!("Acceleration structure host commands enabled");
        }
    }
}
