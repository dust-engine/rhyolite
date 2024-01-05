use std::sync::Arc;

use bevy_app::{App, Plugin};
use bevy_ecs::system::Resource;

use crate::{plugin::RhyoliteApp, Instance};

pub struct SurfacePlugin {}

impl Default for SurfacePlugin {
    fn default() -> Self {
        Self {}
    }
}

impl Plugin for SurfacePlugin {
    fn build(&self, app: &mut App) {
        app.add_instance_extension(ash::extensions::khr::Surface::name());
        #[cfg(target_os = "windows")]
        app.add_instance_extension(ash::extensions::khr::Win32Surface::name());

        #[cfg(target_os = "android")]
        app.add_instance_extension(ash::extensions::khr::AndroidSurface::name());

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        app.add_instance_extension(ash::extensions::ext::MetalSurface::name());

        #[cfg(target_os = "linux")]
        {
            todo!();
            app.add_instance_extension(ash::extensions::khr::WaylandSurface::name());
            app.add_instance_extension(ash::extensions::khr::XcbSurface::name());
            app.add_instance_extension(ash::extensions::khr::XlibSurface::name());
        }
    }
    fn finish(&self, app: &mut App) {
        let instance: &Instance = app.world.resource();
        let surface_loader = ash::extensions::khr::Surface::new(&instance.entry(), instance);
        app.insert_resource(SurfaceLoader {
            instance: instance.clone(),
            loader: Arc::new(surface_loader),
        });
    }
}

#[derive(Resource, Clone)]
pub struct SurfaceLoader {
    instance: Instance,
    loader: Arc<ash::extensions::khr::Surface>,
}
