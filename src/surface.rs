use std::{ops::Deref, sync::Arc};

use ash::{prelude::VkResult, vk};
use bevy_app::{App, Plugin, Update};
use bevy_ecs::prelude::*;
use bevy_window::{RawHandleWrapper, Window};
use raw_window_handle::{HasRawDisplayHandle, RawDisplayHandle};

use crate::{plugin::RhyoliteApp, Device, Instance, HasDevice};

pub struct SurfacePlugin {}

impl Default for SurfacePlugin {
    fn default() -> Self {
        Self {}
    }
}

impl Plugin for SurfacePlugin {
    fn build(&self, app: &mut App) {
        app.add_instance_extension(ash::extensions::khr::Surface::name())
            .unwrap();

        if let Some(event_loop) = app
            .world
            .get_non_send_resource::<winit::event_loop::EventLoop<()>>()
        {
            match event_loop.raw_display_handle() {
                #[cfg(target_os = "windows")]
                RawDisplayHandle::Windows(_) => {
                    app.add_instance_extension(ash::extensions::khr::Win32Surface::name())
                        .unwrap();
                }
                #[cfg(target_os = "linux")]
                RawDisplayHandle::Xlib(_) => {
                    app.add_instance_extension(ash::extensions::khr::XlibSurface::name())
                        .unwrap();
                }
                #[cfg(target_os = "linux")]
                RawDisplayHandle::Xcb(_) => {
                    app.add_instance_extension(ash::extensions::khr::XcbSurface::name())
                        .unwrap();
                }
                #[cfg(target_os = "linux")]
                RawDisplayHandle::Wayland(_) => {
                    app.add_instance_extension(ash::extensions::khr::WaylandSurface::name())
                        .unwrap();
                }
                #[cfg(target_os = "android")]
                RawDisplayHandle::Android(_) => {
                    app.add_instance_extension(ash::extensions::khr::AndroidSurface::name())
                        .unwrap();
                }
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                RawDisplayHandle::UiKit(_) | RawDisplayHandle::AppKit(_) => {
                    app.add_instance_extension(ash::extensions::ext::MetalSurface::name())
                        .unwrap();
                }
                _ => tracing::warn!("Your display is not supported."),
            };
        } else {
            panic!("rhyolite::SurfacePlugin must be inserted after bevy_winit::WinitPlugin.")
        };

        app.add_systems(
            Update,
            extract_surfaces.run_if(|e: EventReader<bevy_window::WindowCreated>| !e.is_empty()),
        );
    }
    fn finish(&self, app: &mut App) {
        let instance: &Instance = app.world.resource();
        let surface_loader = ash::extensions::khr::Surface::new(&instance.entry(), instance);
        app.insert_resource(SurfaceLoader(Arc::new(SurfaceLoaderInner {
            instance: instance.clone(),
            loader: surface_loader,
        })));
    }
}

#[derive(Resource, Clone)]
pub struct SurfaceLoader(Arc<SurfaceLoaderInner>);
struct SurfaceLoaderInner {
    instance: Instance,
    loader: ash::extensions::khr::Surface,
}
impl SurfaceLoader {
    pub fn instance(&self) -> &Instance {
        &self.0.instance
    }
}
impl Deref for SurfaceLoader {
    type Target = ash::extensions::khr::Surface;
    fn deref(&self) -> &Self::Target {
        &self.0.loader
    }
}

#[derive(Component)]
pub struct Surface {
    loader: SurfaceLoader,
    inner: vk::SurfaceKHR,
}
impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_surface(self.inner, None);
        }
    }
}
impl Surface {
    pub fn create(
        loader: SurfaceLoader,
        window_handle: &impl raw_window_handle::HasRawWindowHandle,
        display_handle: &impl raw_window_handle::HasRawDisplayHandle,
    ) -> VkResult<Surface> {
        let surface = unsafe {
            ash_window::create_surface(
                loader.instance().entry(),
                loader.instance(),
                display_handle.raw_display_handle(),
                window_handle.raw_window_handle(),
                None,
            )?
        };
        Ok(Surface {
            loader,
            inner: surface,
        })
    }
    pub fn raw(&self) -> vk::SurfaceKHR {
        self.inner
    }
}
pub(super) fn extract_surfaces(
    mut commands: Commands,
    loader: Res<SurfaceLoader>,
    mut window_created_events: EventReader<bevy_window::WindowCreated>,
    query: Query<(&RawHandleWrapper, Option<&Surface>), With<Window>>,
) {
    for create_event in window_created_events.read() {
        let (raw_handle, surface) = query.get(create_event.window).unwrap();
        let raw_handle = unsafe { raw_handle.get_handle() };
        assert!(surface.is_none());
        let new_surface = Surface::create(loader.clone(), &raw_handle, &raw_handle).unwrap();
        commands.entity(create_event.window).insert(new_surface);
    }
}
