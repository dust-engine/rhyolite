use std::sync::Arc;

use ash::khr;
use ash::{khr::surface::Meta as KhrSurface, prelude::VkResult, vk};
use bevy::app::{App, Plugin, PostUpdate};
use bevy::ecs::prelude::*;
use bevy::window::{RawHandleWrapper, Window};
use raw_window_handle::{
    DisplayHandle, HasDisplayHandle, RawDisplayHandle, RawWindowHandle, WindowHandle,
};

use crate::{plugin::RhyoliteApp, Instance, PhysicalDevice};

pub struct SurfacePlugin {}

impl Default for SurfacePlugin {
    fn default() -> Self {
        Self {}
    }
}

impl Plugin for SurfacePlugin {
    fn build(&self, app: &mut App) {
        app.add_instance_extension::<KhrSurface>().unwrap();

        if let Some(event_loop) = app
            .world()
            .get_non_send_resource::<winit::event_loop::EventLoop<bevy::winit::WakeUp>>()
        {
            match event_loop.display_handle().unwrap().as_raw() {
                #[cfg(target_os = "windows")]
                RawDisplayHandle::Windows(_) => {
                    app.add_instance_extension::<ash::khr::win32_surface::Meta>()
                        .unwrap();
                }
                #[cfg(target_os = "linux")]
                RawDisplayHandle::Xlib(_) => {
                    app.add_instance_extension::<ash::khr::xlib_surface::Meta>()
                        .unwrap();
                }
                #[cfg(target_os = "linux")]
                RawDisplayHandle::Xcb(_) => {
                    app.add_instance_extension::<ash::khr::xcb_surface::Meta>()
                        .unwrap();
                }
                #[cfg(target_os = "linux")]
                RawDisplayHandle::Wayland(_) => {
                    app.add_instance_extension::<ash::khr::wayland_surface::Meta>()
                        .unwrap();
                }
                #[cfg(target_os = "android")]
                RawDisplayHandle::Android(_) => {
                    app.add_instance_extension::<ash::khr::android_surface::Meta>()
                        .unwrap();
                }
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                RawDisplayHandle::UiKit(_) | RawDisplayHandle::AppKit(_) => {
                    app.add_instance_extension::<ash::ext::metal_surface::Meta>()
                        .unwrap();
                }
                _ => tracing::warn!("Your display is not supported."),
            };
        } else {
            panic!("rhyolite::SurfacePlugin must be inserted after bevy_winit::WinitPlugin.")
        };

        app.add_systems(
            PostUpdate,
            extract_surfaces.run_if(|e: EventReader<bevy::window::WindowCreated>| !e.is_empty()),
        );
    }
}

#[derive(Component, Clone)]
pub struct Surface(Arc<SurfaceInner>);
struct SurfaceInner {
    instance: Instance,
    inner: vk::SurfaceKHR,
}
impl Drop for SurfaceInner {
    fn drop(&mut self) {
        unsafe {
            self.instance
                .extension::<KhrSurface>()
                .destroy_surface(self.inner, None);
        }
    }
}
impl Surface {
    pub fn create(
        instance: Instance,
        window_handle: &impl raw_window_handle::HasWindowHandle,
        display_handle: &impl raw_window_handle::HasDisplayHandle,
    ) -> VkResult<Surface> {
        let surface = unsafe {
            create_surface(
                &instance,
                display_handle.display_handle().unwrap(),
                window_handle.window_handle().unwrap(),
            )?
        };
        Ok(Surface(Arc::new(SurfaceInner {
            instance,
            inner: surface,
        })))
    }
    pub fn raw(&self) -> vk::SurfaceKHR {
        self.0.inner
    }
}

impl PhysicalDevice {
    pub fn get_surface_capabilities(
        &self,
        surface: &Surface,
    ) -> VkResult<vk::SurfaceCapabilitiesKHR> {
        unsafe {
            surface
                .0
                .instance
                .extension::<KhrSurface>()
                .get_physical_device_surface_capabilities(self.raw(), surface.0.inner)
        }
    }
    pub fn get_surface_formats(&self, surface: &Surface) -> VkResult<Vec<vk::SurfaceFormatKHR>> {
        unsafe {
            surface
                .0
                .instance
                .extension::<KhrSurface>()
                .get_physical_device_surface_formats(self.raw(), surface.0.inner)
        }
    }
    pub fn get_surface_present_modes(
        &self,
        surface: &Surface,
    ) -> VkResult<Vec<vk::PresentModeKHR>> {
        unsafe {
            surface
                .0
                .instance
                .extension::<KhrSurface>()
                .get_physical_device_surface_present_modes(self.raw(), surface.0.inner)
        }
    }
    pub fn supports_surface(&self, surface: &Surface, queue_family_index: u32) -> VkResult<bool> {
        unsafe {
            surface
                .0
                .instance
                .extension::<KhrSurface>()
                .get_physical_device_surface_support(
                    self.raw(),
                    queue_family_index,
                    surface.0.inner,
                )
        }
    }
}

pub(super) fn extract_surfaces(
    mut commands: Commands,
    instance: Res<Instance>,
    mut window_created_events: EventReader<bevy::window::WindowCreated>,
    query: Query<(&RawHandleWrapper, Option<&Surface>), With<Window>>,
    #[cfg(any(target_os = "macos", target_os = "ios"))] _marker: Option<
        NonSend<bevy::core::NonSendMarker>,
    >,
) {
    for create_event in window_created_events.read() {
        let (raw_handle, surface) = query.get(create_event.window).unwrap();
        let raw_handle = unsafe { raw_handle.get_handle() };
        assert!(surface.is_none());
        let new_surface = Surface::create(instance.clone(), &raw_handle, &raw_handle).unwrap();
        commands.entity(create_event.window).insert(new_surface);
    }
}

unsafe fn create_surface(
    instance: &Instance,
    display_handle: DisplayHandle,
    window_handle: WindowHandle,
) -> VkResult<vk::SurfaceKHR> {
    match (display_handle.as_raw(), window_handle.as_raw()) {
        (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(window)) => instance
            .extension::<khr::win32_surface::Meta>()
            .create_win32_surface(
                &vk::Win32SurfaceCreateInfoKHR {
                    hinstance: window.hinstance.unwrap().get() as ash::vk::HINSTANCE,
                    hwnd: window.hwnd.get() as ash::vk::HWND,
                    ..Default::default()
                },
                None,
            ),

        (RawDisplayHandle::Wayland(display), RawWindowHandle::Wayland(window)) => instance
            .extension::<khr::wayland_surface::Meta>()
            .create_wayland_surface(
                &vk::WaylandSurfaceCreateInfoKHR {
                    display: display.display.as_ptr(),
                    surface: window.surface.as_ptr(),
                    ..Default::default()
                },
                None,
            ),

        (RawDisplayHandle::Xlib(display), RawWindowHandle::Xlib(window)) => instance
            .extension::<khr::xlib_surface::Meta>()
            .create_xlib_surface(
                &vk::XlibSurfaceCreateInfoKHR {
                    dpy: display.display.unwrap().as_ptr() as *mut _,
                    window: window.window,
                    ..Default::default()
                },
                None,
            ),

        (RawDisplayHandle::Xcb(display), RawWindowHandle::Xcb(window)) => instance
            .extension::<khr::xcb_surface::Meta>()
            .create_xcb_surface(
                &vk::XcbSurfaceCreateInfoKHR {
                    connection: display.connection.unwrap().as_ptr(),
                    window: window.window.get(),
                    ..Default::default()
                },
                None,
            ),

        (RawDisplayHandle::Android(_), RawWindowHandle::AndroidNdk(window)) => instance
            .extension::<khr::android_surface::Meta>()
            .create_android_surface(
                &vk::AndroidSurfaceCreateInfoKHR {
                    window: window.a_native_window.as_ptr(),
                    ..Default::default()
                },
                None,
            ),

        #[cfg(target_os = "macos")]
        (RawDisplayHandle::AppKit(_), RawWindowHandle::AppKit(window)) => {
            use raw_window_metal::{appkit, Layer};

            let layer = match appkit::metal_layer_from_handle(window) {
                Layer::Existing(layer) | Layer::Allocated(layer) => layer as *mut _,
            };

            let surface_desc = vk::MetalSurfaceCreateInfoEXT::builder().layer(&*layer);
            instance
                .extension::<ash::ext::metal_surface::Meta>()
                .create_metal_surface(&surface_desc, None)
        }

        #[cfg(target_os = "ios")]
        (RawDisplayHandle::UiKit(_), RawWindowHandle::UiKit(window)) => {
            use raw_window_metal::{uikit, Layer};

            let layer = match uikit::metal_layer_from_handle(window) {
                Layer::Existing(layer) | Layer::Allocated(layer) => layer as *mut _,
            };

            let surface_desc = vk::MetalSurfaceCreateInfoEXT::builder().layer(&*layer);
            instance
                .extension::<ash::ext::MetalSurface>()
                .create_metal_surface(&surface_desc, None)
        }

        _ => Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT),
    }
}
