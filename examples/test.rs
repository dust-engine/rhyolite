use ash::vk;
use bevy_app::Update;
use bevy_ecs::{system::Query, query::With, schedule::IntoSystemConfigs};
use rhyolite::{RhyolitePlugin, SurfacePlugin, SwapchainPlugin, ecs::{RenderCommands, RenderComponent}, SwapchainImage, acquire_swapchain_image, present};

fn main() {
    let mut app = bevy_app::App::new();
    app
        .add_plugins(bevy_window::WindowPlugin::default())
        .add_plugins(bevy_a11y::AccessibilityPlugin)
        .add_plugins(bevy_winit::WinitPlugin::default())
        .add_plugins(bevy_input::InputPlugin::default())
        .add_plugins(SurfacePlugin::default())
        .add_plugins(RhyolitePlugin::default())
        .add_plugins(SwapchainPlugin::default());

    app.add_systems(Update, clear_main_window_color.after(acquire_swapchain_image).before(present));
    //app.add_systems(Update, clear_main_window_color2.after(clear_main_window_color).before(present));
    app.finish();
    app.cleanup();

    //bevy_mod_debugdump::print_schedule_graph(&mut app, bevy_app::Update);
    app.run();
}


fn clear_main_window_color(
    mut commands: RenderCommands<'g'>,
    windows: Query<RenderComponent<SwapchainImage>, With<bevy_window::PrimaryWindow>>,
) {
    println!("Recording...");
    let Ok(swapchain_image) = windows.get_single() else { return };
    commands
    .record_commands()
    .clear_color_image(swapchain_image.inner.image, vk::ImageLayout::PRESENT_SRC_KHR, &vk::ClearColorValue {
        float32: [0.0, 0.0, 1.0, 1.0]
    }, &[
        vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }
    ])
}
fn clear_main_window_color2(
    mut commands: RenderCommands<'c'>,
    mut windows: Query<RenderComponent<SwapchainImage>, With<bevy_window::PrimaryWindow>>,
) {
    let window = windows.get_single() else { return };
}
