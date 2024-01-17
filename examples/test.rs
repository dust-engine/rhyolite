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
    app.finish();
    app.cleanup();

    //bevy_mod_debugdump::print_schedule_graph(&mut app, bevy_app::Update);
    app.run();
}


fn clear_main_window_color(
    mut commands: RenderCommands<'g'>,
    mut windows: Query<RenderComponent<SwapchainImage>, With<bevy_window::PrimaryWindow>>,
) {
    let window = windows.single();
}
