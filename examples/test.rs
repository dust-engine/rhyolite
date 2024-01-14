use rhyolite::{RhyolitePlugin, SurfacePlugin, SwapchainPlugin};

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
    app.finish();
    app.cleanup();

    //bevy_mod_debugdump::print_schedule_graph(&mut app, bevy_app::Update);
    app.run();
}
