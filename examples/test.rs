use rhyolite::RhyolitePlugin;

fn main() {
    bevy_app::App::new()
        .add_plugins(RhyolitePlugin::new().unwrap())
        .run();
}
