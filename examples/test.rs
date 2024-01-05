use rhyolite::{RhyolitePlugin, SurfacePlugin};

fn main() {
    bevy_app::App::new()
        .add_plugins(SurfacePlugin::default())
        .add_plugins(RhyolitePlugin::default())
        .run();
}
