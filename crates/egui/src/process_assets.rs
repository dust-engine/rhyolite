use bevy::{app::Startup, asset::processor::AssetProcessor, prelude::Res, tasks::block_on};

fn main() {
    let mut app = bevy::app::App::new();
    app.add_plugins((
        bevy::log::LogPlugin::default(),
        bevy::core::TaskPoolPlugin::default(),
        bevy::asset::AssetPlugin {
            mode: bevy::asset::AssetMode::Processed,
            ..Default::default()
        },
        rhyolite::shader::loader::GlslPlugin {
            target_vk_version: rhyolite::Version::V1_3,
        },
    ));
    app.add_systems(Startup, wait_for_processor);
    app.run();
}

fn wait_for_processor(processor: Res<AssetProcessor>) {
    block_on(processor.data().wait_until_finished());
}
