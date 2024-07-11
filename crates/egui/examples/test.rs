use rhyolite::ash::vk;

use bevy::app::{PluginGroup, Update};
use bevy::ecs::system::Local;
use bevy::ecs::{entity::Entity, query::With};
use bevy::window::PrimaryWindow;
use rhyolite::debug::DebugUtilsPlugin;
use rhyolite::{RhyolitePlugin, SurfacePlugin, SwapchainConfig, SwapchainPlugin};
use rhyolite_egui::{egui, EguiContexts};

fn main() {
    let mut app = bevy::app::App::new();
    app.add_plugins(bevy::DefaultPlugins.set::<bevy::asset::AssetPlugin>(
        bevy::asset::AssetPlugin {
            mode: bevy::asset::AssetMode::Processed,
            ..Default::default()
        },
    ))
    .add_plugins(SurfacePlugin::default())
    .add_plugins(DebugUtilsPlugin::default())
    .add_plugins(RhyolitePlugin::default())
    .add_plugins(SwapchainPlugin::default())
    .add_plugins(rhyolite_egui::EguiPlugin::<With<PrimaryWindow>>::default());

    app.add_systems(Update, ui_example_system);

    let mut primary_window = app
        .world_mut()
        .query_filtered::<Entity, With<PrimaryWindow>>();
    let primary_window = primary_window
        .iter(app.world())
        .next()
        .unwrap();
    app.world_mut()
        .entity_mut(primary_window)
        .insert(SwapchainConfig {
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..Default::default()
        });

    app.run();
}

#[derive(Default)]
struct UIState {
    name: String,
    age: u32,
}
fn ui_example_system(mut contexts: EguiContexts, mut state: Local<UIState>) {
    egui::Window::new("hello").show(contexts.ctx_mut(), |ui| {
        ui.heading("My egui Application");
        ui.horizontal(|ui| {
            let name_label = ui.label("Your name: ");
            ui.text_edit_singleline(&mut state.name)
                .labelled_by(name_label.id);
        });
        ui.add(egui::Slider::new(&mut state.age, 0..=120).text("age"));
        if ui.button("Increment").clicked() {
            state.age += 1;
        }
        ui.label(format!("Hello '{}', age {}", state.name, state.age));
    });
}
