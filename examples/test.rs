use std::ops::DerefMut;

use ash::vk;

use bevy::app::{PluginGroup, Update};
use bevy::ecs::system::Local;
use bevy::ecs::{
    entity::Entity,
    query::With,
    system::{In, Query},
};
use bevy::window::PrimaryWindow;
use rhyolite::commands::ResourceTransitionCommands;
use rhyolite::debug::DebugUtilsPlugin;
use rhyolite::{
    commands::CommonCommands,
    ecs::{Barriers, RenderCommands},
    Access, RhyolitePlugin, SurfacePlugin, SwapchainConfig, SwapchainImage, SwapchainPlugin,
};
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

    /*
    app.add_systems(
        PostUpdate,
        clear_main_window_color
            .with_barriers(clear_main_window_color_barrier)
            .after(acquire_swapchain_image::<With<PrimaryWindow>>)
            .before(rhyolite_egui::draw::<With<PrimaryWindow>>)
            .before(present),
    );
    */

    let primary_window = app
        .world
        .query_filtered::<Entity, With<PrimaryWindow>>()
        .iter(&app.world)
        .next()
        .unwrap();
    app.world
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
fn clear_main_window_color(
    mut commands: RenderCommands<'g'>,
    mut windows: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
) {
    let Ok(swapchain_image) = windows.get_single_mut() else {
        return;
    };
    commands.clear_color_image(
        swapchain_image.image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &vk::ClearColorValue {
            float32: [0.0, 0.0, 1.0, 1.0],
        },
        &[vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }],
    );
}

fn clear_main_window_color_barrier(
    In(mut barriers): In<Barriers>,
    mut windows: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
) {
    let Ok(mut swapchain_image) = windows.get_single_mut() else {
        return;
    };
    barriers.transition(
    swapchain_image.into_inner().deref_mut(),
        Access {
            stage: vk::PipelineStageFlags2::CLEAR,
            access: vk::AccessFlags2::TRANSFER_WRITE,
        },
        false,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );
}
