use bevy::prelude::{IntoSystemConfigs, Local, Mut, SystemParamFunction};
use rhyolite::ash::vk;

use bevy::app::{PluginGroup, PostUpdate};
use bevy::ecs::system::Query;
use bevy::ecs::{entity::Entity, query::With};
use bevy::window::PrimaryWindow;
use rhyolite::debug::DebugUtilsPlugin;
use rhyolite::future::commands::clear_color_image;
use rhyolite::future::gpu_future;
use rhyolite::future::GPUFutureBlock;
use rhyolite::selectors::UniversalCompute;
use rhyolite::{
    ecs::IntoRenderSystem,
    swapchain::{SwapchainConfig, SwapchainImage, SwapchainPlugin},
    RhyolitePlugin, SurfacePlugin,
};

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
    .add_plugins(SwapchainPlugin::default());

    let primary_window = app
        .world_mut()
        .query_filtered::<Entity, With<PrimaryWindow>>()
        .iter(app.world())
        .next()
        .unwrap();
    app.world_mut()
        .entity_mut(primary_window)
        .insert(SwapchainConfig {
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..Default::default()
        });

    app.add_systems(
        PostUpdate,
        clear_main_window_color
            .into_render_system::<UniversalCompute>()
            .in_set(rhyolite::swapchain::SwapchainSystemSet),
    );

    app.run();
}

fn clear_main_window_color<'w, 's>(
    mut windows: Query<'w, 's, &'w mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut state: Local<'s, u64>,
) -> impl GPUFutureBlock + use<'w, 's> {
    gpu_future! { move
        let Some(mut swapchain_image): Option<Mut<'w, SwapchainImage>> = windows.get_single_mut().ok() else {
            return;
        };
        *state += 1;
        if *state >= 100 {
            *state = 0;
        }
        clear_color_image(&mut *swapchain_image, vk::ClearColorValue {
            float32: [0.0, *state as f32 / 100.0, 0.0, 1.0],
        }, &[vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }]).await;
    }
}
