use bevy::prelude::{IntoSystemConfigs, Local};
use rhyolite::ash::vk;

use bevy::app::{PluginGroup, PostUpdate};
use bevy::ecs::system::{In, Query};
use bevy::ecs::{entity::Entity, query::With};
use bevy::window::PrimaryWindow;
use rhyolite::command::{self, SharedCommandPool, Timeline};
use rhyolite::future::commands::clear_color_image;
use rhyolite::debug::DebugUtilsPlugin;
use rhyolite::selectors::Graphics;
use rhyolite::{Queue, future::gpu_future};
use rhyolite::{SurfacePlugin, RhyolitePlugin, swapchain::{SwapchainPlugin, acquire_swapchain_image, present, SwapchainConfig, SwapchainImage}};

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

    app.add_systems(
        PostUpdate,
        clear_main_window_color
            .after(acquire_swapchain_image::<With<PrimaryWindow>>)
            .before(present),
    );

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

    app.run();
}

fn clear_main_window_color(
    mut queue: Queue<Graphics>,
    mut command_pool: SharedCommandPool<Graphics>,
    windows: Query<&SwapchainImage, With<bevy::window::PrimaryWindow>>,

    timeline: Local<Timeline>,
) {
    let Ok(swapchain_image) = windows.get_single() else {
        return;
    };


    let mut command_encoder = command_pool.start_encoding(&timeline, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT).unwrap();

    // swapchain image won't be destroyed until it's presented.
    // swapchain image won't be presented until drawing is finished on it.
    // so it's ok to use directly here.
    command_encoder.record(gpu_future! {
        clear_color_image(swapchain_image, vk::ClearColorValue {
            float32: [0.0, 0.4, 0.0, 1.0],
        }, &[vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }]).await;
    });



    let command_buffer = command_encoder.end().unwrap();
    let submission = queue.submit_one_and_present(command_buffer, &[
        swapchain_image.blocking_stages(vk::PipelineStageFlags2::CLEAR)
    ], &swapchain_image).unwrap();
    let submission = submission.unwrap_blocked();
    command_pool.recycle(submission);
}
