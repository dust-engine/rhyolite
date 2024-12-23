use bevy::prelude::{Commands, IntoSystem, IntoSystemConfigs, Local, SystemParamFunction};
use rhyolite::ash::vk;

use bevy::app::{PluginGroup, PostUpdate};
use bevy::ecs::system::Query;
use bevy::ecs::{entity::Entity, query::With};
use bevy::window::PrimaryWindow;
use rhyolite::command::states::Pending;
use rhyolite::command::{CommandBuffer, SharedCommandPool, Timeline};
use rhyolite::debug::DebugUtilsPlugin;
use rhyolite::future::commands::{clear_color_image, prepare_image_for_presentation, yield_now};
use rhyolite::future::{GPUFutureBlock};
use rhyolite::selectors::{Graphics, UniversalCompute};
use rhyolite::ImageLike;
use rhyolite::{
    ecs2::IntoRenderSystem,
    swapchain::{
        acquire_swapchain_image, present, SwapchainConfig, SwapchainImage, SwapchainPlugin,
    },
    RhyolitePlugin, SurfacePlugin,
};
use rhyolite::{future::gpu_future, Queue};

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

    app.get_schedule_mut(PostUpdate)
        .as_mut()
        .unwrap()
        .add_build_pass(rhyolite::ecs2::RenderSystemsPass::new())
        .before::<bevy::ecs::schedule::passes::AutoInsertApplyDeferredPass>();

    test(clear_main_window_color);

    app.add_systems(
        PostUpdate,
        clear_main_window_color
            .into_render_system::<UniversalCompute>()
            .in_set(rhyolite::swapchain::SwapchainSystemSet),
    );

    app.run();
}

fn an_actual_system(commands: Commands) {

}


fn test<Marker, S: SystemParamFunction<Marker>>(s: S) {

}



fn clear_main_window_color(
    mut windows: Query< &mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut state: Local<u64>,
) -> impl GPUFutureBlock {
    *state += 1;
    if *state >= 100 {
        *state = 0;
    }

    let state = *state;
    let mut swapchain_image = windows.get_single_mut().unwrap();
    let swapchain_image = swapchain_image.raw_image();

    gpu_future! { move
        let swapchain_image = swapchain_image;
        clear_color_image(&swapchain_image, vk::ClearColorValue {
            float32: [0.0, state as f32 / 100.0, 0.0, 1.0],
        }, &[vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }]).await;
        //prepare_image_for_presentation(&swapchain_image).await;
    }
}
// how do we calculate semaphore stuff?
// the wait stages are the first stage in which the resource was used.
// the signal stages are always all stages.
// when we hook up one system after another there's a reason for it.
