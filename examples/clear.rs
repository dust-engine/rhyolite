use bevy::prelude::{EventReader, IntoSystemConfigs, Local};
use rhyolite::ash::vk;

use bevy::app::{AppExit, PluginGroup, PostUpdate};
use bevy::ecs::system::{In, Query};
use bevy::ecs::{entity::Entity, query::With};
use bevy::window::PrimaryWindow;
use rhyolite::command::states::Pending;
use rhyolite::command::{self, CommandBuffer, SharedCommandPool, Timeline};
use rhyolite::debug::DebugUtilsPlugin;
use rhyolite::future::commands::{clear_color_image, prepare_image_for_presentation};
use rhyolite::future::InFlightFrameMananger;
use rhyolite::selectors::Graphics;
use rhyolite::{future::gpu_future, Queue};
use rhyolite::{
    swapchain::{
        acquire_swapchain_image, present, SwapchainConfig, SwapchainImage, SwapchainPlugin,
    },
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
    mut windows: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,

    timeline: Local<Timeline>,
    mut frames: Local<InFlightFrameMananger<CommandBuffer<Pending>>>,
    mut state: Local<u64>,
) {
    let Ok(mut swapchain_image) = windows.get_single_mut() else {
        return;
    };
    let swapchain_image = &mut *swapchain_image;

    frames
        .take()
        .map(|command_buffer| command_pool.free(command_buffer.wait_for_completion()));

    let mut command_buffer = command_pool
        .allocate(&timeline, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
        .unwrap();

    *state += 1;
    if *state >= 100 {
        *state = 0;
    }

    // swapchain image won't be destroyed until it's presented.
    // swapchain image won't be presented until drawing is finished on it.
    // so it's ok to use directly here.
    command_pool.record(
        &mut command_buffer,
        gpu_future! {
            clear_color_image(&mut *swapchain_image, vk::ClearColorValue {
                float32: [0.0, *state as f32 / 100.0, 0.0, 1.0],
            }, &[vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            }]).await;
            prepare_image_for_presentation(&mut *swapchain_image).await;
        },
    );

    let submission = queue
        .submit_one_and_present(
            command_pool.end(command_buffer),
            &[swapchain_image.blocking_stages(vk::PipelineStageFlags2::CLEAR)],
            &swapchain_image,
        )
        .unwrap();
    frames.next_frame(submission);
}
// how do we calculate semaphore stuff?
// the wait stages are the first stage in which the resource was used.
// the signal stages are always all stages.
// when we hook up one system after another there's a reason for it.
