use bevy::asset::{AssetServer, Assets};
use bevy::ecs::schedule::IntoSystemConfigs;
use bevy::math::UVec3;
use rhyolite::ash::vk;

use bevy::app::{PluginGroup, PostUpdate, Startup};
use bevy::ecs::system::{Commands, In, Query, Res, ResMut, Resource};
use bevy::ecs::{entity::Entity, query::With};
use bevy::window::PrimaryWindow;
use rhyolite::commands::{CommonCommands, ComputeCommands, ResourceTransitionCommands};
use rhyolite::debug::DebugUtilsPlugin;
use rhyolite::dispose::RenderObject;
use rhyolite::ecs::{Barriers, IntoRenderSystemConfigs, RenderCommands};
use rhyolite::pipeline::{CachedPipeline, ComputePipeline, ComputePipelineCreateInfo, DescriptorSetLayout, PipelineCache, PipelineLayout};
use rhyolite::shader::{ShaderModule, SpecializedShader};
use rhyolite::{
    acquire_swapchain_image, present, Access, DeferredOperationTaskPool, Device, RhyolitePlugin, SurfacePlugin, SwapchainConfig, SwapchainImage, SwapchainPlugin
};

use std::sync::Arc;

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
        run_compute_shader
            .with_barriers(run_compute_shader_barrier)
            .after(acquire_swapchain_image::<With<PrimaryWindow>>)
            .before(present),
    )
    .add_systems(Startup, initialize_pipeline);

    let primary_window = app
        .world
        .query_filtered::<Entity, With<PrimaryWindow>>()
        .iter(&app.world)
        .next()
        .unwrap();
    app.world
        .entity_mut(primary_window)
        .insert(SwapchainConfig {
            image_usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..Default::default()
        });

    app.run();
}

#[derive(Resource)]
struct GameOfLifePipeline(CachedPipeline<RenderObject<ComputePipeline>>);
fn initialize_pipeline(
    mut commands: Commands,
    device: Res<Device>,
    pipeline_cache: Res<PipelineCache>,
    assets: Res<AssetServer>,
) {
    let desc0 = DescriptorSetLayout::new(
        device.clone(),
        &playout_macro::layout!("../assets/game_of_life.playout", 0),
        vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR,
    )
    .unwrap();

    let layout = PipelineLayout::new(
        device.clone(),
        vec![Arc::new(desc0)],
        &[vk::PushConstantRange {
            offset: 0,
            size: std::mem::size_of::<[f32; 2]>() as u32,
            stage_flags: vk::ShaderStageFlags::VERTEX,
        }], // Ideally this can be specified automatically
        vk::PipelineLayoutCreateFlags::empty(),
    )
    .unwrap();
    let layout = Arc::new(layout);
    let pipeline = pipeline_cache.create_compute(ComputePipelineCreateInfo {
        device: device.clone(),
        shader: SpecializedShader {
            stage: vk::ShaderStageFlags::FRAGMENT,
            shader: assets.load("game_of_life.comp"),
            ..Default::default()
        },
        layout: layout,
        flags: vk::PipelineCreateFlags::empty(),
    });
    commands.insert_resource(GameOfLifePipeline(pipeline));

}

fn run_compute_shader_barrier(
    In(mut barriers): In<Barriers>,
    mut windows: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
) {
    let Ok(swapchain) = windows.get_single_mut() else {
        return;
    };
    barriers.transition(
        swapchain.into_inner(),
        Access {
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_WRITE,
        },
        false,
        vk::ImageLayout::GENERAL,
    );
}
fn run_compute_shader(
    mut commands: RenderCommands<'u'>,
    mut pipeline: ResMut<GameOfLifePipeline>,
    pipeline_cache: Res<PipelineCache>,
    task_pool: Res<DeferredOperationTaskPool>,
    assets: Res<Assets<ShaderModule>>,
    windows: Query<&SwapchainImage, With<bevy::window::PrimaryWindow>>,
) {
    let Ok(swapchain) = windows.get_single() else {
        return;
    };
    let Some(pipeline) = pipeline_cache.retrieve(&mut pipeline.0, assets.into_inner(), task_pool.into_inner()) else {
        return;
    };
    commands.bind_pipeline(pipeline);
    commands.dispatch(UVec3::new(10, 10, 10));
}
