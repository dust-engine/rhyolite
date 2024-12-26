#![feature(let_chains)]
use bevy::asset::{AssetServer, Assets};
use bevy::core::FrameCount;
use bevy::ecs::schedule::IntoSystemConfigs;
use bevy::prelude::{Mut, Query};
use rhyolite::ash::vk;
use rhyolite::swapchain::SwapchainImage;
use std::ops::Deref;

use bevy::app::{PluginGroup, PostUpdate, Startup};
use bevy::ecs::system::{Commands, Local, Res, ResMut, Resource};
use bevy::ecs::{entity::Entity, query::With};
use bevy::window::PrimaryWindow;
use rhyolite::debug::DebugUtilsPlugin;
use rhyolite::pipeline::{
    CachedPipeline, ComputePipeline, ComputePipelineCreateInfo, DescriptorSetLayout, PipelineCache,
    PipelineLayout,
};
use rhyolite::shader::{ShaderModule, SpecializedShader};
use rhyolite::{
    swapchain::{SwapchainConfig, SwapchainPlugin},
    Allocator, DeferredOperationTaskPool, Device, Image, RhyoliteApp, RhyolitePlugin,
    SurfacePlugin,
};

use rhyolite::commands::{blit_image, record_commands};
use rhyolite::ecs::IntoRenderSystem;
use rhyolite::future::{GPUBorrowedResource, GPUFutureBlock};
use rhyolite::selectors::UniversalCompute;
use rhyolite_macros::gpu_future;
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

    app.add_device_extension::<ash::khr::push_descriptor::Meta>()
        .unwrap();

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

            // We set this to true because the image will be used as storage image.
            // Generally, sRGB image formats cannot be used as storage image.
            srgb_format: false,
            ..Default::default()
        });

    app.add_systems(Startup, initialize_pipeline);
    app.add_systems(
        PostUpdate,
        run_compute_shader
            .into_render_system::<UniversalCompute>()
            .in_set(rhyolite::swapchain::SwapchainSystemSet),
    );

    app.run();
}

#[derive(Resource)]
struct GameOfLifePipeline {
    run_pipeline: CachedPipeline<ComputePipeline>,
    init_pipeline: CachedPipeline<ComputePipeline>,
    game: GPUBorrowedResource<Image>,
    layout: Arc<PipelineLayout>,
}
fn initialize_pipeline(
    mut commands: Commands,
    device: Res<Device>,
    pipeline_cache: Res<PipelineCache>,
    assets: Res<AssetServer>,
    allocator: Res<Allocator>,
) {
    let desc0 = DescriptorSetLayout::new(
        device.clone(),
        &playout_macro::layout!("../assets/game_of_life/game_of_life.playout", 0),
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
    let run_pipeline = pipeline_cache.create_compute(ComputePipelineCreateInfo {
        device: device.clone(),
        shader: SpecializedShader {
            stage: vk::ShaderStageFlags::COMPUTE,
            shader: assets.load("game_of_life/game_of_life.comp"),
            ..Default::default()
        },
        layout: layout.clone(),
        flags: vk::PipelineCreateFlags::empty(),
    });

    let init_pipeline = pipeline_cache.create_compute(ComputePipelineCreateInfo {
        device: device.clone(),
        shader: SpecializedShader {
            stage: vk::ShaderStageFlags::COMPUTE,
            shader: assets.load("game_of_life/game_of_life_init.comp"),
            ..Default::default()
        },
        layout: layout.clone(),
        flags: vk::PipelineCreateFlags::empty(),
    });

    let game_img_create_info = vk::ImageCreateInfo {
        image_type: vk::ImageType::TYPE_2D,
        format: vk::Format::R8G8B8A8_UNORM,
        extent: vk::Extent3D {
            width: 192,
            height: 108,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        initial_layout: vk::ImageLayout::UNDEFINED,
        ..Default::default()
    };
    commands.insert_resource(GameOfLifePipeline {
        init_pipeline,
        run_pipeline,
        game: GPUBorrowedResource::new(
            Image::new_device_image(allocator.clone(), &game_img_create_info).unwrap(),
        ),
        layout,
    });
}

fn run_compute_shader<'w, 's>(
    game_of_life_pipeline: ResMut<'w, GameOfLifePipeline>,
    pipeline_cache: Res<'w, PipelineCache>,
    task_pool: Res<'w, DeferredOperationTaskPool>,
    assets: Res<'w, Assets<ShaderModule>>,
    mut initialized: Local<'s, bool>,
    frame_index: Res<'w, FrameCount>,
    mut windows: Query<'w, 's, &'w mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
) -> impl GPUFutureBlock + use<'w, 's> {
    let game_of_life_pipeline = game_of_life_pipeline.into_inner();
    gpu_future! { move
        let pipeline = pipeline_cache.retrieve(
            if *initialized {
                &mut game_of_life_pipeline.run_pipeline
            } else {
                &mut game_of_life_pipeline.init_pipeline
            },
            assets.into_inner(),
            task_pool.into_inner(),
        );
        if let Some(pipeline) = pipeline && (frame_index.0 % 30 == 0 || !*initialized) {
            *initialized = true;
            record_commands(
                &mut game_of_life_pipeline.game,
                |mut ctx, game| unsafe {
                    ctx.bind_pipeline(pipeline.deref());
                    ctx.device.extension::<ash::khr::push_descriptor::Meta>()
                        .cmd_push_descriptor_set(
                            ctx.command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            game_of_life_pipeline.layout.raw(),
                            0,
                            &[vk::WriteDescriptorSet {
                                dst_binding: 0,
                                descriptor_count: 1,
                                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                                p_image_info: [vk::DescriptorImageInfo {
                                    image_view: game.view,
                                    image_layout: vk::ImageLayout::GENERAL,
                                    ..Default::default()
                                }]
                                .as_ptr(),
                                ..Default::default()
                            }],
                        );
                    ctx.device.cmd_dispatch(ctx.command_buffer, 192, 108, 1);
            }, |mut ctx, game| {
                ctx.use_image_resource(
                    *game,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ,
                    vk::ImageLayout::GENERAL,
                    true,
                )
            }).await;
        }


        let Some(swapchain_image): Option<Mut<'w, SwapchainImage>> = windows.get_single_mut().ok() else {
            return;
        };
        blit_image(&mut game_of_life_pipeline.game, &mut swapchain_image.into_inner()).await;
    }
}
