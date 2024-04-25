use bevy::asset::{AssetServer, Assets};
use bevy::core::FrameCount;
use bevy::ecs::schedule::IntoSystemConfigs;
use bevy::math::UVec3;
use rhyolite::ash::vk;

use bevy::app::{PluginGroup, PostUpdate, Startup};
use bevy::ecs::system::{Commands, In, Local, Query, Res, ResMut, Resource};
use bevy::ecs::{entity::Entity, query::With};
use bevy::window::PrimaryWindow;
use rhyolite::commands::{
    CommonCommands, ComputeCommands, GraphicsCommands, ResourceTransitionCommands,
};
use rhyolite::debug::DebugUtilsPlugin;
use rhyolite::dispose::RenderObject;
use rhyolite::ecs::{Barriers, IntoRenderSystemConfigs, RenderCommands, RenderImage};
use rhyolite::pipeline::{
    CachedPipeline, ComputePipeline, ComputePipelineCreateInfo, DescriptorSetLayout, PipelineCache,
    PipelineLayout,
};
use rhyolite::shader::{ShaderModule, SpecializedShader};
use rhyolite::{
    acquire_swapchain_image, present, Access, Allocator, DeferredOperationTaskPool, Device, Image,
    RhyoliteApp, RhyolitePlugin, SurfacePlugin, SwapchainConfig, SwapchainImage, SwapchainPlugin,
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

    app.add_device_extension::<ash::khr::push_descriptor::Meta>()
        .unwrap();

    app.add_systems(
        PostUpdate,
        (
            run_compute_shader
                .with_barriers(run_compute_shader_barrier)
                .run_if(|count: Res<FrameCount>| count.0 % 30 == 0),
            blit_image_to_swapchain
                .with_barriers(blit_image_to_swapchain_barrier)
                .after(acquire_swapchain_image::<With<PrimaryWindow>>)
                .before(present)
                .after(run_compute_shader),
        ),
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
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,

            // We set this to true because the image will be used as storage image.
            // Generally, sRGB image formats cannot be used as storage image.
            srgb_format: false,
            ..Default::default()
        });

    app.run();
}

#[derive(Resource)]
struct GameOfLifePipeline {
    run_pipeline: CachedPipeline<RenderObject<ComputePipeline>>,
    init_pipeline: CachedPipeline<RenderObject<ComputePipeline>>,
    game: RenderImage<Image>,
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
        game: RenderImage::new(
            Image::new_device_image(allocator.clone(), &game_img_create_info).unwrap(),
        ),
        layout,
    });
}

fn run_compute_shader_barrier(
    In(mut barriers): In<Barriers>,
    mut game_of_life_pipeline: ResMut<GameOfLifePipeline>,
) {
    barriers.transition(
        &mut game_of_life_pipeline.game,
        Access {
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ,
        },
        true,
        vk::ImageLayout::GENERAL,
    );
}
fn run_compute_shader(
    mut commands: RenderCommands<'u'>,
    game_of_life_pipeline: ResMut<GameOfLifePipeline>,
    pipeline_cache: Res<PipelineCache>,
    task_pool: Res<DeferredOperationTaskPool>,
    assets: Res<Assets<ShaderModule>>,
    mut initialized: Local<bool>,
) {
    let game_of_life_pipeline = game_of_life_pipeline.into_inner();
    let Some(pipeline) = pipeline_cache.retrieve(
        if *initialized {
            &mut game_of_life_pipeline.run_pipeline
        } else {
            &mut game_of_life_pipeline.init_pipeline
        },
        assets.into_inner(),
        task_pool.into_inner(),
    ) else {
        return;
    };
    commands.bind_pipeline(pipeline);
    commands.push_descriptor_set(
        &game_of_life_pipeline.layout,
        0,
        &[vk::WriteDescriptorSet {
            dst_binding: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            p_image_info: [vk::DescriptorImageInfo {
                image_view: game_of_life_pipeline.game.view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            }]
            .as_ptr(),
            ..Default::default()
        }],
        vk::PipelineBindPoint::COMPUTE,
    );
    commands.dispatch(UVec3::new(192, 108, 1));
    *initialized = true;
}

fn blit_image_to_swapchain_barrier(
    In(mut barriers): In<Barriers>,
    mut game_of_life_pipeline: ResMut<GameOfLifePipeline>,
    mut windows: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
) {
    let Ok(swapchain) = windows.get_single_mut() else {
        return;
    };
    barriers.transition(
        swapchain.into_inner(),
        Access {
            stage: vk::PipelineStageFlags2::BLIT,
            access: vk::AccessFlags2::TRANSFER_WRITE,
        },
        false,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );
    barriers.transition(
        &mut game_of_life_pipeline.game,
        Access {
            stage: vk::PipelineStageFlags2::BLIT,
            access: vk::AccessFlags2::TRANSFER_READ,
        },
        true,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    );
}
fn blit_image_to_swapchain(
    mut commands: RenderCommands<'g'>,
    windows: Query<&SwapchainImage, With<bevy::window::PrimaryWindow>>,
    game_of_life_pipeline: Res<GameOfLifePipeline>,
) {
    let Ok(swapchain) = windows.get_single() else {
        return;
    };
    commands.blit_image(&game_of_life_pipeline.game, swapchain, vk::Filter::NEAREST);
}
