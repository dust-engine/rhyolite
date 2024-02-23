use ash::vk;
use bevy_app::Update;
use bevy_ecs::{entity::Entity, query::With, schedule::{IntoSystemConfigs, IntoSystemSet}, system::{In, IntoSystem, Query}};
use bevy_window::PrimaryWindow;
use rhyolite::{
    acquire_swapchain_image,
    ecs::{Barriers, RenderApp, RenderCommands, RenderComponent, RenderSystem},
    present, RhyolitePlugin, SurfacePlugin, SwapchainConfig, SwapchainImage, SwapchainPlugin,
};
use rhyolite::ecs::IntoRenderSystemConfigs;


fn main() {
    let mut app = bevy_app::App::new();
    app.add_plugins(bevy_window::WindowPlugin::default())
        .add_plugins(bevy_a11y::AccessibilityPlugin)
        .add_plugins(bevy_winit::WinitPlugin::default())
        .add_plugins(bevy_input::InputPlugin::default())
        .add_plugins(SurfacePlugin::default())
        .add_plugins(RhyolitePlugin::default())
        .add_plugins(SwapchainPlugin::default());

    app.add_render_system(
        ClearMainWindowColor
            .after(acquire_swapchain_image::<With<PrimaryWindow>>)
            .before(present),
    );

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

    //bevy_mod_debugdump::print_schedule_graph(&mut app, bevy_app::Update);
    app.run();
}


// Solution: Each render system will be told the transitions to perform.
// However, only one of them will actually perform the transitions, based on
// runtime scheduling behavior.
// Some systems will have a final "transition out". This "transition out" will be performed
// by the "flush" system.


struct ClearMainWindowColor;
impl RenderSystem for ClearMainWindowColor {
    fn system(&self) -> bevy_ecs::schedule::SystemConfigs {
        fn clear_main_window_color(
            mut commands: RenderCommands<'g'>,
            windows: Query<RenderComponent<SwapchainImage>, With<bevy_window::PrimaryWindow>>,
        ) {
            let Ok(swapchain_image) = windows.get_single() else {
                return;
            };
            commands.record_commands().pipeline_barrier(
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier2 {
                    src_stage_mask: vk::PipelineStageFlags2::CLEAR, // Last time this image was touched by a CLEAR.
                    dst_stage_mask: vk::PipelineStageFlags2::CLEAR,
                    src_access_mask: vk::AccessFlags2::empty(),
                    dst_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    image: swapchain_image.inner.image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }],
            );
            commands.record_commands().clear_color_image(
                swapchain_image.inner.image,
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
            commands.record_commands().pipeline_barrier(
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier2 {
                    src_stage_mask: vk::PipelineStageFlags2::CLEAR,
                    dst_stage_mask: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags2::empty(),
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    image: swapchain_image.inner.image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }],
            );
        }
        clear_main_window_color.into_configs()        
    }

    fn barriers(&self) -> rhyolite::ecs::BoxedBarrierProducer {
        fn empty(In(input): In<Barriers>) {
            println!("lalala pipeline barrier");
        }
        Box::new(IntoSystem::into_system(empty))
    }
}

