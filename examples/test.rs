use ash::vk;
use bevy_app::Update;
use bevy_ecs::{
    entity::Entity,
    query::With,
    schedule::{IntoSystemConfigs, IntoSystemSet},
    system::{In, IntoSystem, Query},
};
use bevy_window::PrimaryWindow;
use rhyolite::ecs::IntoRenderSystemConfigs;
use rhyolite::{
    acquire_swapchain_image,
    ecs::{Barriers, RenderApp, RenderCommands, RenderImage, RenderRes, RenderSystem},
    present, Access, RhyolitePlugin, SurfacePlugin, SwapchainConfig, SwapchainImage,
    SwapchainPlugin,
};

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
            mut windows: Query<&mut RenderImage<SwapchainImage>, With<bevy_window::PrimaryWindow>>,
        ) {
            let Ok(mut swapchain_image) = windows.get_single_mut() else {
                return;
            };

            commands
                .record_commands()
                .transition_resources()
                .image(
                    &mut swapchain_image,
                    Access {
                        stage: vk::PipelineStageFlags2::CLEAR,
                        access: vk::AccessFlags2::TRANSFER_WRITE,
                    },
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                )
                .end();
            commands.record_commands().clear_color_image(
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
            commands
                .record_commands()
                .transition_resources()
                .image(
                    &mut swapchain_image,
                    Access {
                        stage: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                        access: vk::AccessFlags2::empty(),
                    },
                    vk::ImageLayout::PRESENT_SRC_KHR,
                )
                .end();
        }
        clear_main_window_color.into_configs()
    }

    fn barriers(&self) -> rhyolite::ecs::BoxedBarrierProducer {
        fn barrier(
            In(mut barriers): In<Barriers>,
            mut windows: Query<&mut RenderImage<SwapchainImage>, With<bevy_window::PrimaryWindow>>,
        ) {
            let Ok(mut swapchain_image) = windows.get_single_mut() else {
                return;
            };
            barriers.transition_image(
                &mut *swapchain_image,
                Access {
                    stage: vk::PipelineStageFlags2::CLEAR,
                    access: vk::AccessFlags2::TRANSFER_WRITE,
                },
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            )
        }
        Box::new(IntoSystem::into_system(barrier))
    }
}
