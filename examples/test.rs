use ash::vk;

use bevy::app::Update;
use bevy::ecs::{
    entity::Entity,
    query::With,
    schedule::IntoSystemConfigs,
    system::{In, IntoSystem, Query},
};
use bevy::window::PrimaryWindow;
use rhyolite::{
    acquire_swapchain_image,
    ecs::{Barriers, RenderApp, RenderCommands, RenderSystem},
    present, Access, RhyolitePlugin, SurfacePlugin, SwapchainConfig, SwapchainImage,
    SwapchainPlugin,
};
use rhyolite::{debug::DebugUtilsPlugin, ecs::IntoRenderSystemConfigs};
use rhyolite_egui::{egui, EguiContexts};

fn main() {
    let mut app = bevy::app::App::new();
    app.add_plugins(bevy::window::WindowPlugin::default())
        .add_plugins(bevy::a11y::AccessibilityPlugin)
        .add_plugins(bevy::winit::WinitPlugin::default())
        .add_plugins(bevy::input::InputPlugin::default())
        .add_plugins(SurfacePlugin::default())
        .add_plugins(DebugUtilsPlugin::default())
        .add_plugins(RhyolitePlugin::default())
        .add_plugins(SwapchainPlugin::default())
        .add_plugins(rhyolite_egui::EguiPlugin::<With<PrimaryWindow>>::default());

    app.add_systems(Update, ui_example_system);

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

    app.run();
}


fn ui_example_system(mut contexts: EguiContexts) {
    egui::Window::new("Hello").show(contexts.ctx_mut(), |ui| {
        ui.label("world");
    });
}

struct ClearMainWindowColor;
impl RenderSystem for ClearMainWindowColor {
    fn system(&self) -> bevy::ecs::schedule::SystemConfigs {
        fn clear_main_window_color(
            mut commands: RenderCommands<'g'>,
            mut windows: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
        ) {
            let Ok(mut swapchain_image) = windows.get_single_mut() else {
                return;
            };
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
                    true,
                )
                .end();
        }
        clear_main_window_color.into_configs()
    }

    fn barriers(&self) -> rhyolite::ecs::BoxedBarrierProducer {
        fn barrier(
            In(mut barriers): In<Barriers>,
            mut windows: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
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
                false,
            )
        }
        Box::new(IntoSystem::into_system(barrier))
    }
}
