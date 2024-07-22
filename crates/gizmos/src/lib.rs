use std::mem::{offset_of, MaybeUninit};
use std::ops::DerefMut;
use std::sync::Arc;

use bevy::ecs::query::QuerySingleError;
use bevy::gizmos::LineGizmo;
use bevy::window::PrimaryWindow;
use bevy::{ecs::system::SystemParamItem, prelude::*};
use rhyolite::commands::{CommonCommands, GraphicsCommands, RenderPassCommands, ResourceTransitionCommands};
use rhyolite::dispose::RenderObject;
use rhyolite::ecs::{Barriers, IntoRenderSystemConfigs, RenderCommands};
use rhyolite::immediate_buffer_transfer::{ImmediateBufferTransferSet, ImmediateBuffers};
use rhyolite::pipeline::{
    CachedPipeline, GraphicsPipeline, GraphicsPipelineBuildInfo, PipelineCache, PipelineLayout,
};
use rhyolite::shader::{ShaderModule, SpecializedShader};
use rhyolite::{
    ImageLike, ImageViewLike,
    ash::vk,
    buffer::immediate_buffer_transfer::{
        ImmediateBufferTransferManager, ImmediateBufferTransferPlugin,
    },
};
use rhyolite::{acquire_swapchain_image, present, Access, BufferLike, DeferredOperationTaskPool, Device, RhyoliteApp, SwapchainImage};
pub struct GizmosPlugin;

impl Plugin for GizmosPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ImmediateBufferTransferPlugin::<GizmosBufferManager>::new(
            vk::BufferUsageFlags::VERTEX_BUFFER,
            4,
        ));

        app.add_systems(Startup, initialize_pipelines);
        app.add_systems(PostUpdate, draw
            .with_barriers(draw_barriers)
            .after(acquire_swapchain_image::<With<PrimaryWindow>>)
            .before(present)
            .after(ImmediateBufferTransferSet::<GizmosBufferManager>::default())
        );

        app.add_device_extension::<rhyolite::ash::khr::dynamic_rendering::Meta>().unwrap();
        app.add_device_extension::<rhyolite::ash::ext::extended_dynamic_state::Meta>().unwrap();

        app.enable_feature::<vk::PhysicalDeviceDynamicRenderingFeatures>(|x| &mut x.dynamic_rendering).unwrap();
        app.enable_feature::<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>(|x| &mut x.extended_dynamic_state).unwrap();

        app.enable_feature::<vk::PhysicalDeviceFeatures>(|x| &mut x.wide_lines).unwrap();
    }
}

#[repr(C)]
struct GizmoBufferItem {
    position: Vec3,
    color: u32,
}

// region: Buffer copy
#[derive(Resource, Default)]
struct GizmosBufferManager;
impl ImmediateBufferTransferManager for GizmosBufferManager {
    type Params = Res<'static, Assets<LineGizmo>>;
    type Data = GizmoBufferItem;

    fn data_size(&self, assets: &mut SystemParamItem<Self::Params>) -> usize {
        assets.iter().map(|(_, line)| line.positions.len()).sum()
    }

    fn collect_outputs(
        &mut self,
        assets: &mut SystemParamItem<Self::Params>,
        dst: &mut [MaybeUninit<Self::Data>],
    ) {
        let mut i = 0;
        for (_, line) in assets.iter() {
            assert_eq!(line.positions.len(), line.colors.len());
            for (position, color) in line.positions.iter().zip(line.colors.iter()) {
                dst[i].write(GizmoBufferItem {
                    position: *position,
                    color: color.as_u32(),
                });
                i += 1;
            }
        }
    }
}
// endregion

// region: Pipelines
#[derive(Resource)]
pub struct GizmoPipelines {
    pipeline: CachedPipeline<RenderObject<GraphicsPipeline>>,
    layout: Arc<PipelineLayout>,
}
fn initialize_pipelines(
    mut commands: Commands,
    device: Res<Device>,
    pipeline_cache: Res<PipelineCache>,
    assets: Res<AssetServer>,
) {
    let layout = PipelineLayout::new(
        device.clone(),
        vec![],
        &[vk::PushConstantRange {
            offset: 0,
            size: std::mem::size_of::<[f32; 2]>() as u32,
            stage_flags: vk::ShaderStageFlags::VERTEX,
        }], // Ideally this can be specified automatically
        vk::PipelineLayoutCreateFlags::empty(),
    )
    .unwrap();
    let layout = Arc::new(layout);
    let pipeline_create_info = GraphicsPipelineBuildInfo {
        layout: layout.clone(),
        device: device.clone(),
        stages: vec![
            SpecializedShader {
                stage: vk::ShaderStageFlags::VERTEX,
                shader: assets.load("shaders/gizmo.vert"),
                ..Default::default()
            },
            SpecializedShader {
                stage: vk::ShaderStageFlags::FRAGMENT,
                shader: assets.load("shaders/gizmo.frag"),
                ..Default::default()
            },
        ],
        builder: move |builder: rhyolite::pipeline::Builder| {
            builder.build(
                vk::GraphicsPipelineCreateInfo::default()
                    .vertex_input_state(
                        &vk::PipelineVertexInputStateCreateInfo::default()
                            .vertex_attribute_descriptions(&[
                                vk::VertexInputAttributeDescription {
                                    binding: 0,
                                    location: 0,
                                    format: vk::Format::R32G32B32_SFLOAT,
                                    offset: offset_of!(GizmoBufferItem, position) as u32,
                                }, // pos
                                vk::VertexInputAttributeDescription {
                                    binding: 0,
                                    location: 1,
                                    format: vk::Format::R8G8B8A8_UNORM,
                                    offset: offset_of!(GizmoBufferItem, color) as u32,
                                }, // color
                            ])
                            .vertex_binding_descriptions(&[vk::VertexInputBindingDescription {
                                binding: 0,
                                stride: std::mem::size_of::<GizmoBufferItem>() as u32,
                                input_rate: vk::VertexInputRate::VERTEX,
                            }]),
                    )
                    .input_assembly_state(&vk::PipelineInputAssemblyStateCreateInfo {
                        topology: vk::PrimitiveTopology::LINE_LIST,
                        primitive_restart_enable: vk::FALSE,
                        ..Default::default()
                    })
                    .viewport_state(
                        &vk::PipelineViewportStateCreateInfo::default()
                            .viewports(&[vk::Viewport {
                                x: 0.0,
                                y: 0.0,
                                width: 2560.0,
                                height: 1440.0,
                                min_depth: 0.0,
                                max_depth: 1.0,
                            }])
                            .scissors(&[vk::Rect2D {
                                offset: vk::Offset2D { x: 0, y: 0 },
                                extent: vk::Extent2D {
                                    width: 2560,
                                    height: 1440,
                                },
                            }]),
                    )
                    .rasterization_state(&vk::PipelineRasterizationStateCreateInfo {
                        polygon_mode: vk::PolygonMode::FILL,
                        cull_mode: vk::CullModeFlags::NONE,
                        line_width: 1.0,
                        ..Default::default()
                    })
                    .multisample_state(&vk::PipelineMultisampleStateCreateInfo {
                        rasterization_samples: vk::SampleCountFlags::TYPE_1,
                        sample_shading_enable: vk::FALSE,
                        ..Default::default()
                    })
                    .color_blend_state(
                        &vk::PipelineColorBlendStateCreateInfo::default().attachments(&[
                            vk::PipelineColorBlendAttachmentState {
                                color_write_mask: vk::ColorComponentFlags::R
                                    | vk::ColorComponentFlags::G
                                    | vk::ColorComponentFlags::B
                                    | vk::ColorComponentFlags::A,
                                src_color_blend_factor: vk::BlendFactor::ONE,
                                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                                color_blend_op: vk::BlendOp::ADD,
                                src_alpha_blend_factor: vk::BlendFactor::ONE,
                                dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                                alpha_blend_op: vk::BlendOp::ADD,
                                blend_enable: vk::TRUE,
                                ..Default::default()
                            },
                        ]),
                    )
                    .dynamic_state(&vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                        vk::DynamicState::PRIMITIVE_TOPOLOGY,
                        vk::DynamicState::LINE_WIDTH,
                        vk::DynamicState::VIEWPORT,
                        vk::DynamicState::SCISSOR,
                    ]))
                    .push_next(
                        &mut vk::PipelineRenderingCreateInfo::default()
                            .color_attachment_formats(&[vk::Format::B8G8R8A8_UNORM]),
                    ),
            )
        },
    };
    let pipeline = pipeline_cache.create_graphics(pipeline_create_info);
    commands.insert_resource(GizmoPipelines { pipeline, layout });
}
// endregion

fn draw_barriers(
    mut barriers: In<Barriers>,
    mut buffers: ResMut<ImmediateBuffers<GizmosBufferManager>>,
    mut output_images: Query<(&mut SwapchainImage), With<PrimaryWindow>>,
) {
    let swapchain_image = match output_images.get_single_mut() {
        Ok(r) => r,
        Err(QuerySingleError::NoEntities(_)) => return (),
        Err(QuerySingleError::MultipleEntities(_)) => panic!(),
    };
    barriers.transition(
        swapchain_image.into_inner().deref_mut(),
        Access {
            stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        },
        false,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );

    barriers.transition(
        &mut *buffers,
        Access {
            stage: vk::PipelineStageFlags2::VERTEX_INPUT,
            access: vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
        },
        true,
        (),
    );
}

fn draw(
    mut render_commands: RenderCommands<'g'>,
    mut pipelines: ResMut<GizmoPipelines>,
    pipeline_cache: Res<PipelineCache>,
    assets: Res<Assets<ShaderModule>>,
    pool: Res<DeferredOperationTaskPool>,
    mut buffers: ResMut<ImmediateBuffers<GizmosBufferManager>>,
    primary_window: Query<&mut SwapchainImage, With<PrimaryWindow>>,
    line_gizmos: Res<Assets<LineGizmo>>,
    config: Res<GizmoConfigStore>,
) {
    let Some(pipeline) = pipeline_cache.retrieve(&mut pipelines.pipeline, &assets, &pool) else {
        return;
    };
    let swapchain_image = primary_window.single();


    let vertex_buffer = buffers.device_buffer(&render_commands);
    let mut render_pass = render_commands.begin_rendering(&vk::RenderingInfo {
        render_area: vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: swapchain_image.extent().x,
                height: swapchain_image.extent().y,
            },
        },
        layer_count: 1,
        ..Default::default()
    }.color_attachments(&[
        vk::RenderingAttachmentInfo {
            image_view: swapchain_image.raw_image_view(),
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.1, 0.0, 0.0],
                },
            },
            ..Default::default()
        }
    ]));

    render_pass.bind_pipeline(pipeline);

    render_pass.bind_vertex_buffers(0, &[vertex_buffer.raw_buffer()], &[0]);
    let viewport_physical_size = Vec2::new(
        swapchain_image.extent().x as f32,
        swapchain_image.extent().y as f32,
    );
    render_pass.set_viewport(
        0,
        &[vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: viewport_physical_size.x,
            height: viewport_physical_size.y,
            min_depth: 0.0,
            max_depth: 1.0,
        }],
    );
    render_pass.set_scissor(0, &[
        vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: viewport_physical_size.x as u32,
                height: viewport_physical_size.y as u32,
            },
        },
    ]);
    let mut i = 0;
    for (_, line) in line_gizmos.iter() {
        let Some((config, _)) = config.get_config_dyn(&line.config_ty) else {
            continue;
        };
        let vertex_count = line.positions.len() as u32;
        render_pass.set_line_width(config.line_width);
        render_pass.set_primitive_topology(if line.strip {
            vk::PrimitiveTopology::LINE_STRIP
        } else {
            vk::PrimitiveTopology::LINE_LIST
        });
        render_pass.draw(vertex_count, 1, i, 0);
        i += vertex_count;
    }
}
