#![feature(maybe_uninit_write_slice)]

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::DerefMut;
use std::sync::Arc;

use bevy::asset::AssetServer;
use bevy::ecs::prelude::*;

use bevy::utils::HashMap;
use bevy::window::WindowResolution;
use bevy::{
    app::{App, Plugin, PostUpdate, Startup},
    asset::Assets,
    ecs::query::{QueryFilter, QuerySingleError},
    math::Vec2,
    window::PrimaryWindow,
};
use bevy_egui::egui::TextureId;
pub use bevy_egui::*;
use rhyolite::ash::khr::{dynamic_rendering, push_descriptor};
use rhyolite::commands::copy_buffer_to_image;
use rhyolite::future::{gpu_future, GPUFutureBlock};
use rhyolite::ImageWithView;
use rhyolite::{
    ash::vk,
    buffer::{BufferLike, BufferVec, StagingBelt},
    commands::record_commands,
    ecs::IntoRenderSystem,
    future::{GPUBorrowedResource, GPUFutureBlockExt, GPUOwnedResource},
    pipeline::{
        CachedPipeline, DescriptorSetLayout, GraphicsPipeline, GraphicsPipelineBuildInfo,
        PipelineCache, PipelineLayout,
    },
    shader::ShaderModule,
    shader::SpecializedShader,
    swapchain::{SwapchainImage, SwapchainSystemSet},
    Allocator, DeferredOperationTaskPool, Device, HasDevice, Image, ImageExt, ImageLike,
    ImageViewLike, RhyoliteApp, Sampler,
};

pub struct EguiPlugin<Filter: QueryFilter = With<PrimaryWindow>> {
    _filter: std::marker::PhantomData<Filter>,
}
impl<Filter: QueryFilter> Default for EguiPlugin<Filter> {
    fn default() -> Self {
        Self {
            _filter: Default::default(),
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum RenderUiSystem {
    ExtractNode,
}

// Common behavior shared between all instances of EguiPlugin
struct EguiBasePlugin;
impl Plugin for EguiBasePlugin {
    fn build(&self, app: &mut App) {
        use bevy::asset::embedded_asset;
        embedded_asset!(app, "../imported_assets/Default/egui.vert");
        embedded_asset!(app, "../imported_assets/Default/egui.frag");
    }
}

impl<Filter: QueryFilter + Send + Sync + 'static> Plugin for EguiPlugin<Filter> {
    fn build(&self, app: &mut App) {
        app.add_plugins((EguiBasePlugin, bevy_egui::EguiPlugin));
        app.add_systems(
            PostUpdate,
            (
                collect_outputs::<Filter>.after(EguiSet::ProcessOutput),
                prepare_image::<Filter>.after(collect_outputs::<Filter>),
                transfer_image::<Filter>
                    .into_render_system::<rhyolite::selectors::DedicatedTransfer>()
                    .after(prepare_image::<Filter>),
                draw::<Filter>
                    .into_render_system::<rhyolite::selectors::Graphics>()
                    .after(collect_outputs::<Filter>)
                    .after(transfer_image::<Filter>)
                    .in_set(SwapchainSystemSet),
            ),
        );
        app.add_systems(Startup, initialize_pipelines);
        app.add_device_extension::<dynamic_rendering::Meta>()
            .unwrap();
        app.enable_feature::<vk::PhysicalDeviceDynamicRenderingFeatures>(|x| {
            &mut x.dynamic_rendering
        })
        .unwrap();
        app.add_device_extension::<push_descriptor::Meta>().unwrap();
    }
    fn finish(&self, app: &mut App) {
        app.init_resource::<EguiDeviceBuffer<Filter>>();
        app.init_resource::<EguiHostBuffer<Filter>>();
        let allocator = app.world().resource::<Allocator>().clone();

        /*
        if allocator
            .physical_device()
            .properties()
            .memory_model
            .storage_buffer_should_use_staging()
        {
            app.add_systems(
                PostUpdate,
                (
                    resize_device_buffers::<Filter>.after(collect_outputs::<Filter>),
                    copy_buffers::<Filter>
                        .after(resize_device_buffers::<Filter>)
                        .with_barriers(copy_buffers_barrier::<Filter>)
                        .before(draw::<Filter>),
                ),
            );
        }
        */
    }
}

#[derive(Resource)]
pub struct EguiPipelines {
    pipeline: CachedPipeline<GraphicsPipeline>,
    layout: Arc<PipelineLayout>,
}
fn initialize_pipelines(
    mut commands: Commands,
    device: Res<Device>,
    pipeline_cache: Res<PipelineCache>,
    assets: Res<AssetServer>,
) {
    let desc0 = DescriptorSetLayout::new(
        device.clone(),
        &playout_macro::layout!("../assets/draw.playout", 0),
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
    let pipeline_create_info = GraphicsPipelineBuildInfo {
        device: device.clone(),
        layout: layout.clone(),
        stages: vec![
            SpecializedShader {
                stage: vk::ShaderStageFlags::VERTEX,
                shader: assets
                    .load("embedded://rhyolite_egui/../imported_assets/Default/egui.vert"),
                ..Default::default()
            },
            SpecializedShader {
                stage: vk::ShaderStageFlags::FRAGMENT,
                shader: assets
                    .load("embedded://rhyolite_egui/../imported_assets/Default/egui.frag"),
                ..Default::default()
            },
        ],
        builder: move |builder: rhyolite::pipeline::Builder| {
            builder.build(
                vk::GraphicsPipelineCreateInfo::default()
                    .input_assembly_state(&vk::PipelineInputAssemblyStateCreateInfo {
                        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                        primitive_restart_enable: vk::FALSE,
                        ..Default::default()
                    })
                    .vertex_input_state(
                        &vk::PipelineVertexInputStateCreateInfo::default()
                            .vertex_binding_descriptions(&[vk::VertexInputBindingDescription {
                                binding: 0,
                                stride: std::mem::size_of::<egui::epaint::Vertex>() as u32,
                                input_rate: vk::VertexInputRate::VERTEX,
                            }])
                            .vertex_attribute_descriptions(&[
                                vk::VertexInputAttributeDescription {
                                    binding: 0,
                                    location: 0,
                                    format: vk::Format::R32G32_SFLOAT,
                                    offset: 0,
                                }, // pos
                                vk::VertexInputAttributeDescription {
                                    binding: 0,
                                    location: 1,
                                    format: vk::Format::R32G32_SFLOAT,
                                    offset: std::mem::size_of::<[f32; 2]>() as u32,
                                }, // uv
                                vk::VertexInputAttributeDescription {
                                    binding: 0,
                                    location: 2,
                                    format: vk::Format::R8G8B8A8_UNORM,
                                    offset: std::mem::size_of::<[f32; 4]>() as u32,
                                }, // color
                            ]),
                    )
                    .viewport_state(
                        &vk::PipelineViewportStateCreateInfo::default()
                            .scissors(&[vk::Rect2D {
                                offset: vk::Offset2D { x: 0, y: 0 },
                                extent: vk::Extent2D {
                                    width: 2560,
                                    height: 1440,
                                },
                            }])
                            .viewports(&[vk::Viewport {
                                x: 0.0,
                                y: 0.0,
                                width: 2560.0,
                                height: 1440.0,
                                min_depth: 0.0,
                                max_depth: 1.0,
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
                    .dynamic_state(
                        &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                            vk::DynamicState::VIEWPORT,
                            vk::DynamicState::SCISSOR,
                        ]),
                    )
                    .push_next(
                        &mut vk::PipelineRenderingCreateInfo::default()
                            .color_attachment_formats(&[vk::Format::B8G8R8A8_SRGB]),
                    ),
            )
        },
    };
    let pipeline = pipeline_cache.create_graphics(pipeline_create_info);
    commands.insert_resource(EguiPipelines { pipeline, layout });
}

#[derive(Resource)]
pub struct EguiHostBuffer<Filter: QueryFilter> {
    index_buffer: BufferVec<MaybeUninit<u32>>,
    vertex_buffer: BufferVec<MaybeUninit<egui::epaint::Vertex>>,
    marker: std::marker::PhantomData<Filter>,
}
impl<Filter: QueryFilter + Send + Sync + 'static> FromWorld for EguiHostBuffer<Filter> {
    fn from_world(world: &mut World) -> Self {
        let allocator = world.get_resource::<Allocator>().unwrap();
        let is_staging = allocator
            .device()
            .physical_device()
            .properties()
            .memory_model
            .storage_buffer_should_use_staging();
        Self {
            index_buffer: BufferVec::new_upload(
                allocator.clone(),
                1,
                if is_staging {
                    vk::BufferUsageFlags::TRANSFER_SRC
                } else {
                    vk::BufferUsageFlags::INDEX_BUFFER
                },
            ),
            vertex_buffer: BufferVec::new_upload(
                allocator.clone(),
                1,
                if is_staging {
                    vk::BufferUsageFlags::TRANSFER_SRC
                } else {
                    vk::BufferUsageFlags::VERTEX_BUFFER
                },
            ),
            marker: PhantomData,
        }
    }
}

#[derive(Resource)]
pub struct EguiDeviceBuffer<Filter: QueryFilter> {
    total_indices_count: usize,
    total_vertices_count: usize,
    index_buffer: GPUBorrowedResource<BufferVec<MaybeUninit<u32>>>,
    vertex_buffer: GPUBorrowedResource<BufferVec<MaybeUninit<egui::epaint::Vertex>>>,
    textures: BTreeMap<
        u64,
        (
            GPUBorrowedResource<ImageWithView<Image>>,
            egui::TextureOptions,
        ),
    >,
    marker: std::marker::PhantomData<Filter>,
    samplers: HashMap<egui::TextureOptions, Sampler>,
}
impl<Filter: QueryFilter + Send + Sync + 'static> EguiDeviceBuffer<Filter> {
    fn new(allocator: &Allocator) -> Self {
        Self {
            index_buffer: GPUBorrowedResource::new(BufferVec::new_resource(
                allocator.clone(),
                4,
                vk::BufferUsageFlags::INDEX_BUFFER,
            )),
            vertex_buffer: GPUBorrowedResource::new(BufferVec::new_resource(
                allocator.clone(),
                4,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )),
            marker: Default::default(),
            total_indices_count: 0,
            total_vertices_count: 0,
            textures: Default::default(),
            samplers: Default::default(),
        }
    }
}
impl<Filter: QueryFilter + Send + Sync + 'static> FromWorld for EguiDeviceBuffer<Filter> {
    fn from_world(world: &mut World) -> Self {
        let allocator = world.get_resource::<Allocator>().unwrap();
        Self::new(allocator)
    }
}

/// Collect output from egui and copy it into a host-side buffer
/// Create textures
fn collect_outputs<Filter: QueryFilter + Send + Sync + 'static>(
    mut host_buffers: ResMut<EguiHostBuffer<Filter>>,
    mut device_buffers: ResMut<EguiDeviceBuffer<Filter>>,
    egui_render_output: Query<&EguiRenderOutput, Filter>,
) {
    let Ok(output) = egui_render_output.get_single() else {
        return;
    };

    let device_buffers = &mut *device_buffers;
    let mut total_indices_count: usize = 0;
    let mut total_vertices_count: usize = 0;
    for egui::epaint::ClippedPrimitive {
        clip_rect: _,
        primitive,
    } in output.paint_jobs.iter()
    {
        let mesh = match primitive {
            egui::epaint::Primitive::Mesh(mesh) => mesh,
            egui::epaint::Primitive::Callback(_) => {
                unimplemented!("Paint callbacks aren't supported")
            }
        };
        total_indices_count += mesh.indices.len();
        total_vertices_count += mesh.vertices.len();
    }
    host_buffers
        .vertex_buffer
        .resize(total_vertices_count, MaybeUninit::uninit());
    host_buffers
        .index_buffer
        .resize(total_indices_count, MaybeUninit::uninit());

    device_buffers.total_indices_count = total_indices_count;
    device_buffers.total_vertices_count = total_vertices_count;

    // Copy data into the buffer
    total_indices_count = 0;
    total_vertices_count = 0;
    for egui::epaint::ClippedPrimitive { primitive, .. } in output.paint_jobs.iter() {
        let mesh = match primitive {
            egui::epaint::Primitive::Mesh(mesh) => mesh,
            egui::epaint::Primitive::Callback(_) => panic!(),
        };
        MaybeUninit::copy_from_slice(
            &mut host_buffers.vertex_buffer.deref_mut()
                [total_vertices_count..(total_vertices_count + mesh.vertices.len())],
            &mesh.vertices,
        );
        total_vertices_count += mesh.vertices.len();
        MaybeUninit::copy_from_slice(
            &mut host_buffers.index_buffer.deref_mut()
                [total_indices_count..(total_indices_count + mesh.indices.len())],
            &mesh.indices,
        );
        total_indices_count += mesh.indices.len();
    }
    assert_eq!(total_indices_count, device_buffers.total_indices_count);
    assert_eq!(total_vertices_count, device_buffers.total_vertices_count);
    host_buffers.vertex_buffer.flush().unwrap();
    host_buffers.index_buffer.flush().unwrap();
}

/// Create the target images.
fn prepare_image<Filter: QueryFilter + Send + Sync + 'static>(
    host_buffers: ResMut<EguiHostBuffer<Filter>>,
    mut device_buffers: ResMut<EguiDeviceBuffer<Filter>>,
    egui_render_output: Query<&EguiRenderOutput, Filter>,
    allocator: Res<Allocator>,
) {
    let Ok(output) = egui_render_output.get_single() else {
        return;
    };
    for (texture_id, image_delta) in output
        .textures_delta
        .set
        .iter()
        .filter(|(_, image_delta)| image_delta.is_whole())
    {
        let texture_id = match texture_id {
            TextureId::Managed(id) => *id,
            TextureId::User(id) => unimplemented!(),
        };
        if let Some((existing_img, texture_options)) = device_buffers.textures.get(&texture_id) {
            if existing_img.extent().x == image_delta.image.size()[0] as u32
                && existing_img.extent().y == image_delta.image.size()[1] as u32
            {
                continue;
            }
        }
        let size = image_delta.image.size();
        let create_info = vk::ImageCreateInfo {
            format: vk::Format::R8G8B8A8_SRGB,
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D {
                width: size[0] as u32,
                height: size[1] as u32,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let image = Image::new_device_image(allocator.clone(), &create_info)
            .unwrap()
            .with_view()
            .unwrap();
        let image = GPUBorrowedResource::new(image);
        device_buffers
            .textures
            .insert(texture_id, (image, image_delta.options));

        device_buffers
            .samplers
            .entry(image_delta.options)
            .or_insert_with(|| {
                fn convert_filter(filter: egui::TextureFilter) -> vk::Filter {
                    match filter {
                        egui::TextureFilter::Nearest => vk::Filter::NEAREST,
                        egui::TextureFilter::Linear => vk::Filter::LINEAR,
                    }
                }
                let warp = match image_delta.options.wrap_mode {
                    egui::TextureWrapMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    egui::TextureWrapMode::Repeat => vk::SamplerAddressMode::REPEAT,
                    egui::TextureWrapMode::MirroredRepeat => {
                        vk::SamplerAddressMode::MIRRORED_REPEAT
                    }
                };
                Sampler::new(
                    allocator.device().clone(),
                    &vk::SamplerCreateInfo {
                        min_filter: convert_filter(image_delta.options.minification),
                        mag_filter: convert_filter(image_delta.options.magnification),
                        address_mode_u: warp,
                        address_mode_v: warp,
                        ..Default::default()
                    },
                )
                .unwrap()
            });
    }
}

fn transfer_image<'w, 's, Filter: QueryFilter + Send + Sync + 'static>(
    mut device_buffers: ResMut<'w, EguiDeviceBuffer<Filter>>,
    mut egui_render_output: Query<'w, 's, &'w mut EguiRenderOutput, Filter>,
    mut staging_belt: ResMut<'w, StagingBelt>,
) -> impl GPUFutureBlock + use<'w, 's, Filter> {
    gpu_future! { move
        let Ok(output) = egui_render_output.get_single_mut() else {
            return;
        };
        if output.textures_delta.set.is_empty() {
            return;
        }
        for (texture_id, image_delta) in output.textures_delta.set.iter() {
            let texture_id = match texture_id {
                TextureId::Managed(id) => *id,
                TextureId::User(id) => unimplemented!(),
            };
            // Retrieve the image. Because this system runs after `prepare_image`, the image should already
            // been prepared.
            let (target_image, _) = device_buffers.textures.get_mut(&texture_id).unwrap();
            let image_offset = image_delta.pos.unwrap_or([0, 0]);
            let buffer_size_needed = image_delta.image.size().iter().product::<usize>()
                * image_delta.image.bytes_per_pixel();
            let mut staging_buffer = staging_belt.allocate_buffer(buffer_size_needed as u64, 1);
            match &image_delta.image {
                egui::epaint::ImageData::Color(image) => {
                    let slice = bytemuck::cast_slice(image.pixels.as_slice());
                    staging_buffer.copy_from_slice(slice);
                }
                egui::epaint::ImageData::Font(font_image) => {
                    let pixels = font_image.srgba_pixels(None);
                    let target_slice = staging_buffer.deref_mut();
                    let target_slice: &mut [u32] = bytemuck::cast_slice_mut(target_slice);
                    pixels.zip(target_slice).for_each(|(src, dst)| {
                        let src: u32 = bytemuck::cast(src);
                        *dst = src;
                    });
                }
            }
            let update_info = vk::BufferImageCopy {
                buffer_offset: staging_buffer.offset,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D {
                    x: image_offset[0] as i32,
                    y: image_offset[1] as i32,
                    z: 0,
                },
                image_extent: vk::Extent3D {
                    width: image_delta.image.size()[0] as u32,
                    height: image_delta.image.size()[1] as u32,
                    depth: 1,
                },
            };
            let staging_buffer = retain!(staging_buffer);
            let mut staging_buffer = GPUOwnedResource::new(staging_buffer);
            copy_buffer_to_image(
                &mut staging_buffer,
                target_image);
        }
    }
    .run_in_parallel()
}

/// Resize the device buffers if necessary. Only runs on Discrete GPUs.
fn resize_device_buffers<Filter: QueryFilter + Send + Sync + 'static>(
    device_buffers: &mut EguiDeviceBuffer<Filter>,
    allocator: Res<Allocator>,
) {
    let device_buffers: &mut EguiDeviceBuffer<Filter> = &mut *device_buffers;
    if device_buffers.vertex_buffer.len() < device_buffers.total_vertices_count {
        device_buffers.vertex_buffer = {
            let mut buf = BufferVec::new_resource(
                allocator.clone(),
                4,
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            );
            buf.resize(device_buffers.total_vertices_count, MaybeUninit::uninit());
            GPUBorrowedResource::new(buf)
        };
    }

    if device_buffers.index_buffer.len() < device_buffers.total_indices_count {
        device_buffers.index_buffer = {
            let mut buf = BufferVec::new_resource(
                allocator.clone(),
                4,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            );
            buf.resize(device_buffers.total_indices_count, MaybeUninit::uninit());
            GPUBorrowedResource::new(buf)
        };
    }
}

/*
/// Copy data from the host buffers to the device buffers. Only runs on Discrete GPUs.
fn copy_buffers<Filter: QueryFilter + Send + Sync + 'static>(
    mut device_buffers: ResMut<EguiDeviceBuffer<Filter>>,
    mut commands: RenderCommands<'t'>,
    mut host_buffers: ResMut<PerFrame<EguiHostBuffer<Filter>>>,
) {
    let host_buffers = host_buffers.on_frame(&commands);
    let device_buffers: &mut EguiDeviceBuffer<Filter> = &mut *device_buffers;
    if device_buffers.total_vertices_count > 0 {
        commands.copy_buffer(
            host_buffers.vertex_buffer.raw_buffer(),
            device_buffers.vertex_buffer.raw_buffer(),
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: device_buffers.total_vertices_count as u64
                    * std::mem::size_of::<egui::epaint::Vertex>() as u64,
            }],
        );
    }
    if device_buffers.total_indices_count > 0 {
        commands.copy_buffer(
            host_buffers.index_buffer.raw_buffer(),
            device_buffers.index_buffer.raw_buffer(),
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: device_buffers.total_indices_count as u64 * std::mem::size_of::<u32>() as u64,
            }],
        );
    }
}
*/
/// Issue draw commands for egui.
pub fn draw<'w, 's, Filter: QueryFilter + Send + Sync + 'static>(
    mut host_buffers: ResMut<'w, EguiHostBuffer<Filter>>,
    device_buffer: ResMut<'w, EguiDeviceBuffer<Filter>>,
    mut egui_render_output: Query<
        'w,
        's,
        (
            &'w EguiRenderOutput,
            &'w EguiSettings,
            &'w mut SwapchainImage,
            &'w mut RenderTargetSize,
        ),
        Filter,
    >,
    mut egui_pipeline: ResMut<'w, EguiPipelines>,
    pipeline_cache: Res<'w, PipelineCache>,

    assets: Res<'w, Assets<ShaderModule>>,
    task_pool: Res<'w, DeferredOperationTaskPool>,
) -> impl GPUFutureBlock + use<'w, 's, Filter> {
    let egui_pipeline = egui_pipeline.into_inner();
    gpu_future! { move
        let Some(pipeline) = pipeline_cache.retrieve(&mut egui_pipeline.pipeline, &assets, &task_pool)
        else {
            return;
        };
        let (output, egui_settings, swapchain_image, render_target_size) = match egui_render_output.get_single_mut() {
            Ok(r) => r,
            Err(QuerySingleError::NoEntities(_)) => return,
            Err(QuerySingleError::MultipleEntities(_)) => panic!(),
        };
        record_commands(
            (
                swapchain_image.into_inner(),
                device_buffer,
            ),
            |mut ctx, (swapchain_image, device_buffer)| unsafe {
                let mut pass = ctx.begin_rendering(&vk::RenderingInfo {
                    flags: vk::RenderingFlags::empty(),
                    render_area: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: swapchain_image.extent().x,
                            height: swapchain_image.extent().y,
                        },
                    },
                    layer_count: 1,
                    ..Default::default()
                }.color_attachments(&[vk::RenderingAttachmentInfo {
                    image_view: swapchain_image.raw_image_view(),
                    image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::STORE,
                    clear_value: vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 0.0],
                        },
                    },
                    ..Default::default()
                }]));
                pass.bind_pipeline(pipeline);

                let mut vertex_buffer = device_buffer.vertex_buffer.raw_buffer();
                let mut index_buffer = device_buffer.index_buffer.raw_buffer();
                if vertex_buffer == vk::Buffer::null() || index_buffer == vk::Buffer::null() {
                    vertex_buffer = host_buffers.vertex_buffer.raw_buffer();
                    index_buffer = host_buffers.index_buffer.raw_buffer();
                }
                pass.bind_vertex_buffers(0, &[vertex_buffer], &[0]);
                pass.bind_index_buffer(index_buffer, 0, vk::IndexType::UINT32);


                let viewport_physical_size = Vec2::new(
                    swapchain_image.extent().x as f32,
                    swapchain_image.extent().y as f32,
                );
                pass.set_viewport(
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

                let scale_factor = egui_settings.scale_factor * render_target_size.scale_factor;
                let viewport_logical_size = viewport_physical_size / scale_factor;
                pass.push_constants(
                    egui_pipeline.layout.raw(),
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    &bytemuck::cast_slice(&[viewport_logical_size.x, viewport_logical_size.y]),
                );

                let mut current_vertex = 0;
                let mut current_indice = 0;
                for egui::epaint::ClippedPrimitive {
                    clip_rect,
                    primitive,
                } in output.paint_jobs.iter()
                {
                    let mesh = match primitive {
                        egui::epaint::Primitive::Mesh(mesh) => mesh,
                        egui::epaint::Primitive::Callback(_) => panic!(),
                    };
                    let clip_min = Vec2::new(clip_rect.min.x, clip_rect.min.y) * scale_factor;
                    let clip_max = Vec2::new(clip_rect.max.x, clip_rect.max.y) * scale_factor;
                    let clip_extent = clip_max - clip_min;
                    pass.set_scissor(
                        0,
                        &[vk::Rect2D {
                            extent: vk::Extent2D {
                                width: clip_extent.x.round() as u32,
                                height: clip_extent.y.round() as u32,
                            },
                            offset: vk::Offset2D {
                                x: clip_min.x.round() as i32,
                                y: clip_min.y.round() as i32,
                            },
                        }],
                    );
                    let texture_id = match mesh.texture_id {
                        TextureId::Managed(id) => id,
                        TextureId::User(id) => unimplemented!(),
                    };
                    let (texture, options) = device_buffer.textures.get(&texture_id).unwrap();
                    let sampler = device_buffer.samplers.get(options).unwrap();

                    pass.push_descriptor_set(
                        egui_pipeline.layout.as_ref().raw(),
                        0,
                        &[vk::WriteDescriptorSet {
                            dst_binding: 0,
                            descriptor_count: 1,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            p_image_info: &vk::DescriptorImageInfo {
                                image_view: texture.raw_image_view(),
                                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                sampler: sampler.raw(),
                            },
                            ..Default::default()
                        }],
                    );

                    pass.draw_indexed(
                        mesh.indices.len() as u32,
                        1,
                        current_indice,
                        current_vertex as i32,
                        0,
                    );
                    current_vertex += mesh.vertices.len() as u32;
                    current_indice += mesh.indices.len() as u32;
                }

                assert_eq!(current_vertex as usize, device_buffer.total_vertices_count);
                assert_eq!(current_indice as usize, device_buffer.total_indices_count);
            },
            |mut ctx, (ref mut swapchain_image, device_buffer)| {
                ctx.use_image_resource(
                    swapchain_image,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    false,
                );
                for (ref mut texture, _) in device_buffer.textures.values_mut() {
                    ctx.use_image_resource(
                        texture,
                        vk::PipelineStageFlags2::FRAGMENT_SHADER,
                        vk::AccessFlags2::SHADER_READ,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        false,
                    );
                }
            }
        ).await;
    }
}
