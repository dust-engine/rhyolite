#![feature(maybe_uninit_write_slice)]

use std::collections::BTreeMap;
use std::mem::MaybeUninit;
use std::ops::DerefMut;
use std::os::raw::c_void;
use std::sync::Arc;

use bevy::asset::{AssetServer, Assets};
use bevy::ecs::prelude::*;
use bevy::ecs::query::QuerySingleError;

use bevy::math::Vec2;
use bevy::{
    app::{App, Plugin, PostUpdate, Startup},
    ecs::{query::QueryFilter, schedule::IntoSystemConfigs},
    window::PrimaryWindow,
};
use bevy_egui::egui::TextureId;
pub use bevy_egui::*;
use rhyolite::{
    acquire_swapchain_image, ash,
    ash::vk,
    buffer::staging::StagingBelt,
    commands::{CommonCommands, GraphicsCommands, RenderPassCommands, TransferCommands},
    ecs::{Barriers, IntoRenderSystemConfigs},
    ecs::{PerFrameMut, PerFrameResource, RenderCommands, RenderImage, RenderRes},
    pipeline::{
        CachedPipeline, DescriptorSetLayout, GraphicsPipeline, GraphicsPipelineBuildInfo,
        PipelineCache, PipelineLayout,
    },
    present,
    shader::{ShaderModule, SpecializedShader},
    Access, Allocator, BufferArray, BufferLike, DeferredOperationTaskPool, Device, HasDevice,
    Image, ImageLike, ImageViewLike, RhyoliteApp, SwapchainImage,
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

impl<Filter: QueryFilter + Send + Sync + 'static> Plugin for EguiPlugin<Filter> {
    fn build(&self, app: &mut App) {
        // TODO: Rip out the copying, don't add the copy buffer systems if we don't need to copy buffers.
        app.add_plugins(bevy_egui::EguiPlugin);
        app.add_systems(
            PostUpdate,
            (
                collect_outputs::<Filter>.after(EguiSet::ProcessOutput),
                prepare_image::<Filter>.after(collect_outputs::<Filter>),
                transfer_image::<Filter>
                    .after(prepare_image::<Filter>)
                    .with_barriers(image_barrier::<Filter>),
                draw::<Filter>
                    .with_barriers(draw_barriers::<Filter>)
                    .after(collect_outputs::<Filter>)
                    .after(acquire_swapchain_image::<Filter>)
                    .after(transfer_image::<Filter>)
                    .before(present),
            ),
        );
        app.add_systems(Startup, initialize_pipelines);
        app.enable_feature::<vk::PhysicalDeviceVulkan13Features>(|x| &mut x.dynamic_rendering)
            .unwrap();
        app.enable_feature::<vk::PhysicalDeviceVulkan13Features>(|x| &mut x.inline_uniform_block)
            .unwrap();
        app.add_device_extension::<ash::extensions::khr::PushDescriptor>()
            .unwrap();
    }
    fn finish(&self, app: &mut App) {
        app.init_resource::<EguiDeviceBuffer<Filter>>();
        let device = app.world.resource::<Device>();
        if device
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
        &playout_macro::layout!("../../../assets/shaders/draw.playout", 0),
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
    let layout2 = layout.clone();
    let pipeline_create_info = GraphicsPipelineBuildInfo {
        device: device.clone(),
        stages: vec![
            SpecializedShader {
                stage: vk::ShaderStageFlags::VERTEX,
                shader: assets.load("shaders/egui.vert"),
                ..Default::default()
            },
            SpecializedShader {
                stage: vk::ShaderStageFlags::FRAGMENT,
                shader: assets.load("shaders/egui.frag"),
                ..Default::default()
            },
        ],
        builder: move |mut builder| {
            let input_bindings = vk::VertexInputBindingDescription {
                binding: 0,
                stride: std::mem::size_of::<egui::epaint::Vertex>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            };
            let input_attributes = [
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
                    format: vk::Format::B8G8R8A8_UNORM,
                    offset: std::mem::size_of::<[f32; 4]>() as u32,
                }, // color
            ];
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
                vertex_binding_description_count: 1,
                p_vertex_binding_descriptions: &input_bindings,
                vertex_attribute_description_count: input_attributes.len() as u32,
                p_vertex_attribute_descriptions: input_attributes.as_ptr(),
                ..Default::default()
            };
            let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                primitive_restart_enable: vk::FALSE,
                ..Default::default()
            };
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: 2560.0,
                height: 1440.0,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scisser = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: 2560,
                    height: 1440,
                },
            };
            let viewport_state = vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                p_viewports: &viewport,
                scissor_count: 1,
                p_scissors: &scisser,
                ..Default::default()
            };
            let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
                polygon_mode: vk::PolygonMode::FILL,
                cull_mode: vk::CullModeFlags::NONE,
                line_width: 1.0,
                ..Default::default()
            };
            let multisample_state = vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                sample_shading_enable: vk::FALSE,
                ..Default::default()
            };
            let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState {
                color_write_mask: vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
                blend_enable: vk::FALSE,
                ..Default::default()
            };
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
                attachment_count: 1,
                p_attachments: &color_blend_attachment_state,
                ..Default::default()
            };
            let dynamic_rendering_format = vk::Format::B8G8R8A8_UNORM;
            let dynamic_rendering_info = vk::PipelineRenderingCreateInfo {
                color_attachment_count: 1,
                p_color_attachment_formats: &dynamic_rendering_format,
                ..Default::default()
            };
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state = vk::PipelineDynamicStateCreateInfo {
                dynamic_state_count: dynamic_states.len() as u32,
                p_dynamic_states: dynamic_states.as_ptr(),
                ..Default::default()
            };
            builder.p_vertex_input_state = &vertex_input_state;
            builder.p_input_assembly_state = &input_assembly_state;
            builder.p_viewport_state = &viewport_state;
            builder.p_rasterization_state = &rasterization_state;
            builder.p_multisample_state = &multisample_state;
            builder.p_color_blend_state = &color_blend_state;
            builder.p_dynamic_state = &dynamic_state;
            builder.p_next = &dynamic_rendering_info as *const _ as *const c_void;
            builder.layout = layout2.raw();
            builder.build()
        },
    };
    let pipeline = pipeline_cache.create_graphics(pipeline_create_info);
    commands.insert_resource(EguiPipelines { pipeline, layout });
}

pub struct EguiHostBuffer<Filter: QueryFilter> {
    index_buffer: BufferArray<u32>,
    vertex_buffer: BufferArray<egui::epaint::Vertex>,
    marker: std::marker::PhantomData<Filter>,
}
impl<Filter: QueryFilter + Send + Sync + 'static> PerFrameResource for EguiHostBuffer<Filter> {
    type Params = Res<'static, Allocator>;
    fn create(allocator: Res<Allocator>) -> Self {
        Self {
            index_buffer: BufferArray::new_upload(
                allocator.clone(),
                vk::BufferUsageFlags::INDEX_BUFFER,
            ),
            vertex_buffer: BufferArray::new_upload(
                allocator.clone(),
                vk::BufferUsageFlags::VERTEX_BUFFER,
            ),
            marker: Default::default(),
        }
    }
}
#[derive(Resource)]
pub struct EguiDeviceBuffer<Filter: QueryFilter> {
    total_indices_count: usize,
    total_vertices_count: usize,
    index_buffer: RenderRes<BufferArray<u32>>,
    vertex_buffer: RenderRes<BufferArray<egui::epaint::Vertex>>,
    textures: BTreeMap<u64, RenderImage<Image>>,
    marker: std::marker::PhantomData<Filter>,
}
impl<Filter: QueryFilter + Send + Sync + 'static> EguiDeviceBuffer<Filter> {
    fn new(allocator: &Allocator) -> Self {
        Self {
            index_buffer: RenderRes::new(BufferArray::new_resource(
                allocator.clone(),
                vk::BufferUsageFlags::INDEX_BUFFER,
            )),
            vertex_buffer: RenderRes::new(BufferArray::new_resource(
                allocator.clone(),
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )),
            marker: Default::default(),
            total_indices_count: 0,
            total_vertices_count: 0,
            textures: Default::default(),
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
    mut host_buffers: PerFrameMut<EguiHostBuffer<Filter>>,
    mut device_buffers: ResMut<EguiDeviceBuffer<Filter>>,
    mut egui_render_output: Query<(Entity, &mut EguiRenderOutput), Filter>,
) {
    let Ok((window, output)) = egui_render_output.get_single_mut() else {
        return;
    };

    let device_buffers = &mut *device_buffers;
    let mut total_indices_count: usize = 0;
    let mut total_vertices_count: usize = 0;
    for egui::epaint::ClippedPrimitive {
        clip_rect,
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
    let host_buffers = &mut *host_buffers;
    host_buffers
        .vertex_buffer
        .realloc(total_vertices_count)
        .unwrap();
    host_buffers
        .index_buffer
        .realloc(total_indices_count)
        .unwrap();

    device_buffers.total_indices_count = total_indices_count;
    device_buffers.total_vertices_count = total_vertices_count;

    // Copy data into the buffer
    total_indices_count = 0;
    total_vertices_count = 0;
    for egui::epaint::ClippedPrimitive {
        clip_rect,
        primitive,
    } in output.paint_jobs.iter()
    {
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
}

fn prepare_image<Filter: QueryFilter + Send + Sync + 'static>(
    mut commands: RenderCommands<'g'>,
    mut device_buffers: ResMut<EguiDeviceBuffer<Filter>>,
    mut egui_render_output: Query<&mut EguiRenderOutput, Filter>,
    allocator: Res<Allocator>,
) {
    let Ok(output) = egui_render_output.get_single_mut() else {
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
        if let Some(existing_img) = device_buffers.textures.get(&texture_id) {
            if existing_img.extent().width == image_delta.image.size()[0] as u32
                && existing_img.extent().height == image_delta.image.size()[1] as u32
            {
                continue;
            }
        }
        let size = image_delta.image.size();
        let create_info = vk::ImageCreateInfo {
            format: vk::Format::B8G8R8A8_UNORM,
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
        let image = Image::new_device_image(allocator.clone(), &create_info).unwrap();
        let image = RenderImage::new(image);
        if let Some(old) = device_buffers.textures.insert(texture_id, image) {
            commands.retain(old.into_dispose());
        }
    }
}
fn image_barrier<Filter: QueryFilter + Send + Sync + 'static>(
    mut barriers: In<Barriers>,
    mut device_buffers: ResMut<EguiDeviceBuffer<Filter>>,
    mut egui_render_output: Query<&mut EguiRenderOutput, Filter>,
) {
    let Ok(output) = egui_render_output.get_single_mut() else {
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
        barriers.transition_image(
            device_buffers.textures.get_mut(&texture_id).unwrap(),
            Access {
                access: vk::AccessFlags2::TRANSFER_WRITE,
                stage: vk::PipelineStageFlags2::COPY,
            },
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            !image_delta.is_whole(),
        );
    }
}

fn transfer_image<Filter: QueryFilter + Send + Sync + 'static>(
    mut commands: RenderCommands<'g'>,
    mut device_buffers: ResMut<EguiDeviceBuffer<Filter>>,
    mut egui_render_output: Query<&mut EguiRenderOutput, Filter>,
    mut staging_belt: ResMut<StagingBelt>,
    allocator: Res<Allocator>,
) {
    let Ok(mut output) = egui_render_output.get_single_mut() else {
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
        if let Some(existing_img) = device_buffers.textures.get(&texture_id) {
            if existing_img.extent().width == image_delta.image.size()[0] as u32
                && existing_img.extent().height == image_delta.image.size()[1] as u32
            {
                continue;
            }
        }
        let size = image_delta.image.size();
        let create_info = vk::ImageCreateInfo {
            format: vk::Format::B8G8R8A8_UNORM,
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
        let image = Image::new_device_image(allocator.clone(), &create_info).unwrap();
        let image = RenderImage::new(image);
        if let Some(old) = device_buffers.textures.insert(texture_id, image) {
            commands.retain(old.into_dispose());
        }
    }
    if output.textures_delta.set.is_empty() {
        return;
    }
    let mut staging_allocator = staging_belt.start();
    for (texture_id, image_delta) in output.textures_delta.set.iter() {
        let texture_id = match texture_id {
            TextureId::Managed(id) => *id,
            TextureId::User(id) => unimplemented!(),
        };
        let target_image = device_buffers.textures.get(&texture_id).unwrap();
        let image_offset = image_delta.pos.unwrap_or([0, 0]);
        let buffer_size_needed = image_delta.image.size().iter().product::<usize>()
            * image_delta.image.bytes_per_pixel();
        let mut staging_buffer = staging_allocator.allocate_buffer(buffer_size_needed as u64, 1);
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
        commands.copy_buffer_to_image(
            staging_buffer.buffer,
            target_image.raw(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[update_info],
        );
        println!("Copied buffer to texture");
    }
    commands.retain(staging_allocator.finish());
    output.textures_delta.set.clear();
}

/// Resize the device buffers if necessary. Only runs on Discrete GPUs.
fn resize_device_buffers<Filter: QueryFilter + Send + Sync + 'static>(
    mut device_buffers: ResMut<EguiDeviceBuffer<Filter>>,
    mut commands: RenderCommands<'g'>,
    allocator: Res<Allocator>,
) {
    let device_buffers: &mut EguiDeviceBuffer<Filter> = &mut *device_buffers;
    if device_buffers.vertex_buffer.len() < device_buffers.total_vertices_count {
        device_buffers.vertex_buffer.replace(|old| {
            commands.retain(old);
            let mut buf = BufferArray::new_resource(
                allocator.clone(),
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            );
            buf.realloc(device_buffers.total_vertices_count).unwrap();
            RenderRes::new(buf)
        });
    }

    if device_buffers.index_buffer.len() < device_buffers.total_indices_count {
        device_buffers.index_buffer.replace(|old| {
            commands.retain(old);
            let mut buf = BufferArray::new_resource(
                allocator.clone(),
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            );
            buf.realloc(device_buffers.total_indices_count).unwrap();
            RenderRes::new(buf)
        });
    }
}

fn copy_buffers_barrier<Filter: QueryFilter + Send + Sync + 'static>(
    mut barriers: In<Barriers>,
    mut device_buffers: ResMut<EguiDeviceBuffer<Filter>>,
) {
    if device_buffers.total_vertices_count > 0 {
        barriers.transition_buffer(
            &mut device_buffers.vertex_buffer,
            Access {
                access: vk::AccessFlags2::TRANSFER_WRITE,
                stage: vk::PipelineStageFlags2::COPY,
            },
        );
    }

    if device_buffers.total_indices_count > 0 {
        barriers.transition_buffer(
            &mut device_buffers.index_buffer,
            Access {
                access: vk::AccessFlags2::TRANSFER_WRITE,
                stage: vk::PipelineStageFlags2::COPY,
            },
        );
    }
}
/// Copy data from the host buffers to the device buffers. Only runs on Discrete GPUs.
fn copy_buffers<Filter: QueryFilter + Send + Sync + 'static>(
    mut device_buffers: ResMut<EguiDeviceBuffer<Filter>>,
    mut commands: RenderCommands<'g'>,
    host_buffers: PerFrameMut<EguiHostBuffer<Filter>>,
) {
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

fn draw_barriers<Filter: QueryFilter + Send + Sync + 'static>(
    mut barriers: In<Barriers>,
    mut device_buffer: ResMut<EguiDeviceBuffer<Filter>>,
    mut egui_render_output: Query<(&EguiRenderOutput, &mut SwapchainImage), Filter>,
) {
    let (output, mut swapchain_image) = match egui_render_output.get_single_mut() {
        Ok(r) => r,
        Err(QuerySingleError::NoEntities(_)) => return,
        Err(QuerySingleError::MultipleEntities(_)) => panic!(),
    };

    if output.paint_jobs.is_empty() {
        return;
    }
    for i in output.paint_jobs.iter() {
        let mesh = match &i.primitive {
            egui::epaint::Primitive::Mesh(mesh) => mesh,
            egui::epaint::Primitive::Callback(_) => panic!(),
        };
        let texture_id = match mesh.texture_id {
            TextureId::Managed(id) => id,
            TextureId::User(id) => unimplemented!(),
        };
        let texture = device_buffer.textures.get_mut(&texture_id).unwrap();
        barriers.transition_image(
            texture,
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            true,
        );
    }
    barriers.transition_image(
        &mut *swapchain_image,
        Access {
            stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        },
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        false,
    );

    if device_buffer.vertex_buffer.len() > 0 {
        barriers.transition_buffer(
            &mut device_buffer.vertex_buffer,
            Access {
                stage: vk::PipelineStageFlags2::VERTEX_INPUT,
                access: vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
            },
        );
    }
    if device_buffer.index_buffer.len() > 0 {
        barriers.transition_buffer(
            &mut device_buffer.index_buffer,
            Access {
                stage: vk::PipelineStageFlags2::INDEX_INPUT,
                access: vk::AccessFlags2::INDEX_READ,
            },
        );
    }
}
/// Issue draw commands for egui.
pub fn draw<Filter: QueryFilter + Send + Sync + 'static>(
    mut commands: RenderCommands<'g'>,
    host_buffers: PerFrameMut<EguiHostBuffer<Filter>>,
    device_buffer: ResMut<EguiDeviceBuffer<Filter>>,
    mut egui_render_output: Query<(&EguiRenderOutput, &mut SwapchainImage, &WindowSize), Filter>,
    mut egui_pipeline: ResMut<EguiPipelines>,
    egui_settings: Res<EguiSettings>,
    pipeline_cache: Res<PipelineCache>,

    assets: Res<Assets<ShaderModule>>,
    task_pool: Res<DeferredOperationTaskPool>,
) {
    let (output, mut swapchain_image, window_size) = match egui_render_output.get_single_mut() {
        Ok(r) => r,
        Err(QuerySingleError::NoEntities(_)) => return,
        Err(QuerySingleError::MultipleEntities(_)) => panic!(),
    };
    if output.paint_jobs.is_empty() {
        return;
    }
    let Some((pipeline, old_pipeline)) =
        pipeline_cache.retrieve_graphics(&mut egui_pipeline.pipeline, &assets, &task_pool)
    else {
        return;
    };
    if let Some(old_pipeline) = old_pipeline {
        commands.retain(old_pipeline);
    }
    let mut pass = commands.begin_rendering(&vk::RenderingInfo {
        flags: vk::RenderingFlags::empty(),
        render_area: vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: swapchain_image.extent().width,
                height: swapchain_image.extent().height,
            },
        },
        layer_count: 1,
        color_attachment_count: 1,
        p_color_attachments: &vk::RenderingAttachmentInfo {
            image_view: swapchain_image.raw_image_view(),
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            resolve_mode: vk::ResolveModeFlags::NONE,
            resolve_image_view: vk::ImageView::null(),
            resolve_image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [1.0, 0.0, 0.0, 1.0],
                },
            },
            ..Default::default()
        },
        ..Default::default()
    });
    pass.bind_pipeline(pipeline.raw());

    let mut vertex_buffer = device_buffer.vertex_buffer.raw_buffer();
    let mut index_buffer = device_buffer.index_buffer.raw_buffer();
    if vertex_buffer == vk::Buffer::null() || index_buffer == vk::Buffer::null() {
        vertex_buffer = host_buffers.vertex_buffer.raw_buffer();
        index_buffer = host_buffers.index_buffer.raw_buffer();
    }
    let mut current_vertex = 0;
    let mut current_indice = 0;

    pass.bind_vertex_buffers(0, &[vertex_buffer], &[0]);
    pass.bind_index_buffer(index_buffer, 0, vk::IndexType::UINT32);
    let viewport_physical_size = Vec2::new(
        swapchain_image.extent().width as f32,
        swapchain_image.extent().height as f32,
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
    let scale_factor = egui_settings.scale_factor * window_size.scale_factor;
    let viewport_logical_size = viewport_physical_size / scale_factor;
    pass.push_constants(
        egui_pipeline.layout.raw(),
        vk::ShaderStageFlags::VERTEX,
        0,
        &bytemuck::cast_slice(&[viewport_logical_size.x, viewport_logical_size.y]),
    );

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
        pass.draw_indexed(
            mesh.vertices.len() as u32,
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
    drop(pass);

    commands
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
