#![feature(maybe_uninit_write_slice)]

use std::mem::MaybeUninit;
use std::ops::DerefMut;
use std::sync::Arc;

use bevy::ecs::prelude::*;
use bevy::{
    app::{App, Plugin, PostUpdate, Update},
    ecs::{query::QueryFilter, schedule::IntoSystemConfigs},
    log::tracing_subscriber::layer::Filter,
    window::{PrimaryWindow, Window},
};
pub use bevy_egui::*;
use rhyolite::{
    ash,
    ash::vk,
    ecs::{Barriers, IntoRenderSystemConfigs},
    ecs::{PerFrameMut, PerFrameResource, RenderCommands, RenderRes},
    Access, Allocator, BufferArray, BufferLike, Device, HasDevice, Instance,
    PhysicalDeviceMemoryModel, RhyoliteApp,
    SwapchainImage,
    ImageLike,
    ImageViewLike,
    present,
    pipeline::{PipelineLayout,DescriptorSetLayout}
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

impl<Filter: QueryFilter + Send + Sync + 'static> Plugin for EguiPlugin<Filter> {
    fn build(&self, app: &mut App) {
        // TODO: Rip out the copying, don't add the copy buffer systems if we don't need to copy buffers.
        app.add_plugins(bevy::time::TimePlugin); // This should've been declared in bevy_egui instead.
        app.add_plugins(bevy_egui::EguiPlugin);
        app.add_systems(
            PostUpdate,
            (
                collect_outputs::<Filter>.after(EguiSet::ProcessOutput),
                draw::<Filter>.with_barriers(draw_barriers::<Filter>)
                .after(collect_outputs::<Filter>)
                .before(present)
            ),
        );
        app.enable_feature::<vk::PhysicalDeviceVulkan13Features>(|x| &mut x.dynamic_rendering);
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


fn initialize_pipelines(
    device: Res<Device>
) {
    let desc0 = DescriptorSetLayout::new(
        device.clone(),
        &playout_macro::layout!("draw.playout", 0),
        vk::DescriptorSetLayoutCreateFlags::empty(),
    ).unwrap();
    let layout = PipelineLayout::new(
        device.clone(),
        vec![
            Arc::new(desc0)
        ],
        &[],
        vk::PipelineLayoutCreateFlags::empty(),
    );
}

struct EguiHostBuffer<Filter: QueryFilter> {
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
struct EguiDeviceBuffer<Filter: QueryFilter> {
    total_indices_count: usize,
    total_vertices_count: usize,
    index_buffer: RenderRes<BufferArray<u32>>,
    vertex_buffer: RenderRes<BufferArray<egui::epaint::Vertex>>,
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
fn collect_outputs<Filter: QueryFilter + Send + Sync + 'static>(
    mut host_buffers: PerFrameMut<EguiHostBuffer<Filter>>,
    mut device_buffers: ResMut<EguiDeviceBuffer<Filter>>,
    mut egui_render_output: Query<(Entity, &EguiRenderOutput), Filter>,
) {
    let Ok((window, mut output)) = egui_render_output.get_single_mut() else {
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

/// Resize the device buffers if necessary. Only runs on Discrete GPUs.
fn resize_device_buffers<Filter: QueryFilter + Send + Sync + 'static>(
    mut device_buffers: ResMut<EguiDeviceBuffer<Filter>>,
    mut commands: RenderCommands<'t'>,
    host_buffers: PerFrameMut<EguiHostBuffer<Filter>>,
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
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
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
    mut commands: RenderCommands<'t'>,
    host_buffers: PerFrameMut<EguiHostBuffer<Filter>>,
) {
    let device_buffers: &mut EguiDeviceBuffer<Filter> = &mut *device_buffers;
    if device_buffers.total_vertices_count > 0 {
        commands.record_commands().copy_buffer(
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
        commands.record_commands().copy_buffer(
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
    mut egui_render_output: Query<(Entity, &EguiRenderOutput, &mut SwapchainImage), Filter>,
) {
    let (_, output, mut swapchain_image) = egui_render_output.get_single_mut().unwrap();
    if output.paint_jobs.is_empty() {
        return;
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
}
/// Issue draw commands for egui.
fn draw<Filter: QueryFilter + Send + Sync + 'static>(
    mut commands: RenderCommands<'g'>,
    mut host_buffers: PerFrameMut<EguiHostBuffer<Filter>>,
    mut device_buffer: ResMut<EguiDeviceBuffer<Filter>>,
    mut egui_render_output: Query<(Entity, &EguiRenderOutput, &mut SwapchainImage), Filter>,
) {
    let (_, output, swapchain_image) = egui_render_output.get_single_mut().unwrap();
    if output.paint_jobs.is_empty() {
        return;
    }
    let pass = commands.record_commands().begin_rendering(&vk::RenderingInfo {
        flags: vk::RenderingFlags::empty(),
        render_area: vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: swapchain_image.extent().width,
                height: swapchain_image.extent().width,
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
            load_op: vk::AttachmentLoadOp::DONT_CARE,
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
}
