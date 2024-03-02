
#![feature(maybe_uninit_write_slice)]

use std::mem::MaybeUninit;
use std::ops::DerefMut;

use bevy::{app::{App, Plugin, PostUpdate, Update}, ecs::{query::QueryFilter, schedule::IntoSystemConfigs}, log::tracing_subscriber::layer::Filter, window::{PrimaryWindow, Window}};
pub use bevy_egui::*;
use bevy::ecs::prelude::*;
use rhyolite::{BufferArray, ecs::{RenderRes, PerFrameMut, RenderCommands, PerFrameResource}, Allocator, ash::vk, PhysicalDeviceMemoryModel, HasDevice};


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
        app.add_plugins(bevy::time::TimePlugin); // This should've been declared in bevy_egui instead.
        app.add_plugins(bevy_egui::EguiPlugin);
        app.add_systems(PostUpdate, render_egui::<Filter>
            .after(EguiSet::ProcessOutput));

    }
    fn finish(&self, app: &mut App) {
        app.init_resource::<EguiDeviceBuffer<Filter>>();
    }
}


struct EguiHostBuffer<Filter: QueryFilter> {
    index_buffer: BufferArray<u32>,
    vertex_buffer: BufferArray<egui::epaint::Vertex>,
    marker: std::marker::PhantomData<Filter>
}
impl<Filter: QueryFilter + Send + Sync + 'static> PerFrameResource for EguiHostBuffer<Filter> {
    type Params = Res<'static, Allocator>;
    fn create(allocator: Res<Allocator>) -> Self {
        Self {
            index_buffer: BufferArray::new_upload(allocator.clone(), vk::BufferUsageFlags::INDEX_BUFFER),
            vertex_buffer: BufferArray::new_upload(allocator.clone(), vk::BufferUsageFlags::VERTEX_BUFFER),
            marker: Default::default(),
        }
    }
}
#[derive(Resource)]
struct EguiDeviceBuffer<Filter: QueryFilter>{
    index_buffer: RenderRes<BufferArray<u32>>,
    vertex_buffer: RenderRes<BufferArray<egui::epaint::Vertex>>,
    marker: std::marker::PhantomData<Filter>
}
impl<Filter: QueryFilter + Send + Sync + 'static> EguiDeviceBuffer<Filter> {
    fn new(allocator: &Allocator) -> Self {
        Self {
            index_buffer: RenderRes::new(BufferArray::new_resource(allocator.clone(), vk::BufferUsageFlags::INDEX_BUFFER)),
            vertex_buffer: RenderRes::new(BufferArray::new_resource(allocator.clone(), vk::BufferUsageFlags::VERTEX_BUFFER)),
            marker: Default::default(),
        }
    }
}
impl<Filter: QueryFilter + Send + Sync + 'static> FromWorld for EguiDeviceBuffer<Filter> {
    fn from_world(world: &mut World) -> Self {
        let allocator = world.get_resource::<Allocator>().unwrap();
        Self::new(allocator)
    }
}


fn render_egui<Filter: QueryFilter + Send + Sync + 'static>(
    mut commands: RenderCommands<'t'>,
    mut host_buffers: PerFrameMut<EguiHostBuffer<Filter>>,
    mut device_buffer: ResMut<EguiDeviceBuffer<Filter>>,
    mut egui_render_output: Query<(Entity, &EguiRenderOutput), Filter>,
    settings: Res<EguiSettings>,
    allocator: Res<Allocator>,
) {
    let Ok((window, mut output)) = egui_render_output.get_single_mut() else {
        return;
    };
    println!("Rendering egui to window: {:?}", output.paint_jobs.len());

    let mut total_indices_count: usize = 0;
    let mut total_vertices_count: usize = 0;
    for egui::epaint::ClippedPrimitive {
        clip_rect,
        primitive,
    } in output.paint_jobs.iter() {
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
    host_buffers.vertex_buffer.realloc(total_vertices_count).unwrap();
    host_buffers.index_buffer.realloc(total_indices_count).unwrap();

    // Copy data into the buffer
    total_indices_count = 0;
    total_vertices_count = 0;
    for egui::epaint::ClippedPrimitive {
        clip_rect,
        primitive,
    } in output.paint_jobs.iter() {
        let mesh = match primitive {
            egui::epaint::Primitive::Mesh(mesh) => mesh,
            egui::epaint::Primitive::Callback(_) => panic!()
        };
        MaybeUninit::copy_from_slice(&mut host_buffers.vertex_buffer.deref_mut()[total_vertices_count..(total_vertices_count + mesh.vertices.len())], &mesh.vertices);
        total_vertices_count += mesh.vertices.len();
        MaybeUninit::copy_from_slice(&mut host_buffers.index_buffer.deref_mut()[total_indices_count..(total_indices_count + mesh.indices.len())], &mesh.indices);
        total_indices_count += mesh.indices.len();
    }

    if matches!(allocator.physical_device().properties().memory_model, PhysicalDeviceMemoryModel::Discrete | PhysicalDeviceMemoryModel::Bar) {
        let device_buffers = &mut *device_buffer;
        if device_buffers.vertex_buffer.len() < total_vertices_count {
            device_buffers.vertex_buffer.replace(|old| {
                commands.retain(Box::new(old));
                let mut buf = BufferArray::new_resource(allocator.clone(), vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
                buf.realloc(total_vertices_count).unwrap();
                RenderRes::new(buf)
            });
        }

        if device_buffers.index_buffer.len() < total_vertices_count {
            device_buffers.index_buffer.replace(|old| {
                commands.retain(Box::new(old));
                let mut buf = BufferArray::new_resource(allocator.clone(), vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
                buf.realloc(total_vertices_count).unwrap();
                RenderRes::new(buf)
            });
        }

        commands.record_commands().copy_buffer(&host_buffers.vertex_buffer, &mut device_buffers.vertex_buffer);
    }
}
