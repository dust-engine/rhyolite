use std::{fmt::Debug, hash::Hash, mem::MaybeUninit, ops::DerefMut};

use ash::vk;
use bevy::{
    ecs::system::{StaticSystemParam, SystemParam, SystemParamItem},
    prelude::*,
};

use crate::{
    buffer::BufferLike,
    commands::{ResourceTransitionCommands, TransferCommands},
    ecs::{queue_cap::IsQueueCap, Barriers, IntoRenderSystemConfigs, PerFrame, RenderCommands, RenderRes},
    Access, Allocator, BufferArray, Device, HasDevice,
};

/// A plugin that transfers buffer data immediately.
/// Suitable for situations where large amounts of data are transferred every frame.
///
/// On Discrete GPUs, the plugin creates per-frame host-visible buffers and one device-local buffer.
/// The entire buffer will be copied to the device-local buffer every frame on the transfer queue.
///
/// On ReBar and Integrated GPUs, the plugin creates per-frame device-local, host-visible buffers
/// and the data will be directly written to the buffers.
#[derive(Default)]
pub struct ImmediateBufferTransferPlugin<Manager> {
    usage_flags: vk::BufferUsageFlags,
    alignment: vk::DeviceSize,
    _manager: std::marker::PhantomData<Manager>,
}
impl<Manager> ImmediateBufferTransferPlugin<Manager> {
    /// Creates a new plugin with the given buffer usage flags.
    pub fn new(usage_flags: vk::BufferUsageFlags, alignment: vk::DeviceSize) -> Self {
        Self {
            usage_flags,
            alignment,
            _manager: std::marker::PhantomData,
        }
    }
}

#[derive(SystemSet)]
pub struct ImmediateBufferTransferSet<Manager> {
    _manager: std::marker::PhantomData<Manager>,
}
impl<Manager> Default for ImmediateBufferTransferSet<Manager> {
    fn default() -> Self {
        Self {
            _manager: std::marker::PhantomData,
        }
    }
}
impl<Manager> Clone for ImmediateBufferTransferSet<Manager> {
    fn clone(&self) -> Self {
        Self {
            _manager: self._manager.clone(),
        }
    }
}
impl<Manager> PartialEq for ImmediateBufferTransferSet<Manager> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl<Manager> Eq for ImmediateBufferTransferSet<Manager> {}
impl<Manager> Debug for ImmediateBufferTransferSet<Manager> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "ImmediateBufferTransferSet<{}>",
            std::any::type_name::<Manager>()
        ))
    }
}
impl<Manager> Hash for ImmediateBufferTransferSet<Manager> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self._manager.hash(state);
    }
}

impl<Manager: ImmediateBufferTransferManager + FromWorld> Plugin
    for ImmediateBufferTransferPlugin<Manager>
{
    fn build(&self, app: &mut App) {
        app.init_resource::<Manager>();

        app.add_systems(
            PostUpdate,
            collect_outputs::<Manager>.in_set(ImmediateBufferTransferSet::<Manager>::default()),
        );
    }
    fn finish(&self, app: &mut App) {
        let allocator: Allocator = app.world().resource::<Allocator>().clone();

        let mut device_buffers = None;

        if allocator
            .device()
            .physical_device()
            .properties()
            .memory_model
            .storage_buffer_should_use_staging()
        {
            app.add_systems(
                PostUpdate,
                (
                    resize_device_buffers::<Manager>.after(collect_outputs::<Manager>),
                    copy_buffers::<Manager>
                        .after(resize_device_buffers::<Manager>)
                        .with_barriers(copy_buffers_barrier::<Manager>),
                ),
            );
            device_buffers = Some(RenderRes::new(BufferArray::new_resource(
                allocator.clone(),
                self.usage_flags,
                self.alignment,
            )));
        }

        app.insert_resource(ImmediateBuffers::<Manager> {
            host_buffers: PerFrame::new(|_| {
                let usage = if allocator
                    .device()
                    .physical_device()
                    .properties()
                    .memory_model
                    .storage_buffer_should_use_staging()
                {
                    vk::BufferUsageFlags::empty()
                } else {
                    self.usage_flags
                };
                BufferArray::new_upload(allocator.clone(), usage)
            }),
            device_buffers,
            size: 0,
            usage_flags: self.usage_flags,
        });
    }
}

pub trait ImmediateBufferTransferManager: Resource {
    type Params: SystemParam;

    type Data: Send + Sync + 'static;
    /// Number of elements that the buffer needs to reserve for.
    fn data_size(&self, _params: &mut SystemParamItem<Self::Params>) -> usize;

    /// Writes the updated data into `dst`.
    fn collect_outputs(
        &mut self,
        params: &mut SystemParamItem<Self::Params>,
        dst: &mut [MaybeUninit<Self::Data>],
    );
}

#[derive(Resource)]
pub struct ImmediateBuffers<Manager: ImmediateBufferTransferManager> {
    host_buffers: PerFrame<BufferArray<Manager::Data>>,
    device_buffers: Option<RenderRes<BufferArray<Manager::Data>>>,
    size: usize,
    usage_flags: vk::BufferUsageFlags,
}
impl<Manager: ImmediateBufferTransferManager> ImmediateBuffers<Manager> {
    pub fn device_buffer<const Q: char>(&mut self, commands: &RenderCommands<Q> ) -> &BufferArray<Manager::Data> where (): IsQueueCap<Q> {
        if let Some(device_buffers) = self.device_buffers.as_ref() {
            device_buffers
        } else {
            self.host_buffers.on_frame(commands)
        }
    }
}

/// Collect output from egui and copy it into a host-side buffer
/// Create textures
fn collect_outputs<Manager: ImmediateBufferTransferManager>(
    mut manager: ResMut<Manager>,
    commands: RenderCommands<'t'>,
    allocator: Res<Allocator>,
    mut buffers: ResMut<ImmediateBuffers<Manager>>,
    mut params: StaticSystemParam<Manager::Params>,
) {
    let count = manager.data_size(&mut params);
    buffers.size = count;
    let host_buffers = buffers.host_buffers.on_frame(&commands);

    host_buffers.realloc(count).unwrap();

    manager.collect_outputs(&mut params, host_buffers.as_mut());
    host_buffers.flush(..count).unwrap();
}

/// Resize the device buffers if necessary. Only runs on Discrete GPUs.
fn resize_device_buffers<Manager: ImmediateBufferTransferManager>(
    mut buffers: ResMut<ImmediateBuffers<Manager>>,
    allocator: Res<Allocator>,
) {
    if buffers.device_buffers.as_ref().unwrap().len() < buffers.size {
        buffers.device_buffers = {
            let mut buf = BufferArray::new_resource(
                allocator.clone(),
                buffers.usage_flags | vk::BufferUsageFlags::TRANSFER_DST,
                1,
            );
            buf.realloc(buffers.size).unwrap();
            Some(RenderRes::new(buf))
        };
    }
}

fn copy_buffers_barrier<Manager: ImmediateBufferTransferManager>(
    mut barriers: In<Barriers>,
    mut buffers: ResMut<ImmediateBuffers<Manager>>,
) {
    if buffers.size > 0 {
        barriers.transition(
            buffers.device_buffers.as_mut().unwrap(),
            Access {
                access: vk::AccessFlags2::TRANSFER_WRITE,
                stage: vk::PipelineStageFlags2::COPY,
            },
            false,
            (),
        );
    }
}
/// Copy data from the host buffers to the device buffers. Only runs on Discrete GPUs.
fn copy_buffers<Manager: ImmediateBufferTransferManager>(
    mut buffers: ResMut<ImmediateBuffers<Manager>>,
    mut commands: RenderCommands<'t'>,
    allocator: Res<Allocator>,
) {
    if buffers.size > 0 {
        let host_buffers = buffers.host_buffers.on_frame(&commands);
        commands.copy_buffer(
            host_buffers.raw_buffer(),
            buffers.device_buffers.as_ref().unwrap().raw_buffer(),
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: buffers.size as u64 * std::mem::size_of::<Manager::Data>() as u64,
            }],
        );
    }
}
