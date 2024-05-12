use std::collections::{BTreeMap, BTreeSet, VecDeque};

use ash::vk;
use bevy::{
    app::{App, Plugin, PostUpdate},
    asset::{Asset, AssetApp, AssetEvent, AssetId, Assets, Handle},
    ecs::{
        entity::Entity,
        event::EventReader,
        query::{Added, Changed},
        removal_detection::RemovedComponents,
        system::{Local, Query, Res, ResMut, StaticSystemParam, SystemParam, SystemParamItem},
    },
};

use crate::{
    commands::{BatchCopy, TransferCommands},
    task::{AsyncCommandRecorder, AsyncComputeTask, AsyncTaskPool},
};

pub trait AssetUpload: Asset {
    type GPUAsset: Asset;
    type Params: SystemParam;
    fn upload_asset(
        &self,
        commands: &mut impl TransferCommands,
        params: &mut SystemParamItem<Self::Params>,
    ) -> Self::GPUAsset;
}

/// For each asset type, for all new assets, schedule a task to upload them to the GPU.
fn upload_asset<T: AssetUpload>(
    mut events: EventReader<AssetEvent<T>>,
    mut gpu_assets: ResMut<Assets<T::GPUAsset>>,
    cpu_assets: Res<Assets<T>>,
    mut params: StaticSystemParam<T::Params>,
    mut task_pool: ResMut<AsyncTaskPool>,
    mut tasks: Local<VecDeque<AsyncComputeTask<Vec<(AssetId<T::GPUAsset>, T::GPUAsset)>>>>,
) {
    // Complete tasks
    while let Some(task_ref) = tasks.front() {
        if !task_ref.is_finished() {
            break;
        }
        let task = tasks.pop_front().unwrap();
        let results = task_pool.wait_blocked(task);
        for (id, gpu_asset) in results {
            gpu_assets.insert(id, gpu_asset);
        }
    }

    // For all added events, upload the asset to the GPU
    if events.len() > 0 {
        let mut command_recorder = task_pool.spawn_transfer();
        let mut batch_copy = BatchCopy::new(&mut command_recorder);
        let mut results: Vec<(AssetId<T::GPUAsset>, T::GPUAsset)> =
            Vec::with_capacity(events.len());
        for event in events.read() {
            match event {
                AssetEvent::Added { id } | AssetEvent::LoadedWithDependencies { id } => {
                    let asset = cpu_assets.get(*id).unwrap();
                    let gpu_asset = asset.upload_asset(&mut batch_copy, &mut params);
                    let untyped_id = id.untyped();
                    gpu_assets.reserve_asset_id(untyped_id);
                    results.push((untyped_id.typed_unchecked(), gpu_asset));
                }
                _ => {}
            }
        }
        drop(batch_copy);
        if results.len() > 0 {
            tasks.push_back(command_recorder.finish(results, vk::PipelineStageFlags2::TRANSFER));
        }
    }
}

pub struct AssetUploadPlugin<T: AssetUpload> {
    _marker: std::marker::PhantomData<T>,
}
impl<T: AssetUpload> Plugin for AssetUploadPlugin<T> {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, upload_asset::<T>)
            .init_asset::<T>()
            .init_asset::<T::GPUAsset>();
    }
}
impl<T: AssetUpload> Default for AssetUploadPlugin<T> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}
