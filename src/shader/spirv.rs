use std::future::Future;

use crate::Device;

use ash::vk;
use bevy::asset::saver::{AssetSaver, SavedAsset};
use bevy::asset::{Asset, AssetLoader, AsyncReadExt, AsyncWriteExt};
use bevy::ecs::world::FromWorld;
use bevy::reflect::TypePath;
use bevy::utils::ConditionalSendFuture;

use thiserror::Error;

use super::ShaderModule;

#[derive(TypePath, Asset)]
pub struct SpirvShaderSource {
    pub(crate) source: Vec<u32>,
}

#[derive(Debug, Error)]
pub enum SpirvLoaderError {
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("vulkan error: {0:?}")]
    VkError(#[from] vk::Result),
}
pub struct SpirvLoader {
    device: Device,
}
impl FromWorld for SpirvLoader {
    fn from_world(world: &mut bevy::ecs::world::World) -> Self {
        let device = world.resource::<Device>().clone();
        Self::new(device)
    }
}
impl SpirvLoader {
    pub(crate) fn new(device: Device) -> Self {
        Self { device }
    }
}
impl AssetLoader for SpirvLoader {
    type Asset = ShaderModule;
    type Settings = ();
    type Error = SpirvLoaderError;
    fn load<'a>(
        &'a self,
        reader: &'a mut bevy::asset::io::Reader,
        _settings: &'a Self::Settings,
        _load_context: &'a mut bevy::asset::LoadContext,
    ) -> impl ConditionalSendFuture + Future<Output = Result<Self::Asset, Self::Error>> {
        let device = self.device.clone();
        return Box::pin(async move {
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes).await?;
            assert!(bytes.len() % 4 == 0);
            let bytes = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4)
            };
            let shader = ShaderModule::new(device, bytes)?;
            Ok(shader)
        });
    }

    fn extensions(&self) -> &[&str] {
        &["spv"]
    }
}

pub struct SpirvSaver;
impl AssetSaver for SpirvSaver {
    type Asset = SpirvShaderSource;
    type Settings = ();
    type OutputLoader = SpirvLoader;
    type Error = std::io::Error;
    fn save<'a>(
        &'a self,
        writer: &'a mut bevy::asset::io::Writer,
        asset: SavedAsset<'a, Self::Asset>,
        _settings: &'a Self::Settings,
    ) -> impl ConditionalSendFuture + Future<Output = Result<(), Self::Error>> {
        Box::pin(async move {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    asset.source.as_ptr() as *const u8,
                    asset.source.len() * 4,
                )
            };
            writer.write_all(slice).await?;
            Ok(())
        })
    }
}
