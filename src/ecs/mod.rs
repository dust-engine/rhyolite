mod commands;
mod config;
mod pass;
mod res;
#[cfg(test)]
mod tests;

use ash::vk;

pub use commands::*;
pub use config::*;
pub use pass::*;
pub use res::*;

pub struct RenderResAccess {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}
