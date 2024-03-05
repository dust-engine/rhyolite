#![feature(associated_type_defaults)]
#![feature(noop_waker)]

mod access;
mod alloc;
mod buffer;
mod command_pool;
mod commands;
pub mod debug;
mod device;
pub mod ecs;
pub mod extensions;
mod future;
mod image;
mod instance;
mod physical_device;
mod plugin;
mod queue;
mod semaphore;
mod surface;
mod swapchain;
pub mod pipeline;
mod shader;
mod deferred;

pub mod utils;

pub use access::*;
pub use alloc::Allocator;
pub use buffer::*;
pub use cstr::cstr;
pub use device::*;
pub use future::*;
pub use image::*;
pub use instance::*;
pub use physical_device::*;
pub use plugin::{RhyoliteApp, RhyolitePlugin};
pub use queue::*;
pub use surface::*;
pub use swapchain::*;
pub use shader::*;

pub use ash;
