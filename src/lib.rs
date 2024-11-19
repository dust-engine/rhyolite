#![feature(associated_type_defaults)]
#![feature(generic_const_exprs)]
#![feature(get_mut_unchecked)]
#![feature(noop_waker)]
#![feature(ptr_metadata)]
#![feature(alloc_layout_extra)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(let_chains)]
#![feature(type_changing_struct_update)]
#![feature(ptr_as_uninit)]
#![feature(specialization)]
#![feature(context_ext)]
#![feature(local_waker)]

mod alloc;
//pub mod buffer;
mod command;
//pub mod commands;
//pub mod debug;
pub mod deferred;
mod device;
//pub mod ecs;
pub mod extensions;
mod future;
mod image;
mod instance;
mod physical_device;
pub mod pipeline;
mod plugin;
mod query_pool;
mod queue;
mod sampler;
mod semaphore;
pub mod shader;
mod surface;
//mod swapchain;
//pub mod task;

pub mod utils;

pub use alloc::Allocator;
pub use ash;
pub use cstr::cstr;
pub use deferred::*;
pub use device::*;
pub use future::*;
pub use image::*;
pub use instance::*;
pub use physical_device::*;
pub use plugin::{RhyoliteApp, RhyolitePlugin};
pub use query_pool::*;
pub use queue::*;
pub use sampler::Sampler;
pub use surface::*;
pub use vk_mem;
