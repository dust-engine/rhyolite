#![feature(generators, generator_trait)]
#![feature(trait_alias)]
#![feature(cstr_from_bytes_until_nul)]
#![feature(negative_impls)]
#![feature(specialization)]
#![feature(array_zip)]

pub use cstr::cstr;

pub extern crate ash;
pub extern crate rhyolite_macro as macros;
pub extern crate self as rhyolite;

mod allocator;
pub mod commands;
pub mod debug;
mod device;
mod dho;
pub mod future;
mod instance;
pub mod shader;
mod physical_device;
pub mod queue;
mod resources;
mod semaphore;
mod surface;
pub mod swapchain;
pub mod utils;

pub use device::{Device, HasDevice};
pub use instance::*;
pub use physical_device::*;
pub use queue::*;
pub use resources::*;
pub use semaphore::*;
pub use surface::*;
pub use swapchain::*;

pub use allocator::Allocator;
// TODO: Test two consequtive reads, with different image layouts.
