#![feature(generators, generator_trait)]
#![feature(trait_alias)]
#![feature(cstr_from_bytes_until_nul)]
#![feature(negative_impls)]
#![feature(specialization)]
#![feature(array_zip)]

pub use cstr::cstr;

pub extern crate ash;
pub extern crate async_ash_macro as macros;

pub mod commands;
pub mod debug;
mod device;
pub mod future;
mod instance;
mod physical_device;
pub mod queue;
mod resources;
mod semaphore;
pub mod swapchain;
pub mod utils;

pub use device::{Device, HasDevice};
pub use instance::*;
pub use physical_device::*;
pub use queue::*;
pub use resources::*;
pub use semaphore::*;

// TODO: Test two consequtive reads, with different image layouts.
