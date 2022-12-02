#![feature(min_specialization)]
#![feature(array_methods)]
#![feature(waker_getters)]
#![feature(pin_macro)]
#![feature(generators, generator_trait, generator_clone)]
#![feature(return_position_impl_trait_in_trait)]
#![feature(async_fn_in_trait)]
#![feature(iter_from_generator)]
#![feature(async_closure)]
#![feature(trait_alias)]

extern crate self as async_ash;

mod debug;
mod device;
pub mod future;
mod instance;
mod physical_device;
mod queue;
mod resources;
mod semaphore;

pub use debug::{DebugObject, DebugUtilsMessenger};
pub use device::{Device, HasDevice};
pub use instance::Instance;
pub use physical_device::*;
pub use queue::*;
pub use resources::*;
pub use semaphore::*;
