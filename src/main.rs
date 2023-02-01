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
#![feature(type_alias_impl_trait)]
#![feature(specialization)]

use std::{
    cell::RefCell,
    ops::{Deref, DerefMut, Generator},
    pin::Pin,
    sync::Arc,
    task::Poll,
};

use ash::vk::{self};
use async_ash::{
    commands::{use_command_buffer, use_command_pool},
    cstr,
    future::*,
    Buffer, CopyBufferFuture, DeviceCreateInfo, FencePool, InstanceCreateInfo,
    PhysicalDeviceFeatures, QueueFuture, QueueFuturePoll, QueueMask, QueueRef, QueueType,
    QueuesRouter, SubmissionContext, TimelineSemaphorePool,
};
use pin_project::pin_project;

fn main() {
    use async_ash_macro::{commands, gpu, join};
    let entry = unsafe { ash::Entry::load().unwrap() };
    let instance = Arc::new(
        async_ash::Instance::create(Arc::new(entry), &InstanceCreateInfo::default()).unwrap(),
    );
    let physical_device = async_ash::PhysicalDevice::enumerate(&instance)
        .unwrap()
        .into_iter()
        .skip(1)
        .next()
        .unwrap();
    println!(
        "Using {:?}, api version {:?}, driver version {:?}",
        physical_device.properties().device_name(),
        physical_device.properties().api_version(),
        physical_device.properties().driver_version()
    );
    let mut queues_router = QueuesRouter::new(&physical_device);

    let (device, mut queues) = physical_device
        .create_device(DeviceCreateInfo {
            enabled_features: PhysicalDeviceFeatures {
                v13: vk::PhysicalDeviceVulkan13Features {
                    synchronization2: vk::TRUE,
                    ..Default::default()
                },
                v12: vk::PhysicalDeviceVulkan12Features {
                    timeline_semaphore: vk::TRUE,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..DeviceCreateInfo::with_queue_create_callback(|queue_family_index| {
                queues_router.priorities(queue_family_index)
            })
        })
        .unwrap();

    let mut shared_command_pools = queues.make_shared_command_pools();
    let mut shared_semaphore_pool = TimelineSemaphorePool::new(device.clone());
    let mut shared_fence_pool = FencePool::new(device.clone());

    let src1 = Buffer::from_raw(device.clone(), unsafe { std::mem::transmute(231_usize) });
    let src2 = Buffer::from_raw(device.clone(), unsafe { std::mem::transmute(232_usize) });
    let src3 = Buffer::from_raw(device.clone(), unsafe { std::mem::transmute(233_usize) });

    use async_ash::debug::command_debug;
    let mut state = Default::default();
    let wait = queues.submit(
        gpu! {
            commands! {
                command_debug(cstr!("hello")).await;
                command_debug(cstr!("world")).await;
            }.schedule().await;

            commands! {
                command_debug(cstr!("im")).await;
                command_debug(cstr!("coming")).await;
            }.schedule_on_queue(queues_router.of_type(QueueType::Transfer)).await;
        },
        &mut shared_command_pools,
        &mut shared_semaphore_pool,
        &mut shared_fence_pool,
        &mut state,
    );

    futures::executor::block_on(wait);
}
