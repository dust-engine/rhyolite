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

use std::sync::Arc;

use async_ash_alloc::Allocator;
use async_ash_core::{
    ash,
    ash::vk,
    copy_buffer, cstr,
    future::*,
    macros::{commands, gpu}, DeviceCreateInfo, FencePool, Instance, InstanceCreateInfo, PhysicalDevice,
    PhysicalDeviceFeatures, QueueFuture, QueueType, QueuesRouter, TimelineSemaphorePool,
};

fn main() {
    let entry = unsafe { ash::Entry::load().unwrap() };
    let instance =
        Arc::new(Instance::create(Arc::new(entry), &InstanceCreateInfo::default()).unwrap());
    let physical_device = PhysicalDevice::enumerate(&instance)
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
    let queues_router = QueuesRouter::new(&physical_device);

    let (device, mut queues) = physical_device
        .create_device(DeviceCreateInfo {
            enabled_features: PhysicalDeviceFeatures {
                v13: vk::PhysicalDeviceVulkan13Features {
                    synchronization2: vk::TRUE,
                    ..Default::default()
                },
                v12: vk::PhysicalDeviceVulkan12Features {
                    timeline_semaphore: vk::TRUE,
                    buffer_device_address: vk::TRUE,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..DeviceCreateInfo::with_queue_create_callback(|queue_family_index| {
                queues_router.priorities(queue_family_index)
            })
        })
        .unwrap();
    let allocator = Allocator::new(device.clone());

    let mut shared_command_pools = queues.make_shared_command_pools();
    let mut shared_semaphore_pool = TimelineSemaphorePool::new(device.clone());
    let mut shared_fence_pool = FencePool::new(device.clone());

    let src = allocator
        .create_device_buffer_with_data(
            &[10, 0, 2, 2],
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .unwrap();
    let mut dst = allocator
        .create_device_buffer_uninit(
            4,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .unwrap();
    let mut read_back = allocator.create_readback_buffer(4).unwrap();

    use async_ash_core::debug::command_debug;
    let mut state = Default::default();
    let wait = queues.submit(
        gpu! {
            commands! {
                let src_buffer = src.await;
                let  mut dst_buffer = import!(dst);
                copy_buffer(&mut dst_buffer, &src_buffer).await;


                let mut read_back = import!(&mut read_back);
                copy_buffer(&mut read_back, &dst_buffer).await;
            }.schedule_on_queue(queues_router.of_type(QueueType::Transfer)).await;
        },
        &mut shared_command_pools,
        &mut shared_semaphore_pool,
        &mut shared_fence_pool,
        &mut state,
    );
    futures::executor::block_on(wait);

    
    let buf = read_back.contents().unwrap();
    println!("{:?}", buf);
}
 
// TODO: Add ability to skip empty stages.