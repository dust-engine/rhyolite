use std::{
    marker::PhantomData, ops::{Deref, DerefMut}
};

use ash::vk;
use bevy::{ecs::{component::{ComponentId, Tick}, system::{SystemMeta, SystemParam}, world::unsafe_world_cell::UnsafeWorldCell}, prelude::{FromWorld, ResMut, Resource, World}};
use crate::{command::CommandPool, Device};

const PRIORITY_HIGH: [f32; 2] = [1.0, 0.1];
const PRIORITY_MEDIUM: [f32; 2] = [0.5, 0.1];
const PRIORITY_LOW: [f32; 2] = [0.0, 0.0];


pub struct QueueInner {
    pub device: Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
}


/// Returns a good queue creation strategies for many interactive applications.
/// 
/// We attempts to create two queues for the first three queue families.
pub(crate) fn find_default_queue_create_info(available_queue_family: &[vk::QueueFamilyProperties]) -> Vec<vk::DeviceQueueCreateInfo>  {
    // Create 2 of each queue family
    available_queue_family
        .iter()
        .enumerate()
        .take(3)
        .map(|(queue_family_index, props)| {
            let priority: &'static [f32] =
                if props.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    &PRIORITY_HIGH
                } else if props.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    &PRIORITY_MEDIUM
                } else {
                    &PRIORITY_LOW
                };
            let queue_count = if props.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                1
            } else {
                props.queue_count.min(2)
            };
            vk::DeviceQueueCreateInfo {
                queue_family_index: queue_family_index as u32,
                queue_count,
                p_queue_priorities: priority.as_ptr(),
                ..Default::default()
            }
        })
        .collect::<Vec<_>>()
}

#[derive(Resource)]
pub struct QueueConfiguration {
    families: Vec<QueueConfigurationFamily>
}
struct QueueConfigurationFamily {
    /// Pointing to a QueueInner
    queues: smallvec::SmallVec<[ComponentId; 2]>,
    shared_command_pool: ComponentId,
    flags: vk::QueueFlags,
}

impl QueueConfiguration {
    /// Safety: must only be called once per device, otherwise we end up with multiple copies of the same queue,
    /// violating assumptions on exclusive queue references.
    pub(crate) unsafe fn create_in_world(
        world: &mut World,
        queue_create_info: &[vk::DeviceQueueCreateInfo],
        queue_family_info: &[vk::QueueFamilyProperties],
    ) {
        use bevy::ecs::component::ComponentDescriptor;
        use bevy::ptr::OwningPtr;
        use smallvec::SmallVec;
        let device = world.resource::<Device>().clone();

        let mut this = QueueConfiguration {
            families: queue_family_info.iter().map(|info| QueueConfigurationFamily {
                flags: info.queue_flags,
                shared_command_pool: ComponentId::new(0),
                queues: SmallVec::new(),
            }).collect()
        };

        for vk::DeviceQueueCreateInfo { queue_count, queue_family_index, ..} in queue_create_info.iter() {
            let family = &mut this.families[*queue_family_index as usize];


            assert_eq!(family.shared_command_pool.index(), 0, "Each queue family index inside `queue_create_info` should be unique");

            // Create shared command pool.
            let command_pool = CommandPool::new(device.clone(), *queue_family_index, vk::CommandPoolCreateFlags::TRANSIENT).unwrap();
            let component_id = world.init_component_with_descriptor(ComponentDescriptor::new_resource::<CommandPool>());
            OwningPtr::make(command_pool, |ptr| unsafe {
                // SAFETY: component_id was just initialized and corresponds to resource of type R.
                world.insert_resource_by_id(component_id, ptr);
            });
            family.shared_command_pool = component_id;


            for queue_index in 0..*queue_count {
                let queue = unsafe {
                    device.get_device_queue(*queue_family_index, queue_index)
                };
                let component_id = world.init_component_with_descriptor(ComponentDescriptor::new_resource::<QueueInner>());
                let queue_inner = QueueInner {
                    device: device.clone(),
                    queue,
                    queue_family: *queue_family_index
                };
                OwningPtr::make(queue_inner, |ptr| unsafe {
                    // SAFETY: component_id was just initialized and corresponds to resource of type R.
                    world.insert_resource_by_id(component_id, ptr);
                });
                family.queues.push(component_id);
            }
        }
        world.insert_resource(this);
    }
}



pub trait QueueSelector {
    fn family_index(config: &QueueConfiguration) -> u32;
    fn component_id(config: &QueueConfiguration) -> ComponentId;
    fn shared_command_pool_component_id(config: &QueueConfiguration) -> ComponentId {
        config.families[Self::family_index(config) as usize].shared_command_pool
    }
}
pub struct Queue<'a, T: QueueSelector> {
    queue: &'a mut QueueInner,
    _marker: PhantomData<T>
}
impl<'a, T: QueueSelector> Deref for Queue<'a, T> {
    type Target = QueueInner;

    fn deref(&self) -> &Self::Target {
        self.queue
    }
}
impl<'a, T: QueueSelector> DerefMut for Queue<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.queue
    }
}
unsafe impl<'a, T: QueueSelector> SystemParam for Queue<'a, T> {
    type State = ComponentId;

    type Item<'world, 'state> = Queue<'world, T>;

    fn init_state(world: &mut World, system_meta: &mut SystemMeta) -> Self::State {
        let config = world.resource::<QueueConfiguration>();
        let component_id = T::component_id(config);

        let combined_access = system_meta.component_access_set.combined_access();
        if combined_access.has_write(component_id) {
            panic!(
                "error[B0002]: ResMut<{}> in system {} conflicts with a previous ResMut<{0}> access. Consider removing the duplicate access. See: https://bevyengine.org/learn/errors/#b0002",
                std::any::type_name::<Self>(), system_meta.name);
        } else if combined_access.has_read(component_id) {
            panic!(
                "error[B0002]: ResMut<{}> in system {} conflicts with a previous Res<{0}> access. Consider removing the duplicate access. See: https://bevyengine.org/learn/errors/#b0002",
                std::any::type_name::<Self>(), system_meta.name);
        }
        system_meta
            .component_access_set
            .add_unfiltered_write(component_id);

        let archetype_component_id = world
            .get_resource_archetype_component_id(component_id)
            .unwrap();
        system_meta
            .archetype_component_access
            .add_write(archetype_component_id);
        component_id
    }

    unsafe fn get_param<'world, 'state>(
        state: &'state mut Self::State,
        system_meta: &SystemMeta,
        world: UnsafeWorldCell<'world>,
        _change_tick: Tick,
    ) -> Self::Item<'world, 'state> {
        let value = world
            .get_resource_mut_by_id(*state)
            .unwrap_or_else(|| {
                panic!(
                    "Resource requested by {} does not exist: {}",
                    system_meta.name,
                    std::any::type_name::<T>()
                )
            });
        Queue {
            queue: value.value.deref_mut::<QueueInner>(),
            _marker: PhantomData
        }
    }
}


pub mod selectors {
    use ash::vk;

    use super::{QueueSelector, QueueConfiguration, ComponentId};

    macro_rules! return_queue {
        ($config: expr) => {
            let family_index = Self::family_index($config);
            let family = &$config.families[family_index as usize];
            if ASYNC {
                if family.queues.len() >= 2 {
                    return family.queues[1];
                } else {
                    return family.queues[0];
                }
            } else {
                return family.queues[0];
            }
        };
    }

    /// Selects the first queue with [`vk::QueueFlags::GRAPHICS`]
    /// If ASYNC is set to true, when multiple queues are available for the target queue family, this will select the secondary queue.
    pub struct Graphics<const ASYNC: bool = false>;
    impl<const ASYNC: bool> QueueSelector for Graphics<ASYNC> {
        fn family_index(config: &QueueConfiguration) -> u32 {
            for (family_index, family) in config.families.iter().enumerate() {
                if family.flags.contains(vk::QueueFlags::GRAPHICS) && !family.queues.is_empty(){
                    return family_index as u32;
                }
            }
            panic!("Cannot find Render queue family!")
        }
        fn component_id(config: &QueueConfiguration) -> ComponentId {
            return_queue!(config);
        }
    }


    /// Selects the queue with both [`vk::QueueFlags::GRAPHICS`] and [`vk::QueueFlags::COMPUTE`].
    /// If no such queue exists, select any queue with [`vk::QueueFlags::COMPUTE`].
    pub struct UniversalCompute<const ASYNC: bool = false>;
    impl<const ASYNC: bool> QueueSelector for UniversalCompute<ASYNC> {
        fn family_index(config: &QueueConfiguration) -> u32 {
            for (family_index, family) in config.families.iter().enumerate() {
                if family.flags.contains(vk::QueueFlags::GRAPHICS) && family.flags.contains(vk::QueueFlags::COMPUTE) && !family.queues.is_empty(){
                    return family_index as u32;
                }
            }
            for (family_index, family) in config.families.iter().enumerate() {
                if family.flags.contains(vk::QueueFlags::COMPUTE) && !family.queues.is_empty(){
                    return family_index as u32;
                }
            }
            panic!("Cannot find Compute queue!")
        }
        fn component_id(config: &QueueConfiguration) -> ComponentId {
            return_queue!(config);
        }
    }

    /// Selects the first queue with dedicated [`vk::QueueFlags::COMPUTE`] flag. If no such queue exists, returns any queue with compute capabilities.
    pub struct DedicatedCompute<const ASYNC: bool = false>;

    impl<const ASYNC: bool> QueueSelector for DedicatedCompute<ASYNC> {
        fn family_index(config: &QueueConfiguration) -> u32 {
            // Find the dedicated compute queue with no graphics capabilities.
            for (family_index, family) in config.families.iter().enumerate() {
                if family.flags.contains(vk::QueueFlags::COMPUTE) && !family.flags.contains(vk::QueueFlags::GRAPHICS) && !family.queues.is_empty(){
                    return family_index as u32;
                }
            }

            // Find any queue with compute capabilities.
            for (family_index, family) in config.families.iter().enumerate() {
                if family.flags.contains(vk::QueueFlags::COMPUTE) && !family.queues.is_empty(){
                    return family_index as u32;
                }
            }
            panic!("Cannot find dedicated compute queue!")
        }
        fn component_id(config: &QueueConfiguration) -> ComponentId {
            return_queue!(config);
        }
    }

    /// Selects the first queue with dedicated [`vk::QueueFlags::TRANSFER`] flag. If no such queue exists, returns any queue with transfer capabilities.
    pub struct DedicatedTransfer<const ASYNC: bool = false>;

    impl<const ASYNC: bool> QueueSelector for DedicatedTransfer<ASYNC> {
        fn family_index(config: &QueueConfiguration) -> u32 {
            // Find the dedicated compute queue with no graphics or compute capabilities.
            for (family_index, family) in config.families.iter().enumerate() {
                if family.flags.contains(vk::QueueFlags::TRANSFER) && !family.flags.contains(vk::QueueFlags::GRAPHICS) && !family.flags.contains(vk::QueueFlags::COMPUTE) && !family.queues.is_empty(){
                    return family_index as u32;
                }
            }

            // Find queue with no graphics capabilities. Prefer compute-based transfers.
            for (family_index, family) in config.families.iter().enumerate() {
                if family.flags.contains(vk::QueueFlags::TRANSFER) && !family.flags.contains(vk::QueueFlags::COMPUTE) && !family.queues.is_empty(){
                    return family_index as u32;
                }
            }
            // Find any queue with the transfer flag set.
            for (family_index, family) in config.families.iter().enumerate() {
                if family.flags.contains(vk::QueueFlags::TRANSFER) || !family.queues.is_empty(){
                    return family_index as u32;
                }
            }
            // Find any queue with compute flag.
            for (family_index, family) in config.families.iter().enumerate() {
                if family.flags.contains(vk::QueueFlags::COMPUTE) || !family.queues.is_empty(){
                    return family_index as u32;
                }
            }
            // Find any queue with graphics flag.
            for (family_index, family) in config.families.iter().enumerate() {
                if family.flags.contains(vk::QueueFlags::GRAPHICS) || !family.queues.is_empty(){
                    return family_index as u32;
                }
            }
            panic!("Cannot find dedicated transfer queue!")
        }

        fn component_id(config: &QueueConfiguration) -> ComponentId {
            return_queue!(config);
        }
    }

    pub type AsyncGraphics = Graphics<true>;
    pub type AsyncCompute = DedicatedCompute<true>;
    pub type AsyncTransfer = DedicatedTransfer<true>;

}
