use std::sync::Arc;

use super::Queue;
use crate::{Device, PhysicalDevice};
use ash::vk;

#[derive(Clone, Copy, Debug)]
pub enum QueueType {
    Graphics = 0,
    Compute = 1,
    Transfer = 2,
    SparseBinding = 3,
}

impl QueueType {
    pub fn priority(&self) -> &'static f32 {
        [
            &QUEUE_PRIORITY_HIGH,
            &QUEUE_PRIORITY_HIGH,
            &QUEUE_PRIORITY_MID,
            &QUEUE_PRIORITY_LOW,
        ][*self as usize]
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct QueueIndex(pub(crate) usize);

/// A collection of QueueDispatcher. It creates manages a number of QueueDispatcher based on the device-specific queue flags.
/// On submission, it routes the submission to the queue with the minimal number of declared capabilities.
///
/// The current implementation creates at most one queue for each queue family.
pub struct Queues {
    queues: Vec<Queue>,
    queue_type_to_dispatcher: [u32; 4],
}

impl Queues {
    pub fn of_type(&self, ty: QueueType) -> &Queue {
        self.of_index(self.index_of_type(ty))
    }
    pub fn of_index(&self, index: QueueIndex) -> &Queue {
        &self.queues[index.0]
    }
    pub fn index_of_type(&self, ty: QueueType) -> QueueIndex {
        let i = self.queue_type_to_dispatcher[ty as usize];
        QueueIndex(i as usize)
    }
}

impl Queues {
    // Safety: Can only be called once for each device.
    pub(crate) unsafe fn from_device(device: &Arc<Device>, create_info: &QueuesCreateInfo) -> Self {
        let queue_dispatchers: Vec<Queue> = create_info
            .create_infos
            .iter()
            .zip(create_info.queue_family_to_types.iter())
            .enumerate()
            .map(|(index, (queue_create_info, ty))| {
                // We always just create at most one queue for each queue family
                let queue = device.get_device_queue(queue_create_info.queue_family_index, 0);
                Queue {
                    device: device.clone(),
                    queue,
                    family_index: queue_create_info.queue_family_index,
                }
            })
            .collect();
        Queues {
            queues: queue_dispatchers,
            queue_type_to_dispatcher: create_info.queue_type_to_family,
        }
    }
}

pub struct QueuesCreateInfo {
    pub(crate) create_infos: Vec<vk::DeviceQueueCreateInfo>,
    pub(crate) queue_family_to_types: Vec<Option<QueueType>>,
    pub(crate) queue_type_to_family: [u32; 4],
}

const QUEUE_PRIORITY_HIGH: f32 = 1.0;
const QUEUE_PRIORITY_MID: f32 = 0.5;
const QUEUE_PRIORITY_LOW: f32 = 0.1;

impl QueuesCreateInfo {
    pub fn find(physical_device: &PhysicalDevice) -> QueuesCreateInfo {
        let available_queue_family = physical_device.get_queue_family_properties();
        Self::find_with_queue_family_properties(&available_queue_family)
    }
    fn find_with_queue_family_properties(
        available_queue_family: &[vk::QueueFamilyProperties],
    ) -> QueuesCreateInfo {
        let graphics_queue_family = available_queue_family
            .iter()
            .enumerate()
            .filter(|&(_i, family)| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .max_by_key(|&(_i, family)| {
                let mut priority: i32 = 0;
                if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    priority -= 1;
                }
                if family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING) {
                    priority -= 1;
                }
                priority
            })
            .unwrap()
            .0 as u32;
        let compute_queue_family = available_queue_family
            .iter()
            .enumerate()
            .filter(|&(_id, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .max_by_key(|&(_, family)| {
                // Use first compute-capable queue family
                let mut priority: i32 = 0;
                if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    priority -= 100;
                }
                if family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING) {
                    priority -= 1;
                }
                priority
            })
            .unwrap()
            .0 as u32;
        let transfer_queue_family = available_queue_family
            .iter()
            .enumerate()
            .max_by_key(|&(_, family)| {
                // Use first compute-capable queue family
                let mut priority: i32 = 0;
                if family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                    priority += 100;
                }
                if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    priority -= 10;
                }
                if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    priority -= 20;
                }
                if family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING) {
                    priority -= 1;
                }
                priority
            })
            .unwrap()
            .0 as u32;
        let sparse_binding_queue_family = available_queue_family
            .iter()
            .enumerate()
            .filter(|&(_id, family)| family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING))
            .max_by_key(|&(_, family)| {
                // Use first compute-capable queue family
                let mut priority: i32 = 0;
                if family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                    priority -= 1;
                }
                if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    priority -= 10;
                }
                if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    priority -= 20;
                }
                priority
            })
            .unwrap()
            .0 as u32;

        let mut queue_family_to_types: Vec<Option<QueueType>> =
            vec![None; available_queue_family.len()];
        queue_family_to_types[graphics_queue_family as usize] = Some(QueueType::Graphics);
        queue_family_to_types[compute_queue_family as usize] = Some(QueueType::Compute);
        queue_family_to_types[transfer_queue_family as usize] = Some(QueueType::Transfer);
        queue_family_to_types[sparse_binding_queue_family as usize] =
            Some(QueueType::SparseBinding);

        let queue_type_to_family: [u32; 4] = [
            graphics_queue_family,
            compute_queue_family,
            transfer_queue_family,
            sparse_binding_queue_family,
        ];

        let create_infos = queue_family_to_types
            .iter()
            .enumerate()
            .map(
                |(queue_family_index, queue_type)| vk::DeviceQueueCreateInfo {
                    flags: vk::DeviceQueueCreateFlags::empty(),
                    queue_family_index: queue_family_index as u32,
                    queue_count: 1,
                    p_queue_priorities: queue_type
                        .map_or(&QUEUE_PRIORITY_LOW, |queue_type| queue_type.priority()),
                    ..Default::default()
                },
            )
            .collect();

        QueuesCreateInfo {
            create_infos,
            queue_family_to_types,
            queue_type_to_family,
        }
    }
    pub fn queue_family_index_for_type(&self, ty: QueueType) -> u32 {
        self.queue_type_to_family[ty as usize]
    }
    pub fn assigned_queue_type_for_family_index(
        &self,
        queue_family_index: u32,
    ) -> Option<QueueType> {
        self.queue_family_to_types[queue_family_index as usize]
    }
}
