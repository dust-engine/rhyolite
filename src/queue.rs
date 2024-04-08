use std::sync::{atomic::AtomicBool, Mutex};

use ash::vk;
use bevy::ecs::system::Resource;

use crate::Device;

/// Index of a created queue
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct QueueRef {
    pub(crate) index: u32,
    pub(crate) family: u32,
    pub(crate) caps: vk:: QueueFlags,
}
impl QueueRef {
    pub fn null() -> Self {
        QueueRef {
            index: 0,
            family: 0,
            caps: vk::QueueFlags::empty()
        }
    }
    pub fn is_null(&self) -> bool {
        self.index == u32::MAX
    }
}
impl Default for QueueRef {
    fn default() -> Self {
        Self::null()
    }
}

enum QueueSharing {
    Shared(Mutex<vk::Queue>),
    Exclusive {
        queue: vk::Queue,
        marker: AtomicBool,
    }
}

#[derive(Resource)]
pub struct QueuesRouter {
    queues: Vec<QueueSharing>,
    queue_refs: Vec<QueueRef>,
}

const QUEUE_FLAGS_ASYNC: vk::QueueFlags = vk::QueueFlags::from_raw(1 << 31);

impl QueuesRouter {
    pub fn with_caps(&self, required_caps: vk::QueueFlags, preferred_caps: vk::QueueFlags) -> Option<QueueRef> {
        if !preferred_caps.is_empty() {
            return self.queue_refs.iter().find(|r| r.caps.contains(required_caps | preferred_caps)).map(|r| *r)
        }

        return self.queue_refs.iter().find(|r| r.caps.contains(required_caps)).map(|r| *r)
    }
    pub(crate) fn find_with_queue_family_properties(
        available_queue_family: &[vk::QueueFamilyProperties],
    ) -> Vec<vk::DeviceQueueCreateInfo> {
        // Create 2 of each queue family
        available_queue_family.iter().enumerate().take(3).map(|(queue_family_index, props)| {
            let priority: &'static [f32] = if props.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
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
        }).collect::<Vec<_>>()
    }
    pub(crate) fn create(device: &ash::Device, create_info: &[vk::DeviceQueueCreateInfo],
        available_queue_family: &[vk::QueueFamilyProperties],) -> Self {
        let (queues, mut queue_refs) : (Vec<_>, Vec<_>) = create_info.iter().flat_map(|info| {
            let queue_family = &available_queue_family[info.queue_family_index as usize];
            (0..info.queue_count).map(|i| {
                let queue = unsafe{ device.get_device_queue(info.queue_family_index, i) };
                let sharing = if info.queue_count >= 2 {
                    QueueSharing::Exclusive { queue, marker: AtomicBool::new(false) }
                } else {
                    QueueSharing::Shared(Mutex::new(queue))
                };
                let r = QueueRef {
                    index: 0,
                    family: info.queue_family_index,
                    caps: queue_family.queue_flags,
                };
                (sharing, r)
            })
        }).unzip();
        queue_refs.iter_mut().enumerate().for_each(|(i, r)| r.index = i as u32);
        queue_refs.sort_by_cached_key(|i| i.caps.as_raw().count_ones());
        Self {
            queues,
            queue_refs,
        }
    }
}

const PRIORITY_HIGH: [f32; 2] = [1.0, 0.1];
const PRIORITY_MEDIUM: [f32; 2] = [0.5, 0.1];
const PRIORITY_LOW: [f32; 2] = [0.0, 0.0];
