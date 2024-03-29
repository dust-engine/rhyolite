use ash::vk;
use bevy::ecs::system::Resource;

/// Index of a created queue
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct QueueRef {
    pub(crate) index: u32,
    pub(crate) family: u32,
}
impl QueueRef {
    pub fn null() -> Self {
        QueueRef {
            index: 0,
            family: 0,
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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum QueueType {
    Graphics = 0,
    Compute = 1,
    Transfer = 2,
    SparseBinding = 3,
}

impl QueueType {
    pub fn mask(&self) -> vk::QueueFlags {
        match self {
            QueueType::Graphics => vk::QueueFlags::GRAPHICS,
            QueueType::Compute => vk::QueueFlags::COMPUTE,
            QueueType::Transfer => vk::QueueFlags::TRANSFER,
            QueueType::SparseBinding => vk::QueueFlags::SPARSE_BINDING,
        }
    }
}

#[derive(Resource, Debug)]
pub struct QueuesRouter {
    pub(crate) queue_family_to_types: Vec<vk::QueueFlags>,
    pub(crate) queue_type_to_index: [QueueRef; 4],
    pub(crate) queue_type_to_family: [u32; 4],
}

impl QueuesRouter {
    pub fn of_type(&self, ty: QueueType) -> QueueRef {
        self.queue_type_to_index[ty as usize]
    }
    pub fn queue_family_of_type(&self, ty: QueueType) -> u32 {
        let queue_family = self.queue_type_to_family[ty as usize];
        queue_family
    }
    pub(crate) fn find_with_queue_family_properties(
        available_queue_family: &[vk::QueueFamilyProperties],
    ) -> Self {
        // Must include GRAPHICS. Prefer not COMPUTE or SPARSE_BINDING.
        let graphics_queue_family = available_queue_family
            .iter()
            .enumerate()
            .rev()
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
        // Must include COMPUTE. Prefer not GRAPHICS or SPARSE_BINDING
        let compute_queue_family = available_queue_family
            .iter()
            .enumerate()
            .rev()
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
                if family.queue_flags.contains(vk::QueueFlags::OPTICAL_FLOW_NV) {
                    priority -= 10;
                }
                if family
                    .queue_flags
                    .contains(vk::QueueFlags::VIDEO_DECODE_KHR)
                {
                    priority -= 10;
                }
                if family
                    .queue_flags
                    .contains(vk::QueueFlags::VIDEO_ENCODE_KHR)
                {
                    priority -= 10;
                }
                priority
            })
            .unwrap()
            .0 as u32;
        // Prefer TRANSFER, COMPUTE, then GRAPHICS.
        let transfer_queue_family = available_queue_family
            .iter()
            .enumerate()
            .rev()
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
                if family.queue_flags.contains(vk::QueueFlags::OPTICAL_FLOW_NV) {
                    priority -= 10;
                }
                if family
                    .queue_flags
                    .contains(vk::QueueFlags::VIDEO_DECODE_KHR)
                {
                    priority -= 10;
                }
                if family
                    .queue_flags
                    .contains(vk::QueueFlags::VIDEO_ENCODE_KHR)
                {
                    priority -= 10;
                }
                priority
            })
            .unwrap()
            .0 as u32;
        let sparse_binding_queue_family = available_queue_family
            .iter()
            .enumerate()
            .rev()
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
                if family.queue_flags.contains(vk::QueueFlags::OPTICAL_FLOW_NV) {
                    priority -= 10;
                }
                if family
                    .queue_flags
                    .contains(vk::QueueFlags::VIDEO_DECODE_KHR)
                {
                    priority -= 10;
                }
                if family
                    .queue_flags
                    .contains(vk::QueueFlags::VIDEO_ENCODE_KHR)
                {
                    priority -= 10;
                }
                priority
            })
            .map(|(i, _)| i as u32)
            .unwrap_or(u32::MAX);
        tracing::info!(graphics = %graphics_queue_family, compute = %compute_queue_family, transfer = %transfer_queue_family, sparse_binding = %sparse_binding_queue_family, "Queue families");

        let mut queue_family_to_types: Vec<vk::QueueFlags> =
            vec![vk::QueueFlags::empty(); available_queue_family.len()];
        if sparse_binding_queue_family != u32::MAX {
            queue_family_to_types[sparse_binding_queue_family as usize] |=
                vk::QueueFlags::SPARSE_BINDING;
        }
        queue_family_to_types[transfer_queue_family as usize] |= vk::QueueFlags::TRANSFER;
        queue_family_to_types[compute_queue_family as usize] |= vk::QueueFlags::COMPUTE;
        queue_family_to_types[graphics_queue_family as usize] |= vk::QueueFlags::GRAPHICS;

        let queue_type_to_family: [u32; 4] = [
            graphics_queue_family,
            compute_queue_family,
            transfer_queue_family,
            sparse_binding_queue_family,
        ];

        let mut queue_type_to_index: [QueueRef; 4] = [QueueRef::null(); 4];
        for (i, ty) in queue_family_to_types
            .iter()
            .filter(|x| !x.is_empty())
            .enumerate()
        {
            if ty.contains(vk::QueueFlags::GRAPHICS) {
                queue_type_to_index[QueueType::Graphics as usize] = QueueRef {
                    index: i as u32,
                    family: graphics_queue_family,
                };
            }
            if ty.contains(vk::QueueFlags::COMPUTE) {
                queue_type_to_index[QueueType::Compute as usize] = QueueRef {
                    index: i as u32,
                    family: compute_queue_family,
                };
            }
            if ty.contains(vk::QueueFlags::TRANSFER) {
                queue_type_to_index[QueueType::Transfer as usize] = QueueRef {
                    index: i as u32,
                    family: transfer_queue_family,
                };
            }
            if ty.contains(vk::QueueFlags::SPARSE_BINDING) {
                queue_type_to_index[QueueType::SparseBinding as usize] = QueueRef {
                    index: i as u32,
                    family: sparse_binding_queue_family,
                };
            }
        }
        for i in queue_type_to_index.iter().take(3) {
            assert!(!i.is_null(), "All queue types should've been assigned")
        }

        Self {
            queue_family_to_types,
            queue_type_to_index,
            queue_type_to_family,
        }
    }

    pub(crate) fn create_infos(&self) -> Vec<vk::DeviceQueueCreateInfo> {
        let mut infos: Vec<vk::DeviceQueueCreateInfo> =
            Vec::with_capacity(self.queue_family_to_types.len());
        for (family, flags) in self.queue_family_to_types.iter().enumerate() {
            if flags.is_empty() {
                continue;
            }
            let priority: &'static f32 = if flags.contains(vk::QueueFlags::GRAPHICS) {
                &PRIORITY_HIGH
            } else if flags.contains(vk::QueueFlags::COMPUTE) {
                &PRIORITY_MEDIUM
            } else {
                &PRIORITY_LOW
            };
            infos.push(vk::DeviceQueueCreateInfo {
                queue_family_index: family as u32,
                queue_count: 1,
                p_queue_priorities: priority,
                ..Default::default()
            });
        }
        infos
    }
}

const PRIORITY_HIGH: f32 = 1.0;
const PRIORITY_MEDIUM: f32 = 0.5;
const PRIORITY_LOW: f32 = 0.0;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_nvidia_windows() {
        let default = vk::QueueFamilyProperties {
            queue_flags: vk::QueueFlags::empty(),
            queue_count: 16,
            timestamp_valid_bits: 64,
            min_image_transfer_granularity: vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
        };
        let router = QueuesRouter::find_with_queue_family_properties(&[
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::GRAPHICS
                    | vk::QueueFlags::COMPUTE
                    | vk::QueueFlags::TRANSFER
                    | vk::QueueFlags::SPARSE_BINDING,
                queue_count: 16,
                ..default
            },
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING,
                queue_count: 2,
                ..default
            },
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::TRANSFER
                    | vk::QueueFlags::COMPUTE
                    | vk::QueueFlags::SPARSE_BINDING,
                queue_count: 8,
                ..default
            },
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::TRANSFER
                    | vk::QueueFlags::VIDEO_DECODE_KHR
                    | vk::QueueFlags::SPARSE_BINDING,
                queue_count: 8,
                timestamp_valid_bits: 32,
                ..default
            },
        ]);
        assert_eq!(
            router.queue_family_to_types,
            vec![
                vk::QueueFlags::GRAPHICS,
                vk::QueueFlags::TRANSFER | vk::QueueFlags::SPARSE_BINDING,
                vk::QueueFlags::COMPUTE,
                vk::QueueFlags::empty(),
            ]
        );
        for (i, expected_index) in [0, 2, 1, 1].iter().enumerate() {
            assert_eq!(router.queue_type_to_index[i].index, *expected_index);
        }
        assert_eq!(router.queue_type_to_family, [0, 2, 1, 1,]);
    }
    #[test]
    fn test_intel_windows() {
        let default = vk::QueueFamilyProperties {
            queue_flags: vk::QueueFlags::empty(),
            queue_count: 16,
            timestamp_valid_bits: 36,
            min_image_transfer_granularity: vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
        };
        let router = QueuesRouter::find_with_queue_family_properties(&[
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::GRAPHICS
                    | vk::QueueFlags::COMPUTE
                    | vk::QueueFlags::TRANSFER
                    | vk::QueueFlags::SPARSE_BINDING,
                queue_count: 1,
                ..default
            },
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::VIDEO_DECODE_KHR,
                queue_count: 2,
                ..default
            },
        ]);
        assert_eq!(
            router.queue_family_to_types,
            vec![
                vk::QueueFlags::GRAPHICS
                    | vk::QueueFlags::COMPUTE
                    | vk::QueueFlags::SPARSE_BINDING
                    | vk::QueueFlags::TRANSFER,
                vk::QueueFlags::empty(),
            ]
        );
        for (i, expected_index) in [0, 0, 0, 0].iter().enumerate() {
            assert_eq!(router.queue_type_to_index[i].index, *expected_index);
        }
        assert_eq!(router.queue_type_to_family, [0, 0, 0, 0,]);
    }

    #[test]
    fn test_single_queue_no_transfer_bit() {
        let default = vk::QueueFamilyProperties {
            queue_flags: vk::QueueFlags::empty(),
            queue_count: 3,
            timestamp_valid_bits: 48,
            min_image_transfer_granularity: vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
        };
        let router =
            QueuesRouter::find_with_queue_family_properties(&[vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::GRAPHICS
                    | vk::QueueFlags::COMPUTE
                    | vk::QueueFlags::SPARSE_BINDING,
                queue_count: 3,
                ..default
            }]);
        assert_eq!(
            router.queue_family_to_types,
            vec![
                vk::QueueFlags::GRAPHICS
                    | vk::QueueFlags::COMPUTE
                    | vk::QueueFlags::SPARSE_BINDING
                    | vk::QueueFlags::TRANSFER,
            ]
        );
        for (i, expected_index) in [0, 0, 0, 0].iter().enumerate() {
            assert_eq!(router.queue_type_to_index[i].index, *expected_index);
        }
        assert_eq!(router.queue_type_to_family, [0, 0, 0, 0,]);
    }
}
