use ash::vk;
use bevy_ecs::system::Resource;


/// Index of a created queue
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct QueueRef(pub u8);
impl QueueRef {
    pub fn null() -> Self {
        QueueRef(u8::MAX)
    }
    pub fn is_null(&self) -> bool {
        self.0 == u8::MAX
    }
}
impl Default for QueueRef {
    fn default() -> Self {
        Self::null()
    }
}


#[derive(Clone, Copy, Debug)]
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

#[derive(Resource)]
pub struct QueuesRouter {
    queue_family_to_types: Vec<vk::QueueFlags>,
    queue_type_to_index: [QueueRef; 4],
    queue_type_to_family: [u32; 4],
}

impl QueuesRouter {
    pub fn of_type(&self, ty: QueueType) -> QueueRef {
        self.queue_type_to_index[ty as usize]
    }
    fn find_with_queue_family_properties(
        available_queue_family: &[vk::QueueFamilyProperties],
    ) -> Self {
        // Must include GRAPHICS. Prefer not COMPUTE or SPARSE_BINDING.
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
                (priority, family.timestamp_valid_bits)
            })
            .unwrap()
            .0 as u32;
        // Must include COMPUTE. Prefer not GRAPHICS or SPARSE_BINDING
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
                (priority, family.timestamp_valid_bits)
            })
            .unwrap()
            .0 as u32;
        // Prefer TRANSFER, COMPUTE, then GRAPHICS.
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
                (priority, family.timestamp_valid_bits)
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
                (priority, family.timestamp_valid_bits)
            })
            .unwrap()
            .0 as u32;

        let mut queue_family_to_types: Vec<vk::QueueFlags> =
            vec![vk::QueueFlags::empty(); available_queue_family.len()];
        queue_family_to_types[sparse_binding_queue_family as usize] |= vk::QueueFlags::SPARSE_BINDING;
        queue_family_to_types[transfer_queue_family as usize] |= vk::QueueFlags::TRANSFER;
        queue_family_to_types[compute_queue_family as usize] |= vk::QueueFlags::COMPUTE;
        queue_family_to_types[graphics_queue_family as usize] |= vk::QueueFlags::GRAPHICS;

        let queue_type_to_family: [u32; 4] = [
            graphics_queue_family,
            compute_queue_family,
            transfer_queue_family,
            sparse_binding_queue_family,
        ];

        let mut queue_type_to_index: [QueueRef; 4] = [QueueRef(u8::MAX); 4];
        for (i, ty) in queue_family_to_types
            .iter()
            .enumerate()
        {
            if ty.contains(vk::QueueFlags::GRAPHICS) {
                queue_type_to_index[QueueType::Graphics as usize] = QueueRef(i as u8);
            }
            if ty.contains(vk::QueueFlags::COMPUTE) {
                queue_type_to_index[QueueType::Compute as usize] = QueueRef(i as u8);
            }
            if ty.contains(vk::QueueFlags::TRANSFER) {
                queue_type_to_index[QueueType::Transfer as usize] = QueueRef(i as u8);
            }
            if ty.contains(vk::QueueFlags::SPARSE_BINDING) {
                queue_type_to_index[QueueType::SparseBinding as usize] = QueueRef(i as u8);
            }
        }
        for i in queue_type_to_index.iter() {
            assert_ne!(i.0, u8::MAX, "All queue types should've been assigned")
        }

        Self {
            queue_family_to_types,
            queue_type_to_index,
            queue_type_to_family,
        }
    }
}
