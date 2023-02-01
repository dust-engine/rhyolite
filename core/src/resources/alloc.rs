use ash::{prelude::VkResult, vk};
use std::{ops::Deref, sync::Arc};

use crate::{Device, MemoryHeap, MemoryType};

pub use vk::BufferUsageFlags;
pub use vk_mem::{Alloc, Allocation, AllocationCreateFlags, MemoryUsage};

pub struct Allocator {
    pub(crate) allocator: vk_mem::Allocator,
    device: Arc<Device>,
    memory_model: DeviceMemoryModel,
    pub(crate) heaps: Box<[MemoryHeap]>,
    pub(crate) types: Box<[MemoryType]>,
}
impl crate::HasDevice for Allocator {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

pub enum DeviceMemoryModel {
    Integrated,
    /// Discrete GPU without HOST_VISIBLE DEVICE_LOCAL memory
    Discrete,
    DiscreteBar,
    DiscreteReBar,
}

impl Allocator {
    pub fn new(device: Arc<Device>) -> Self {
        let mut allocator_flags: vk_mem::AllocatorCreateFlags =
            vk_mem::AllocatorCreateFlags::empty();
        if device
            .physical_device()
            .features()
            .v12
            .buffer_device_address
            == vk::TRUE
        {
            allocator_flags |= vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
        }

        let allocator = vk_mem::Allocator::new(
            vk_mem::AllocatorCreateInfo::new(
                device.instance().as_ref().deref(),
                device.as_ref().deref(),
                device.physical_device().raw(),
            )
            .vulkan_api_version(vk::make_api_version(0, 1, 3, 0))
            .flags(allocator_flags),
        )
        .unwrap();
        let (heaps, types) = device.physical_device().get_memory_properties();
        let memory_model = if device.physical_device().integrated() {
            DeviceMemoryModel::Integrated
        } else {
            let bar_heap = types
                .iter()
                .find(|ty| {
                    ty.property_flags.contains(
                        vk::MemoryPropertyFlags::DEVICE_LOCAL
                            | vk::MemoryPropertyFlags::HOST_VISIBLE,
                    ) && heaps[ty.heap_index as usize]
                        .flags
                        .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                })
                .map(|a| &heaps[a.heap_index as usize]);
            if let Some(bar_heap) = bar_heap {
                if bar_heap.size <= 256 * 1024 * 1024 {
                    // regular 256MB bar
                    DeviceMemoryModel::DiscreteBar
                } else {
                    DeviceMemoryModel::DiscreteReBar
                }
            } else {
                // Can't find a BAR heap
                DeviceMemoryModel::Discrete
            }
        };
        Self {
            allocator,
            device,
            heaps,
            types,
            memory_model,
        }
    }

    pub(crate) fn create_info_by_scenario(
        &self,
        flags: vk_mem::AllocationCreateFlags,
        scenario: &MemoryAllocScenario,
    ) -> vk_mem::AllocationCreateInfo {
        let mut required_flags = vk::MemoryPropertyFlags::empty();
        let mut preferred_flags = vk::MemoryPropertyFlags::empty();
        let mut non_preferred_flags = vk::MemoryPropertyFlags::empty();
        let mut memory_usage = vk_mem::MemoryUsage::Unknown;
        match scenario {
            MemoryAllocScenario::StagingBuffer => {
                required_flags |= vk::MemoryPropertyFlags::HOST_VISIBLE;
                non_preferred_flags |= vk::MemoryPropertyFlags::HOST_CACHED;
                match self.memory_model {
                    DeviceMemoryModel::Integrated => {
                        preferred_flags |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
                    }
                    DeviceMemoryModel::Discrete
                    | DeviceMemoryModel::DiscreteBar
                    | DeviceMemoryModel::DiscreteReBar => {
                        non_preferred_flags |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
                    }
                }
            }
            MemoryAllocScenario::DeviceAccess => {
                preferred_flags |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
                non_preferred_flags |= vk::MemoryPropertyFlags::HOST_VISIBLE;
            }
            MemoryAllocScenario::AssetBuffer => {
                preferred_flags |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
                non_preferred_flags |= vk::MemoryPropertyFlags::HOST_CACHED;
                match self.memory_model {
                    DeviceMemoryModel::Integrated => {
                        preferred_flags |= vk::MemoryPropertyFlags::HOST_VISIBLE;
                    }
                    DeviceMemoryModel::Discrete
                    | DeviceMemoryModel::DiscreteBar
                    | DeviceMemoryModel::DiscreteReBar => {
                        non_preferred_flags |= vk::MemoryPropertyFlags::HOST_VISIBLE;
                    }
                }
            }
            MemoryAllocScenario::DynamicUniform => {
                preferred_flags |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
                non_preferred_flags |= vk::MemoryPropertyFlags::HOST_CACHED;
                match self.memory_model {
                    DeviceMemoryModel::Integrated
                    | DeviceMemoryModel::DiscreteBar
                    | DeviceMemoryModel::DiscreteReBar => {
                        preferred_flags |= vk::MemoryPropertyFlags::HOST_VISIBLE;
                    }
                    _ => {}
                }
            }
            MemoryAllocScenario::DynamicStorage => {
                preferred_flags |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
                non_preferred_flags |= vk::MemoryPropertyFlags::HOST_CACHED;
                match self.memory_model {
                    DeviceMemoryModel::Integrated
                    | DeviceMemoryModel::DiscreteBar
                    | DeviceMemoryModel::DiscreteReBar => {
                        preferred_flags |= vk::MemoryPropertyFlags::HOST_VISIBLE;
                    }
                    _ => {}
                }
            }
            MemoryAllocScenario::Custom {
                memory_usage: memory_usage_self,
                require_flags: required_flags_self,
                preferred_flags: preferred_flags_self,
                non_preferred_flags: non_preferred_flags_self,
            } => {
                memory_usage = *memory_usage_self;
                required_flags = *required_flags_self;
                preferred_flags = *preferred_flags_self;
                non_preferred_flags = *non_preferred_flags_self;
            }
        }
        non_preferred_flags |= vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD;
        vk_mem::AllocationCreateInfo {
            flags,
            usage: memory_usage,
            required_flags,
            preferred_flags,
            ..Default::default()
        }
    }

    pub fn allocate_memory(
        &self,
        memory_requirements: &ash::vk::MemoryRequirements,
        create_info: &vk_mem::AllocationCreateInfo,
    ) -> VkResult<Allocation> {
        let allocation = unsafe {
            self.allocator
                .allocate_memory(memory_requirements, create_info)
        }?;
        Ok(allocation)
    }
}

#[derive(Clone)]
pub enum MemoryAllocScenario {
    /// On integrated GPU, allocate buffer on DEVICE_LOCAL, HOST_VISIBLE non-cached memory.
    /// If no such memory exist, allocate buffer on HOST_VISIBLE non-cached memory.
    ///
    /// On discrete and SAM GPU, allocate buffer on HOST_VISIBLE, non-DEVICE_LOCAL memory.
    StagingBuffer,
    /// On all GPUs, allocate on DEVICE_LOCAL, non-HOST_VISIBLE memory.
    DeviceAccess,
    /// For uploading large assets. Always use staging buffer on discrete GPUs but prefer
    /// HOST_VISIBLE on integrated GPUs.
    AssetBuffer,
    /// For small uniform buffers that frequently gets updated
    /// On integrated GPU, allocate buffer on DEVICE_LOCAL, HOST_VISIBLE, HOST_CACHED memory.
    /// On discrete GPU, allocate buffer on BAR (DEVICE_LOCAL, HOST_VISIBLE, non-cached) if possible.
    /// Otherwise, explicit transfer is required and the buffer will be allocated on DEVICE_LOCAL.
    /// On discrete GPU with SAM, allocate buffer on BAR.
    DynamicUniform,
    /// For large storage buffers that frequently gets updated on certain parts.
    /// On integrated GPU, allocate buffer on DEVICE_LOCAL, HOST_VISIBLE, HOST_CACHED memory.
    /// On discrete GPU, allocate buffer on DEVICE_LOCAL, non host-visible memory.
    /// On discrete GPU with SAM, allocate buffer on BAR.
    DynamicStorage,
    Custom {
        memory_usage: vk_mem::MemoryUsage,
        require_flags: vk::MemoryPropertyFlags,
        preferred_flags: vk::MemoryPropertyFlags,
        non_preferred_flags: vk::MemoryPropertyFlags,
    },
}
#[derive(Clone)]
pub struct BufferRequest<'a> {
    pub size: u64,
    /// If this value is 0, the memory will be allocated based on the buffer requirements.
    /// The actual alignment used on the allocation is buffer_request.alignment.max(buffer_requirements.alignment).
    pub alignment: u64,
    pub usage: vk::BufferUsageFlags,
    pub scenario: MemoryAllocScenario,
    pub allocation_flags: AllocationCreateFlags,
    pub sharing_mode: vk::SharingMode,
    pub queue_families: &'a [u32],
}
impl<'a> Default for BufferRequest<'a> {
    fn default() -> Self {
        Self {
            size: 0,
            alignment: 0,
            usage: vk::BufferUsageFlags::empty(),
            scenario: MemoryAllocScenario::DeviceAccess,
            allocation_flags: vk_mem::AllocationCreateFlags::empty(),
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_families: &[],
        }
    }
}
