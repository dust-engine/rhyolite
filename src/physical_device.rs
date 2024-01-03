use crate::Version;

use super::{Device, Instance};
use ash::{prelude::VkResult, vk};
use core::ffi::{c_char, c_void};
use std::{
    ffi::CStr,
    ops::{Deref, DerefMut},
    sync::Arc,
};
pub struct PhysicalDevice {
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    properties: Box<PhysicalDeviceProperties>,
    memory_properties: Box<vk::PhysicalDeviceMemoryProperties>,
    memory_model: PhysicalDeviceMemoryModel,
}

#[derive(Debug, Clone, Copy)]
pub enum PhysicalDeviceMemoryModel {
    /// Completely separated VRAM and system RAM. No device-local memory is host visible.
    Discrete,
    /// Like `Discrete`, but 256MB of the VRAM is also host-visible. The CPU may map this
    /// memory into system memory range and `memcpy` into it directly.
    Bar,
    /// Discrete GPUs where all of its VRAM can be made host-visible.
    ReBar,
    /// Integrated GPUs with one large memory pool that is both device local and host-visible.
    Unified,
    /// Integrated GPUs with smaller device-local memory pool.
    /// Examples:
    /// [256MB DEVICE_LOCAL] [32GB HOST_VISIBLE] [256MB DEVICE_LOCAL|HOST_VISIBLE] (AMD APU)
    /// [256MB DEVICE_LOCAL] [32GB HOST_VISIBLE] (also AMD APU)
    BiasedUnified,
}

impl PhysicalDevice {
    pub fn instance(&self) -> &Instance {
        &self.instance
    }
    pub fn raw(&self) -> vk::PhysicalDevice {
        self.physical_device
    }
    pub fn enumerate(instance: &Instance) -> VkResult<Vec<Self>> {
        // Safety: No Host Syncronization rules for vkEnumeratePhysicalDevices.
        // It should be OK to call this method and obtain multiple copies of VkPhysicalDevice,
        // because nothing except vkDestroyInstance require exclusive access to VkPhysicalDevice.
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let results = physical_devices
            .into_iter()
            .map(|pdevice| {
                let properties = PhysicalDeviceProperties::new(instance, pdevice);
                let memory_properties =
                    unsafe { instance.get_physical_device_memory_properties(pdevice) };
                let types = &memory_properties.memory_types
                    [0..memory_properties.memory_type_count as usize];
                let heaps = &memory_properties.memory_heaps
                    [0..memory_properties.memory_heap_count as usize];

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
                let memory_model =
                    if properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
                        if let Some(bar_heap) = bar_heap {
                            if bar_heap.size <= 256 * 1024 * 1024 {
                                // regular 256MB bar
                                PhysicalDeviceMemoryModel::BiasedUnified
                            } else {
                                PhysicalDeviceMemoryModel::Unified
                            }
                        } else {
                            // Can't find a BAR heap
                            // Note: this case doesn't exist in real life.
                            // We select BiasedUnified based on the assumption that when requesting
                            // DEVICE_LOCAL | HOST_VISIBLE memory, the allocator will fallback to
                            // non-device-local memory.
                            PhysicalDeviceMemoryModel::BiasedUnified
                        }
                    } else {
                        if let Some(bar_heap) = bar_heap {
                            if bar_heap.size <= 256 * 1024 * 1024 {
                                // regular 256MB bar
                                PhysicalDeviceMemoryModel::Bar
                            } else {
                                PhysicalDeviceMemoryModel::ReBar
                            }
                        } else {
                            // Can't find a BAR heap
                            PhysicalDeviceMemoryModel::Discrete
                        }
                    };
                PhysicalDevice {
                    instance: instance.clone(), // Retain reference to Instance here
                    physical_device: pdevice,   // Borrow VkPhysicalDevice from Instance
                    // Borrow is safe because we retain a reference to Instance here,
                    // ensuring that Instance wouldn't be dropped as long as the borrow is still there.
                    properties,
                    memory_model,
                    memory_properties: Box::new(memory_properties),
                }
            })
            .collect();
        Ok(results)
    }
    pub fn properties(&self) -> &PhysicalDeviceProperties {
        &self.properties
    }
    pub fn memory_model(&self) -> PhysicalDeviceMemoryModel {
        self.memory_model
    }
    pub fn memory_types(&self) -> &[vk::MemoryType] {
        &self.memory_properties.memory_types[0..self.memory_properties.memory_type_count as usize]
    }
    pub fn memory_heaps(&self) -> &[vk::MemoryHeap] {
        &self.memory_properties.memory_heaps[0..self.memory_properties.memory_heap_count as usize]
    }
    pub fn image_format_properties(
        &self,
        format_info: &vk::PhysicalDeviceImageFormatInfo2,
    ) -> VkResult<Option<vk::ImageFormatProperties2>> {
        let mut out = vk::ImageFormatProperties2::default();
        unsafe {
            match self.instance.get_physical_device_image_format_properties2(
                self.physical_device,
                format_info,
                &mut out,
            ) {
                Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => Ok(None),
                Ok(_) => Ok(Some(out)),
                Err(_) => panic!(),
            }
        }
    }
    pub(crate) fn get_queue_family_properties(&self) -> Vec<vk::QueueFamilyProperties> {
        unsafe {
            self.instance
                .get_physical_device_queue_family_properties(self.physical_device)
        }
    }
}

pub struct PhysicalDeviceProperties {
    pub inner: vk::PhysicalDeviceProperties2,
    pub v11: vk::PhysicalDeviceVulkan11Properties,
    pub v12: vk::PhysicalDeviceVulkan12Properties,
    pub v13: vk::PhysicalDeviceVulkan13Properties,
    pub acceleration_structure: vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
    pub ray_tracing: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
}
unsafe impl Send for PhysicalDeviceProperties {}
unsafe impl Sync for PhysicalDeviceProperties {}
impl PhysicalDeviceProperties {
    fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Box<PhysicalDeviceProperties> {
        let mut this = Box::pin(Self {
            inner: vk::PhysicalDeviceProperties2::default(),
            v11: vk::PhysicalDeviceVulkan11Properties::default(),
            v12: vk::PhysicalDeviceVulkan12Properties::default(),
            v13: vk::PhysicalDeviceVulkan13Properties::default(),
            acceleration_structure: vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default(),
            ray_tracing: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default(),
        });
        this.inner.p_next = &mut this.v11 as *mut _ as *mut c_void;
        this.v11.p_next = &mut this.v12 as *mut _ as *mut c_void;
        this.v12.p_next = &mut this.v13 as *mut _ as *mut c_void;
        this.v13.p_next = &mut this.acceleration_structure as *mut _ as *mut c_void;
        this.acceleration_structure.p_next = &mut this.ray_tracing as *mut _ as *mut c_void;
        unsafe {
            instance.get_physical_device_properties2(physical_device, &mut this.inner);
        }
        std::pin::Pin::into_inner(this)
    }
    pub fn device_name(&self) -> &CStr {
        unsafe {
            CStr::from_bytes_until_nul(std::slice::from_raw_parts(
                self.inner.properties.device_name.as_ptr() as *const _,
                self.inner.properties.device_name.len(),
            ))
            .unwrap()
        }
    }
    pub fn api_version(&self) -> Version {
        Version(self.inner.properties.api_version)
    }
    pub fn driver_version(&self) -> Version {
        Version(self.inner.properties.driver_version)
    }
}
impl Deref for PhysicalDeviceProperties {
    type Target = vk::PhysicalDeviceProperties;
    fn deref(&self) -> &Self::Target {
        &self.inner.properties
    }
}
impl DerefMut for PhysicalDeviceProperties {
    fn deref_mut(&mut self) -> &mut vk::PhysicalDeviceProperties {
        &mut self.inner.properties
    }
}

pub struct MemoryType {
    pub property_flags: vk::MemoryPropertyFlags,
    pub heap_index: u32,
}

pub struct MemoryHeap {
    pub size: vk::DeviceSize,
    pub flags: vk::MemoryHeapFlags,
    pub budget: vk::DeviceSize,
    pub usage: vk::DeviceSize,
}
