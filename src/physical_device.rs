use crate::{queue, Queues};

use super::{Device, Instance};
use ash::{prelude::VkResult, vk};
use core::ffi::{c_char, c_void};
use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};
pub struct PhysicalDevice {
    instance: Arc<Instance>,
    physical_device: vk::PhysicalDevice,
    properties: Box<PhysicalDeviceProperties>,
    features: Box<PhysicalDeviceFeatures>,
}

pub struct DeviceCreateInfo<'a, F: Fn(u32) -> Vec<f32>> {
    pub enabled_layer_names: &'a [*const c_char],
    pub enabled_extension_names: &'a [*const c_char],
    pub enabled_features: vk::PhysicalDeviceFeatures2,
    pub queue_create_callback: F,
}
impl<'a, F: Fn(u32) -> Vec<f32>> DeviceCreateInfo<'a, F> {
    pub fn with_queue_create_callback(callback: F) -> Self {
        Self {
            enabled_layer_names: &[],
            enabled_extension_names: &[],
            enabled_features: Default::default(),
            queue_create_callback: callback
        }
    }
}

impl PhysicalDevice {
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }
    pub fn raw(&self) -> vk::PhysicalDevice {
        self.physical_device
    }
    pub fn enumerate(instance: &Arc<Instance>) -> VkResult<Vec<Self>> {
        // Safety: No Host Syncronization rules for vkEnumeratePhysicalDevices.
        // It should be OK to call this method and obtain multiple copies of VkPhysicalDevice,
        // because nothing except vkDestroyInstance require exclusive access to VkPhysicalDevice.
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let results = physical_devices
            .into_iter()
            .map(|pdevice| PhysicalDevice {
                instance: instance.clone(), // Retain reference to Instance here
                physical_device: pdevice,   // Borrow VkPhysicalDevice from Instance
                // Borrow is safe because we retain a reference to Instance here,
                // ensuring that Instance wouldn't be dropped as long as the borrow is still there.
                properties: PhysicalDeviceProperties::new(instance, pdevice),
                features: PhysicalDeviceFeatures::new(instance, pdevice),
            })
            .collect();
        Ok(results)
    }
    pub fn properties(&self) -> &PhysicalDeviceProperties {
        &self.properties
    }
    pub fn features(&self) -> &PhysicalDeviceFeatures {
        &self.features
    }
    pub fn integrated(&self) -> bool {
        self.properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
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
    pub fn create_device<'a>(
        self,
        infos: &'a DeviceCreateInfo<'a, impl Fn(u32) -> Vec<f32>>
    ) -> VkResult<(Arc<Device>, Queues)> {
        let mut num_queue_families: u32 = 0;
        unsafe {
            (self.instance.fp_v1_0().get_physical_device_queue_family_properties)(
                self.physical_device,
                &mut num_queue_families,
                std::ptr::null_mut(),
            );
        }
        assert!(num_queue_families > 0);
        let mut list_priorities = Vec::new();
        let queue_create_infos: Vec<_> = (0..num_queue_families).filter_map(|queue_family_index| {
            let priorities = (infos.queue_create_callback)(queue_family_index);
            if priorities.is_empty() {
                return None;
            }
            let queue_count = priorities.len() as u32;
            let p_queue_priorities = priorities.as_ptr();
            list_priorities.push(priorities);
            Some(vk::DeviceQueueCreateInfo {
                queue_family_index,
                queue_count,
                p_queue_priorities,
                ..Default::default()
            })
        }).collect();
        let create_info = vk::DeviceCreateInfo {
            p_next: &infos.enabled_features as *const vk::PhysicalDeviceFeatures2 as *const _,
            queue_create_info_count: queue_create_infos.len() as u32,
            p_queue_create_infos: queue_create_infos.as_ptr(),

            enabled_layer_count: infos.enabled_layer_names.len() as u32,
            pp_enabled_layer_names: infos.enabled_layer_names.as_ptr(),

            enabled_extension_count: infos.enabled_extension_names.len() as u32,
            pp_enabled_extension_names: infos.enabled_extension_names.as_ptr(),
            // No need to specify this pointer with vk::PhysicalDeviceFeatures2 in the p_next chain
            p_enabled_features: std::ptr::null(),
            ..Default::default()
        };

        // Safety: No Host Syncronization rules for VkCreateDevice.
        // Device retains a reference to Instance, ensuring that Instance is dropped later than Device.
        let device = unsafe {
            self.instance
                .create_device(self.physical_device, &create_info, None)?
        };
        let device = Arc::new(Device::new(self, device));
        drop(list_priorities);

        let queues = unsafe { Queues::new(&device,  num_queue_families, &queue_create_infos) };
        Ok((device, queues))
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

pub struct PhysicalDeviceFeatures {
    pub inner: vk::PhysicalDeviceFeatures2,
    pub v11: vk::PhysicalDeviceVulkan11Features,
    pub v12: vk::PhysicalDeviceVulkan12Features,
    pub v13: vk::PhysicalDeviceVulkan13Features,
    pub acceleration_structure: vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
    pub ray_tracing: vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
}
unsafe impl Send for PhysicalDeviceFeatures {}
unsafe impl Sync for PhysicalDeviceFeatures {}
impl PhysicalDeviceFeatures {
    fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Box<PhysicalDeviceFeatures> {
        let mut this = Box::pin(Self {
            inner: vk::PhysicalDeviceFeatures2::default(),
            v11: vk::PhysicalDeviceVulkan11Features::default(),
            v12: vk::PhysicalDeviceVulkan12Features::default(),
            v13: vk::PhysicalDeviceVulkan13Features::default(),
            acceleration_structure: vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default(),
            ray_tracing: vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default(),
        });
        this.inner.p_next = &mut this.v11 as *mut _ as *mut c_void;
        this.v11.p_next = &mut this.v12 as *mut _ as *mut c_void;
        this.v12.p_next = &mut this.v13 as *mut _ as *mut c_void;
        this.v13.p_next = &mut this.acceleration_structure as *mut _ as *mut c_void;
        this.acceleration_structure.p_next = &mut this.ray_tracing as *mut _ as *mut c_void;
        unsafe {
            instance.get_physical_device_features2(physical_device, &mut this.inner);
        }
        std::pin::Pin::into_inner(this)
    }
}
impl Deref for PhysicalDeviceFeatures {
    type Target = vk::PhysicalDeviceFeatures;
    fn deref(&self) -> &Self::Target {
        &self.inner.features
    }
}
impl DerefMut for PhysicalDeviceFeatures {
    fn deref_mut(&mut self) -> &mut vk::PhysicalDeviceFeatures {
        &mut self.inner.features
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

impl PhysicalDevice {
    pub fn get_memory_properties(&self) -> (Box<[MemoryHeap]>, Box<[MemoryType]>) {
        let mut out = vk::PhysicalDeviceMemoryProperties2::default();
        let mut budget_out = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        out.p_next = &mut budget_out as *mut _ as *mut c_void;
        unsafe {
            self.instance
                .get_physical_device_memory_properties2(self.physical_device, &mut out)
        }
        let heaps = out.memory_properties.memory_heaps
            [0..out.memory_properties.memory_heap_count as usize]
            .iter()
            .enumerate()
            .map(|(i, heap)| MemoryHeap {
                size: heap.size,
                flags: heap.flags,
                budget: budget_out.heap_budget[i],
                usage: budget_out.heap_usage[i],
            })
            .collect::<Vec<MemoryHeap>>()
            .into_boxed_slice();
        let tys = out.memory_properties.memory_types
            [0..out.memory_properties.memory_type_count as usize]
            .iter()
            .map(|ty| MemoryType {
                property_flags: ty.property_flags,
                heap_index: ty.heap_index,
            })
            .collect::<Vec<MemoryType>>()
            .into_boxed_slice();
        (heaps, tys)
    }
}
