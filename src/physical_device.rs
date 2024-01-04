use crate::Version;

use super::{Device, Instance};
use ash::{prelude::VkResult, vk};
use core::ffi::{c_char, c_void};
use std::{
    ffi::CStr,
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock, RwLockReadGuard}, collections::BTreeMap, any::{TypeId, Any}, ptr::NonNull,
};
pub struct PhysicalDevice {
    instance: Instance,
    physical_device: vk::PhysicalDevice,
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
    pub fn memory_model(&self) -> PhysicalDeviceMemoryModel {
        self.memory_model
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


pub unsafe trait PhysicalDeviceProperty: Sized + Default + 'static {
    fn get_new(this: &PhysicalDeviceProperties) -> Self {
        let mut wrapper = vk::PhysicalDeviceProperties2::default();
        let mut item = Self::default();
        unsafe {
            wrapper.p_next = &mut item as *mut Self as *mut c_void;
            this.pdevice.instance().get_physical_device_properties2(this.pdevice.raw(), &mut wrapper);
        }
        item
    }
    fn get(this: &PhysicalDeviceProperties) -> &Self {
        let properties = this.properties.read().unwrap();
        if let Some(entry) = properties.get(&TypeId::of::<Self>()) {
            let item = entry.deref().downcast_ref::<Self>().unwrap();
            let item: NonNull<Self> = item.into();
            unsafe {
                // This is ok because entry is boxed and never removed as long as self is still alive.
                return item.as_ref();
            }
        }
        drop(properties);
        let item = Self::get_new(this);
        let item: Box<dyn Any> = Box::new(item);
        let item_ptr = item.downcast_ref::<Self>().unwrap();
        let item_ptr: NonNull<Self> = item_ptr.into();

        let mut properties = this.properties.write().unwrap();
        properties.insert(TypeId::of::<Self>(), item);
        drop(properties);

        unsafe {
            // This is ok because entry is boxed and never removed as long as self is still alive.
            return item_ptr.as_ref();
        }
    }
}

pub struct PhysicalDeviceProperties {
    pdevice: PhysicalDevice,
    inner: vk::PhysicalDeviceProperties,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    properties: RwLock<BTreeMap<TypeId, Box<dyn Any>>>,
}
unsafe impl Send for PhysicalDeviceProperties {}
unsafe impl Sync for PhysicalDeviceProperties {}
impl PhysicalDeviceProperties {
    pub fn properties<T: PhysicalDeviceProperty + Default + 'static>(&self) -> &T {
        T::get(self)
    }
    pub fn device_name(&self) -> &CStr {
        unsafe {
            CStr::from_bytes_until_nul(std::slice::from_raw_parts(
                self.inner.device_name.as_ptr() as *const _,
                self.inner.device_name.len(),
            ))
            .unwrap()
        }
    }
    pub fn api_version(&self) -> Version {
        Version(self.inner.api_version)
    }
    pub fn driver_version(&self) -> Version {
        Version(self.inner.driver_version)
    }
    pub fn memory_types(&self) -> &[vk::MemoryType] {
        &self.memory_properties.memory_types[0..self.memory_properties.memory_type_count as usize]
    }
    pub fn memory_heaps(&self) -> &[vk::MemoryHeap] {
        &self.memory_properties.memory_heaps[0..self.memory_properties.memory_heap_count as usize]
    }
}
impl Deref for PhysicalDeviceProperties {
    type Target = vk::PhysicalDeviceProperties;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

unsafe impl PhysicalDeviceProperty for vk::PhysicalDeviceProperties {
    fn get(this: &PhysicalDeviceProperties) -> &Self {
        &this.inner
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
