use crate::{utils::VkTaggedObject, Version};

use super::Instance;
use ash::{
    extensions::khr,
    prelude::VkResult,
    vk::{self, ExtendsPhysicalDeviceFeatures2, ExtendsPhysicalDeviceProperties2, TaggedStructure},
};
use bevy::ecs::system::Resource;
use core::ffi::c_void;
use std::{
    any::{Any, TypeId},
    collections::BTreeMap,
    ffi::CStr,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::{Arc, RwLock},
};

#[derive(Clone, Resource)]
pub struct PhysicalDevice(Arc<PhysicalDeviceInner>);

struct PhysicalDeviceInner {
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    properties: PhysicalDeviceProperties,
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
impl PhysicalDeviceMemoryModel {
    pub fn storage_buffer_should_use_staging(&self) -> bool {
        matches!(self, Self::Bar | Self::Discrete)
    }
    pub fn uniform_buffer_should_use_staging(&self) -> bool {
        matches!(self, Self::Discrete)
    }
}

impl Instance {
    pub fn enumerate_physical_devices<'a>(
        &'a self,
    ) -> VkResult<impl ExactSizeIterator<Item = PhysicalDevice> + 'a> {
        let pdevices = unsafe { self.deref().enumerate_physical_devices().unwrap() };
        Ok(pdevices.into_iter().map(|pdevice| {
            let properties = PhysicalDeviceProperties::new(self.clone(), pdevice);
            PhysicalDevice(Arc::new(PhysicalDeviceInner {
                instance: self.clone(),
                physical_device: pdevice,
                properties,
            }))
        }))
    }
}

impl PhysicalDevice {
    pub fn instance(&self) -> &Instance {
        &self.0.instance
    }
    pub fn raw(&self) -> vk::PhysicalDevice {
        self.0.physical_device
    }
    pub fn image_format_properties(
        &self,
        format_info: &vk::PhysicalDeviceImageFormatInfo2,
    ) -> VkResult<Option<vk::ImageFormatProperties2>> {
        let mut out = vk::ImageFormatProperties2::default();
        unsafe {
            match self
                .0
                .instance
                .get_physical_device_image_format_properties2(
                    self.0.physical_device,
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
            self.0
                .instance
                .get_physical_device_queue_family_properties(self.0.physical_device)
        }
    }
    pub fn properties(&self) -> &PhysicalDeviceProperties {
        &self.0.properties
    }
}

pub unsafe trait PhysicalDeviceProperty:
    Sized + Default + 'static + TaggedStructure + ExtendsPhysicalDeviceProperties2
{
}
unsafe impl<T> PhysicalDeviceProperty for T where
    T: ExtendsPhysicalDeviceProperties2 + TaggedStructure + Default + Sized + 'static
{
}
pub struct PhysicalDeviceProperties {
    instance: Instance,
    pdevice: vk::PhysicalDevice,
    inner: vk::PhysicalDeviceProperties,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub memory_model: PhysicalDeviceMemoryModel,
    properties: RwLock<BTreeMap<vk::StructureType, Box<VkTaggedObject>>>,
}
unsafe impl Send for PhysicalDeviceProperties {}
unsafe impl Sync for PhysicalDeviceProperties {}
impl PhysicalDeviceProperties {
    fn new(instance: Instance, pdevice: vk::PhysicalDevice) -> Self {
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(pdevice) };
        let pdevice_properties = unsafe { instance.get_physical_device_properties(pdevice) };
        let types =
            &memory_properties.memory_types[0..memory_properties.memory_type_count as usize];
        let heaps =
            &memory_properties.memory_heaps[0..memory_properties.memory_heap_count as usize];

        let bar_heap = types
            .iter()
            .find(|ty| {
                ty.property_flags.contains(
                    vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
                ) && heaps[ty.heap_index as usize]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            })
            .map(|a| &heaps[a.heap_index as usize]);
        let memory_model =
            if pdevice_properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
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
        Self {
            instance,
            pdevice,
            properties: Default::default(),
            memory_model,
            memory_properties,
            inner: pdevice_properties,
        }
    }
    pub fn get<T: PhysicalDeviceProperty + Default + 'static>(&self) -> &T {
        let properties = self.properties.read().unwrap();
        if let Some(entry) = properties.get(&T::STRUCTURE_TYPE) {
            let item = entry.deref().downcast_ref::<T>().unwrap();
            let item: NonNull<T> = item.into();
            unsafe {
                // This is ok because entry is boxed and never removed as long as self is still alive.
                return item.as_ref();
            }
        }
        drop(properties);

        let mut wrapper = vk::PhysicalDeviceProperties2::default();
        let mut item = T::default();
        unsafe {
            wrapper.p_next = &mut item as *mut T as *mut c_void;
            self.instance
                .get_physical_device_properties2(self.pdevice, &mut wrapper);
        }
        let item = VkTaggedObject::new(item);
        let item_ptr = item.downcast_ref::<T>().unwrap();
        let item_ptr: NonNull<T> = item_ptr.into();

        let mut properties = self.properties.write().unwrap();
        properties.insert(T::STRUCTURE_TYPE, item);
        drop(properties);

        unsafe {
            // This is ok because entry is boxed and never removed as long as self is still alive.
            return item_ptr.as_ref();
        }
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
    /// Returns the maximum supported API version for this physical device.
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

#[derive(Default)]
pub struct FeatureMap {
    physical_device_features: vk::PhysicalDeviceFeatures,
    features: BTreeMap<vk::StructureType, Box<VkTaggedObject>>,
}

impl FeatureMap {
    pub fn get<T: Feature + Default + 'static>(&self) -> Option<&T> {
        let enabled_features = self.features.get(&T::STRUCTURE_TYPE)?;
        let enabled_features = enabled_features.deref().downcast_ref::<T>().unwrap();
        Some(enabled_features)
    }
}

unsafe impl Send for FeatureMap {}
unsafe impl Sync for FeatureMap {}

impl FeatureMap {
    pub fn as_physical_device_features(&mut self) -> vk::PhysicalDeviceFeatures2 {
        let mut features = vk::PhysicalDeviceFeatures2 {
            features: self.physical_device_features,
            ..Default::default()
        };
        // build p_next chain
        let mut last = &mut features.p_next;
        for f in self.features.values_mut() {
            *last = f.as_mut() as *mut _ as *mut c_void;
            last = &mut f.p_next;
        }
        features
    }
}

#[derive(Resource)]
pub struct PhysicalDeviceFeaturesSetup {
    physical_device: PhysicalDevice,
    available_features: FeatureMap,
    enabled_features: FeatureMap,
    vulkan_version: Version,
}

unsafe impl Send for PhysicalDeviceFeaturesSetup {}
unsafe impl Sync for PhysicalDeviceFeaturesSetup {}

pub unsafe trait Feature: TaggedStructure + 'static {
    type Extension;
    const REQUIRED_VK_VERSION: Version;
}

impl PhysicalDeviceFeaturesSetup {
    pub fn enable_feature<T: Feature + Default>(
        &mut self,
        mut selector: impl FnMut(&mut T) -> &mut vk::Bool32,
    ) -> Option<()> {
        if self.vulkan_version < T::REQUIRED_VK_VERSION {
            // feature requires a higher Vulkan version
            return None;
        }
        let feature: &mut Box<VkTaggedObject> = self
            .available_features
            .features
            .entry(T::STRUCTURE_TYPE)
            .or_insert_with(|| {
                let mut feature = T::default();
                let mut wrapper = vk::PhysicalDeviceFeatures2::default();
                wrapper.p_next = &mut feature as *mut T as *mut std::ffi::c_void;
                unsafe {
                    self.physical_device
                        .instance()
                        .get_physical_device_features2(self.physical_device.raw(), &mut wrapper);
                };
                VkTaggedObject::new(feature)
            });
        let feature = feature.deref_mut().downcast_mut::<T>().unwrap();
        let feature_available: vk::Bool32 = *selector(feature);
        if feature_available == vk::FALSE {
            // feature unavailable
            return None;
        }

        let enabled_features = self
            .enabled_features
            .features
            .entry(T::STRUCTURE_TYPE)
            .or_insert_with(|| VkTaggedObject::new(T::default()));
        let enabled_features = enabled_features.deref_mut().downcast_mut::<T>().unwrap();
        let feature_to_enable = selector(enabled_features);
        *feature_to_enable = vk::TRUE;
        Some(())
    }

    pub fn enabled_features<T: Feature + Default + 'static>(&self) -> Option<&T> {
        self.enabled_features.get::<T>()
    }

    pub(crate) fn new(pdevice: PhysicalDevice) -> Self {
        let available_features = FeatureMap {
            physical_device_features: unsafe {
                pdevice
                    .instance()
                    .get_physical_device_features(pdevice.raw())
            },
            features: Default::default(),
        };
        PhysicalDeviceFeaturesSetup {
            vulkan_version: pdevice.instance().api_version(),
            physical_device: pdevice,
            available_features,
            enabled_features: Default::default(),
        }
    }

    pub(crate) fn finalize(self) -> FeatureMap {
        self.enabled_features
    }
}

macro_rules! impl_feature {
    ($feature:ty, $ext:ty) => {
        unsafe impl Feature for $feature {
            type Extension = $ext;
            const REQUIRED_VK_VERSION: Version = Version::new(0, 1, 0, 0);
        }
    };
}
unsafe impl Feature for vk::PhysicalDeviceVulkan11Features {
    type Extension = ();
    const REQUIRED_VK_VERSION: Version = Version::new(0, 1, 1, 0);
}
unsafe impl Feature for vk::PhysicalDeviceVulkan12Features {
    type Extension = ();
    const REQUIRED_VK_VERSION: Version = Version::new(0, 1, 2, 0);
}
unsafe impl Feature for vk::PhysicalDeviceVulkan13Features {
    type Extension = ();
    const REQUIRED_VK_VERSION: Version = Version::new(0, 1, 3, 0);
}
impl_feature!(
    vk::PhysicalDeviceSynchronization2FeaturesKHR,
    khr::Synchronization2
);
impl_feature!(
    vk::PhysicalDeviceDynamicRenderingFeatures,
    khr::DynamicRendering
);
impl_feature!(
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
    khr::RayTracingPipeline
);
impl_feature!(
    vk::PhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT,
    khr::RayTracingPipeline
);
