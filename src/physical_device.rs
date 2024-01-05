use crate::Version;

use super::{Device, Instance};
use ash::{prelude::VkResult, vk};
use bevy_ecs::system::Resource;
use core::ffi::{c_char, c_void};
use std::{
    any::{Any, TypeId},
    collections::BTreeMap,
    ffi::CStr,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::{Arc, RwLock, RwLockReadGuard},
};

#[derive(Clone, Resource)]
pub struct PhysicalDevice {
    instance: Instance,
    physical_device: vk::PhysicalDevice,
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

impl Instance {
    pub fn enumerate_physical_devices<'a>(
        &'a self,
    ) -> VkResult<impl ExactSizeIterator<Item = PhysicalDevice> + 'a> {
        let pdevices = unsafe { self.deref().enumerate_physical_devices().unwrap() };
        Ok(pdevices.into_iter().map(|pdevice| PhysicalDevice {
            instance: self.clone(),
            physical_device: pdevice,
        }))
    }
}

impl PhysicalDevice {
    pub fn instance(&self) -> &Instance {
        &self.instance
    }
    pub fn raw(&self) -> vk::PhysicalDevice {
        self.physical_device
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
            this.pdevice
                .instance()
                .get_physical_device_properties2(this.pdevice.raw(), &mut wrapper);
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

#[derive(Resource)]
pub struct PhysicalDeviceProperties {
    pdevice: PhysicalDevice,
    inner: vk::PhysicalDeviceProperties,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub memory_model: PhysicalDeviceMemoryModel,
    properties: RwLock<BTreeMap<TypeId, Box<dyn Any>>>,
}
unsafe impl Send for PhysicalDeviceProperties {}
unsafe impl Sync for PhysicalDeviceProperties {}
impl PhysicalDeviceProperties {
    pub fn new(pdevice: PhysicalDevice) -> Self {
        let memory_properties = unsafe {
            pdevice
                .instance()
                .get_physical_device_memory_properties(pdevice.raw())
        };
        let pdevice_properties = unsafe {
            pdevice
                .instance()
                .get_physical_device_properties(pdevice.raw())
        };
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
            pdevice,
            properties: Default::default(),
            memory_model,
            memory_properties,
            inner: pdevice_properties,
        }
    }
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



#[derive(Default)]
struct FeatureMap {
    physical_device_features: vk::PhysicalDeviceFeatures,
    features: BTreeMap<TypeId, Box<dyn Feature>>,
}
unsafe impl Send for FeatureMap {}
unsafe impl Sync for FeatureMap {}

#[derive(Resource)]
pub enum PhysicalDeviceFeatures {
    Setup {
        physical_device: PhysicalDevice,
        available_features: FeatureMap,
        enabled_features: FeatureMap,
    },
    Finalized {
        physical_device_features: vk::PhysicalDeviceFeatures2,
        enabled_features: BTreeMap<TypeId, Box<dyn Feature>>,
    },
}
unsafe impl Send for PhysicalDeviceFeatures {}
unsafe impl Sync for PhysicalDeviceFeatures {}


pub unsafe trait Feature: 'static {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn p_next(&mut self) -> &mut *mut c_void;
}

impl PhysicalDeviceFeatures {
    pub fn enable_feature<T: Feature + Default>(
        &mut self,
        mut selector: impl FnMut(&mut T) -> &mut vk::Bool32,
    ) -> Option<()> {
        let PhysicalDeviceFeatures::Setup {
            physical_device,
            available_features,
            enabled_features,
        } = self
        else {
            panic!("Cannot enable features outside plugin build phase");
        };
        let feature: &mut Box<dyn Feature> = available_features
            .features
            .entry(TypeId::of::<T>())
            .or_insert_with(|| {
                let mut feature = T::default();
                let mut wrapper = vk::PhysicalDeviceFeatures2::default();
                wrapper.p_next = &mut feature as *mut T as *mut std::ffi::c_void;
                unsafe {
                    physical_device
                        .instance()
                        .get_physical_device_features2(physical_device.raw(), &mut wrapper);
                };
                Box::new(feature)
            });
        let feature = feature.deref_mut().as_any_mut().downcast_mut::<T>().unwrap();
        let feature_available: vk::Bool32 = *selector(feature);
        if feature_available == vk::FALSE {
            // feature unavailable
            return None;
        }

        let enabled_features = enabled_features
            .features
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(T::default()));
        let enabled_features = enabled_features.deref_mut().as_any_mut().downcast_mut::<T>().unwrap();
        let feature_to_enable = selector(enabled_features);
        *feature_to_enable = vk::TRUE;
        Some(())
    }

    pub fn enabled_features<T: Feature + Default + 'static>(&self) -> Option<&T> {
        let enabled_features = match self {
            Self::Setup {
                enabled_features, ..
            } => &enabled_features.features,
            Self::Finalized {
                enabled_features, ..
            } => enabled_features,
        };
        let enabled_features = enabled_features.get(&TypeId::of::<T>())?;
        let enabled_features = enabled_features.deref().as_any().downcast_ref::<T>().unwrap();
        Some(enabled_features)
    }

    pub(crate) fn new(pdevice: PhysicalDevice) -> Self {
        let available_features = FeatureMap {
            physical_device_features: unsafe {
                pdevice.instance().get_physical_device_features(pdevice.raw())
            },
            features: Default::default(),
        };
        Self::Setup {
            physical_device: pdevice,
            available_features,
            enabled_features: Default::default(),
        }
    }

    pub(crate) fn finalize(&mut self) {
        if let Self::Setup {
            enabled_features, ..
        } = self
        {
            let FeatureMap{ physical_device_features, mut features } = std::mem::take(enabled_features);
            let mut physical_device_features = vk::PhysicalDeviceFeatures2 {
                features: physical_device_features,
                ..Default::default()
            };
            {
                // build p_next chain
                let mut last = &mut physical_device_features.p_next;
                for f in features.values_mut() {
                    *last = f.as_mut() as *mut _ as *mut c_void;
                    last = f.p_next();
                }
            }
            *self = Self::Finalized { enabled_features: features, physical_device_features }
        }
    }

    /// Returns a valid [`vk::PhysicalDeviceFeatures2`] for use in the p_next chain of [`vk::DeviceCreateInfo`].
    pub(crate) fn pdevice_features2(&self) -> &vk::PhysicalDeviceFeatures2 {
        if let Self::Finalized {
            physical_device_features, ..
        } = self
        {
            physical_device_features
        } else {
            panic!("Must be called after finalization")
        }
    }
}
