use std::{
    any::{Any, TypeId},
    collections::BTreeMap,
    ops::{Deref, DerefMut},
    sync::Mutex,
};

use ash::vk;
use bevy_ecs::system::Resource;

use crate::{Device, PhysicalDevice};

#[derive(Default)]
struct FeatureMap {
    physical_device_features: vk::PhysicalDeviceFeatures2,
    features: BTreeMap<TypeId, Box<dyn Feature>>,
}
unsafe impl Send for FeatureMap {}
unsafe impl Sync for FeatureMap {}

#[derive(Resource)]
enum Features {
    Setup {
        physical_device: PhysicalDevice,
        available_features: Mutex<FeatureMap>,
        enabled_features: FeatureMap,
    },
    Finalized {
        available_features: FeatureMap,
        enabled_features: FeatureMap,
    },
}

pub trait Feature {
    fn as_mut_any(&mut self) -> &mut dyn Any;
    fn as_any(&self) -> &dyn Any;
}

impl Features {
    fn get_or_retrieve_available_features<'a, T: Feature + Default + 'static>(
        available_features: &'a mut FeatureMap,
        physical_device: &'a PhysicalDevice,
    ) -> &'a mut T {
        let feature: &mut Box<dyn Feature> = available_features
            .features
            .entry(TypeId::of::<T>())
            .or_insert_with(|| {
                let feature = T::default();
                unsafe {
                    physical_device.instance().get_physical_device_features2(
                        physical_device.raw(),
                        &mut available_features.physical_device_features,
                    );
                };
                Box::new(feature)
            });
        Box::as_mut(feature)
            .as_mut_any()
            .downcast_mut::<T>()
            .unwrap()
    }
    pub fn enable_feature<T: Feature + Default + 'static>(
        &mut self,
        mut selector: impl FnMut(&mut T) -> &mut vk::Bool32,
    ) -> Option<()> {
        let Features::Setup {
            physical_device,
            available_features,
            enabled_features,
        } = self
        else {
            panic!("Cannot enable features outside plugin build phase");
        };
        let mut available_features = available_features.lock().unwrap();
        let available_features = available_features.deref_mut();
        let feature: &mut T =
            Self::get_or_retrieve_available_features(available_features, physical_device);

        let feature_available: vk::Bool32 = *selector(feature);
        if feature_available == vk::FALSE {
            // feature unavailable
            return None;
        }

        let enabled_features = enabled_features
            .features
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(T::default()));
        let enabled_features = Box::as_mut(enabled_features)
            .as_mut_any()
            .downcast_mut::<T>()
            .unwrap();
        let feature_to_enable = selector(enabled_features);
        *feature_to_enable = vk::TRUE;
        Some(())
    }

    pub fn feature_is_enabled<T: Feature + Default + 'static>(
        &self,
        mut selector: impl FnMut(&T) -> &vk::Bool32,
    ) -> bool {
        let enabled_features = match self {
            Features::Setup {
                enabled_features, ..
            } => enabled_features,
            Features::Finalized {
                enabled_features, ..
            } => enabled_features,
        };
        let Some(enabled_features) = enabled_features.features.get(&TypeId::of::<T>()) else {
            return false;
        };
        let enabled_features = Box::as_ref(enabled_features)
            .as_any()
            .downcast_ref::<T>()
            .unwrap();
        let feature_enabled = selector(enabled_features);
        if *feature_enabled == vk::FALSE {
            return false;
        } else {
            return true;
        }
    }
    pub fn feature_is_available<T: Feature + Default + 'static>(
        &self,
        mut selector: impl FnMut(&T) -> &vk::Bool32,
    ) -> bool {
        let feature_available = match self {
            Features::Setup {
                available_features,
                physical_device,
                ..
            } => {
                let mut available_features = available_features.lock().unwrap();
                let available_features = available_features.deref_mut();
                let feature = Self::get_or_retrieve_available_features::<T>(
                    available_features,
                    physical_device,
                );
                let feature_available = selector(feature);
                *feature_available
            }
            Features::Finalized {
                available_features, ..
            } => {
                let Some(available_features) = available_features.features.get(&TypeId::of::<T>())
                else {
                    return false;
                };
                let feature = Box::as_ref(available_features)
                    .as_any()
                    .downcast_ref::<T>()
                    .unwrap();
                let feature_available = selector(feature);
                *feature_available
            }
        };
        if feature_available == vk::FALSE {
            return false;
        } else {
            return true;
        }
    }
}
