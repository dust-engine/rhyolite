mod format;
mod future;
use ash::vk::{self, TaggedStructure};
pub use format::*;
pub use future::*;
use std::{ops::Deref, ptr::NonNull};

#[derive(Debug, Clone)]
pub enum SharingMode<T>
where
    T: Deref<Target = [u32]>,
{
    Exclusive,
    Concurrent { queue_family_indices: T },
}

impl<T: Deref<Target = [u32]>> SharingMode<T> {
    pub fn as_raw(&self) -> vk::SharingMode {
        match self {
            Self::Exclusive => vk::SharingMode::EXCLUSIVE,
            Self::Concurrent { .. } => vk::SharingMode::CONCURRENT,
        }
    }

    pub fn queue_family_indices(&self) -> &[u32] {
        match self {
            Self::Exclusive => &[],
            Self::Concurrent {
                queue_family_indices,
            } => &queue_family_indices.deref(),
        }
    }
}

#[repr(C)]
pub struct VkTaggedObject {
    pub s_type: vk::StructureType,
    pub p_next: *mut std::ffi::c_void,
    rest: [u8],
}
impl VkTaggedObject {
    pub fn from_ref<T: TaggedStructure>(obj: &T) -> &Self {
        let ptr = std::ptr::from_raw_parts::<Self>(
            obj as *const T as *const (),
            std::mem::size_of::<T>() - std::mem::size_of::<vk::BaseInStructure>(),
        );
        unsafe { &*ptr }
    }
    pub fn new<T: TaggedStructure>(obj: T) -> Box<Self> {
        unsafe {
            let boxed = Box::new(obj);
            let boxed_ptr = NonNull::new_unchecked(Box::into_raw(boxed));
            let fat_ptr = NonNull::<Self>::from_raw_parts(
                boxed_ptr.cast(),
                std::mem::size_of::<T>() - std::mem::size_of::<vk::BaseInStructure>(),
            );
            Box::from_raw(fat_ptr.as_ptr())
        }
    }
    pub fn downcast_ref<T: TaggedStructure>(&self) -> Option<&T> {
        if self.s_type == T::STRUCTURE_TYPE {
            Some(unsafe { &*(self as *const Self as *const T) })
        } else {
            None
        }
    }
    pub fn downcast_mut<T: TaggedStructure>(&mut self) -> Option<&mut T> {
        if self.s_type == T::STRUCTURE_TYPE {
            Some(unsafe { &mut *(self as *mut Self as *mut T) })
        } else {
            None
        }
    }
}
