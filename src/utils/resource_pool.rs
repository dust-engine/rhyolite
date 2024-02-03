use std::{
    fmt::Debug,
    mem::{ManuallyDrop, MaybeUninit},
    ops::{Deref, DerefMut},
    sync::{atomic::AtomicU64, Arc},
    thread::JoinHandle,
};

use ash::vk;
use bevy_ecs::system::Resource;

use crate::Device;
pub struct ResourcePool<T> {
    recycler: crossbeam_channel::Receiver<T>,
    sender: crossbeam_channel::Sender<T>,
}

pub struct ResourcePoolItem<T> {
    /// Always initialized until the moment before `ResourcePoolItem<T>` gets dropped.
    item: MaybeUninit<T>,
    recycler: crossbeam_channel::Sender<T>,
}
impl<T> Deref for ResourcePoolItem<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { self.item.assume_init_ref() }
    }
}
impl<T> DerefMut for ResourcePoolItem<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.item.assume_init_mut() }
    }
}

impl<T> Drop for ResourcePoolItem<T> {
    fn drop(&mut self) {
        let item = unsafe { self.item.assume_init_read() };
        self.recycler.send(item).unwrap();
    }
}

impl<T> ResourcePool<T> {
    pub fn new() -> Self {
        let (sender, recycler) = crossbeam_channel::unbounded();
        Self { recycler, sender }
    }
    pub fn create(&self, create_new: impl FnOnce() -> T) -> ResourcePoolItem<T> {
        match self.recycler.try_recv() {
            Ok(item) => {
                return ResourcePoolItem {
                    item: MaybeUninit::new(item),
                    recycler: self.sender.clone(),
                }
            }
            Err(crossbeam_channel::TryRecvError::Empty) => {
                return ResourcePoolItem {
                    item: MaybeUninit::new(create_new()),
                    recycler: self.sender.clone(),
                };
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => unreachable!(),
        }
    }
}
