use std::{fmt::Debug, mem::ManuallyDrop, sync::{Arc, atomic::AtomicU64}, thread::JoinHandle};

use ash::vk;
use bevy_ecs::system::Resource;

use crate::Device;

#[derive(Resource)]
pub(crate) struct BinarySemaphorePool {
    device: Device,
    recycler: crossbeam_channel::Receiver<vk::Semaphore>,
    sender: crossbeam_channel::Sender<vk::Semaphore>
}


pub struct BinarySemaphore {
    semaphore: vk::Semaphore,
    recycler: crossbeam_channel::Sender<vk::Semaphore>
}
impl BinarySemaphore {
    pub fn raw(&self) -> vk::Semaphore {
        self.semaphore
    }
}
impl Debug for BinarySemaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("BinarySemaphore")
            .field(&self.semaphore)
            .finish()
    }
}

impl Drop for BinarySemaphore {
    fn drop(&mut self) {
        self.recycler.send(self.semaphore).unwrap();
    }
}


impl BinarySemaphorePool {
    pub fn new(device: Device) -> Self {
        let (sender, recycler) = crossbeam_channel::unbounded();
        Self {
            device,
            recycler,
            sender
        }
    }
    pub fn create(&self) -> BinarySemaphore {
        match self.recycler.try_recv() {
            Ok(semaphore) => return BinarySemaphore {
                semaphore,
                recycler: self.sender.clone()
            },
            Err(crossbeam_channel::TryRecvError::Empty) => {
                let semaphore = unsafe {
                    let mut type_info = vk::SemaphoreTypeCreateInfo::builder()
                        .semaphore_type(vk::SemaphoreType::TIMELINE)
                        .build();
                    let info = vk::SemaphoreCreateInfo::builder()
                        .push_next(&mut type_info)
                        .build();
                    self.device.create_semaphore(&info, None)
                }.unwrap();
                return BinarySemaphore {
                    semaphore,
                    recycler: self.sender.clone()
                }
            },
            Err(crossbeam_channel::TryRecvError::Disconnected) => unreachable!()
        }
    }
}

