use std::{sync::{Arc, atomic::AtomicU64}, thread::JoinHandle, mem::ManuallyDrop};

use ash::vk;
use bevy_ecs::system::Resource;

use crate::Device;

#[derive(Resource)]
pub(crate) struct BinarySemaphorePool {
    device: Device,
    recycler: crossbeam_channel::Receiver<vk::Semaphore>,
    sender: crossbeam_channel::Sender<vk::Semaphore>
}

pub(crate) struct BinarySemaphore {
    semaphore: vk::Semaphore,
    recycler: crossbeam_channel::Sender<vk::Semaphore>
}
impl Drop for BinarySemaphore {
    fn drop(&mut self) {
        self.recycler.send(self.semaphore).unwrap();
    }
}


impl BinarySemaphorePool {
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


pub(crate) struct TimelineSemaphorePool {
    sender: std::mem::ManuallyDrop<crossbeam_channel::Sender<(TimelineSemaphore, u64)>>,
    service_thread: Option<JoinHandle<()>>
}
impl Drop for TimelineSemaphorePool {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.sender);
        }
        self.service_thread.take().unwrap().join().unwrap();
    }
}

impl TimelineSemaphorePool {
    pub fn new(device: Device) -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded();
        let service_thread = std::thread::spawn(move || {
            let mut semaphores_to_wait: Vec<(TimelineSemaphore, u64)> = Vec::new();
            loop {
                if semaphores_to_wait.is_empty() {
                    // Block the thread until at least one semaphore is to be waited on
                    match receiver.recv() {
                        Ok(semaphore) => semaphores_to_wait.push(semaphore),
                        Err(_) => break
                    }
                }
                loop {
                    // Receive as much as we can
                    match receiver.try_recv() {
                        Ok(semaphore) => semaphores_to_wait.push(semaphore),
                        Err(crossbeam_channel::TryRecvError::Empty) => break,
                        Err(crossbeam_channel::TryRecvError::Disconnected) => return
                    }
                }
                semaphores_to_wait.sort_unstable_by_key(|item| (item.0.inner.semaphore, item.1));
                let result = unsafe {
                    let mut deduped_semaphores: Vec<_> = semaphores_to_wait.iter().map(|item| (item.0.inner.semaphore, item.1)).collect();
                    deduped_semaphores.dedup_by_key(|item| item.0);
                    let semaphores = semaphores_to_wait.iter().map(|(semaphore, _)| semaphore.inner.semaphore).collect::<Vec<_>>();
                    let values = semaphores_to_wait.iter().map(|(_, value)| *value).collect::<Vec<_>>();
                    // Wait on the semaphore
                    device.wait_semaphores(&vk::SemaphoreWaitInfo {
                        flags: vk::SemaphoreWaitFlags::ANY,
                        semaphore_count: semaphores.len() as u32,
                        p_semaphores: semaphores.as_ptr(),
                        p_values: values.as_ptr(),
                        ..Default::default()
                    }, 1000000) // 0.001s
                };
                match result {
                    Ok(_) => {
                        // query all the semaphores
                        let mut cached_semaphore_value = (vk::Semaphore::null(), 0);
                        semaphores_to_wait.retain(|(semaphore, dst_value)| {
                            let current_value = if cached_semaphore_value.0 == semaphore.inner.semaphore {
                                cached_semaphore_value.1
                            } else {
                                let current_value = unsafe {
                                    device.get_semaphore_counter_value(semaphore.inner.semaphore).unwrap()
                                };
                                semaphore.inner.current_value.store(current_value, std::sync::atomic::Ordering::Relaxed);
                                cached_semaphore_value = (semaphore.inner.semaphore, current_value);
                                current_value
                            };
                            current_value < *dst_value
                        });
                    },
                    Err(vk::Result::TIMEOUT) => continue,
                    Err(err) => panic!("{:?}", err)
                }
            }
        });
        
        let pool = TimelineSemaphorePool {
            sender: ManuallyDrop::new(sender),
            service_thread: Some(service_thread),
        };
        pool
    }
}

pub struct TimelineSemaphore {
    inner: Arc<TimelineSemaphoreInner>,
}
struct TimelineSemaphoreInner {
    device: Device,
    semaphore: vk::Semaphore,
    current_value: AtomicU64,
}
impl Drop for TimelineSemaphoreInner {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.semaphore, None);
        }
    }
}
