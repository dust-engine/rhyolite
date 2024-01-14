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
        self.recycler.send(self.semaphore);
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


