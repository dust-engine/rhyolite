use ash::vk;

use crate::ecs::queue_cap::IsGraphicsQueueCap;

use super::CommandRecorder;

impl<'w, const Q: char> CommandRecorder<'w, Q>
where
    (): IsGraphicsQueueCap<Q>,
{
    pub fn begin_rendering(self, info: &vk::RenderingInfo) -> RenderPass<'w, Q>{
        unsafe {
            self.device.cmd_begin_rendering(
                self.cmd_buf,
                &info,
            );
            RenderPass {
                recorder: self,
                is_dynamic_rendering: true,
            }
        }
    }
}

pub struct RenderPass<'w, const Q: char> where
(): IsGraphicsQueueCap<Q> {
    recorder: CommandRecorder<'w, Q>,
    is_dynamic_rendering: bool,
}

impl<'w, const Q: char> Drop for RenderPass<'w, Q>
where
    (): IsGraphicsQueueCap<Q>,
{
    fn drop(&mut self) {
        unsafe {
            if self.is_dynamic_rendering {
                self.recorder.device.cmd_end_rendering(self.recorder.cmd_buf);
            } else {
                self.recorder.device.cmd_end_render_pass(self.recorder.cmd_buf);
            }
        }
    }
}


impl<'w, const Q: char> RenderPass<'w, Q>
where
    (): IsGraphicsQueueCap<Q>,
{
}
