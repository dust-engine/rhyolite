use ash::vk;
use std::task::Poll;
use crate::Device;

use super::GPUCommandFuture;

pub struct GPUCommandFutureContext {
    write_stages: vk::PipelineStageFlags2,
    write_accesses: vk::AccessFlags2,
    read_stages: vk::PipelineStageFlags2,
    read_accesses: vk::AccessFlags2,
}

impl Default for GPUCommandFutureContext {
    #[inline]
    fn default() -> Self {
        GPUCommandFutureContext {
            write_stages: vk::PipelineStageFlags2::empty(),
            write_accesses: vk::AccessFlags2::empty(),
            read_stages: vk::PipelineStageFlags2::empty(),
            read_accesses: vk::AccessFlags2::empty()
        }
    }
}
impl Clone for GPUCommandFutureContext {
    #[inline]
    fn clone(&self) -> Self {
        GPUCommandFutureContext {
            write_stages: self.write_stages,
            write_accesses: self.write_accesses,
            read_stages: self.read_stages,
            read_accesses: self.read_accesses
        }
    }
}
impl GPUCommandFutureContext {
    /// Declare a global memory write
    #[inline]
    pub fn write(&mut self, stages: vk::PipelineStageFlags2, accesses: vk::AccessFlags2) {
        self.write_stages |= stages;
        self.write_accesses |= accesses;
    }
    /// Declare a global memory read
    #[inline]
    pub fn read(&mut self, stages: vk::PipelineStageFlags2, accesses: vk::AccessFlags2) {
        self.read_stages |= stages;
        self.read_accesses |= accesses;
    }
    #[inline]
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            write_stages: self.write_stages | other.write_stages,
            write_accesses: self.write_accesses | other.write_accesses,
            read_stages: self.read_stages | other.read_stages,
            read_accesses: self.read_accesses | other.read_accesses,
        }
    }
}


pub trait GPUCommandFutureRecordAll: GPUCommandFuture + Sized {
    #[inline]
    fn record_all(self, command_buffer: vk::CommandBuffer) -> Self::Output {
        let mut this = std::pin::pin!(self);
        this.as_mut().init();
        let mut current_context = this.as_mut().context();
        let result = loop {
            if let Poll::Ready(result) = this.as_mut().record(command_buffer) {
                break result;
            }
            // Now, record the pipeline barrier.
            let next_context = this.as_ref().context();

            current_context = next_context;
        };
        result
    }
}
impl<T> GPUCommandFutureRecordAll for T where T: GPUCommandFuture {}