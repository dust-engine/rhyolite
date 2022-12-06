use crate::Device;
use ash::vk;
use std::{
    collections::{BTreeMap, HashMap},
    task::Poll,
};

use super::GPUCommandFuture;

pub struct Res<T> {
    id: u32,
    inner: T,
}
impl<T> Res<T> {
    pub fn new(id: u32, inner: T) -> Self {
        Self { id, inner }
    }
}

pub struct Access {
    read_stages: vk::PipelineStageFlags2,
    read_access: vk::AccessFlags2,
    write_stages: vk::PipelineStageFlags2,
    write_access: vk::AccessFlags2,
}

pub struct GlobalContext {
    pub total_resource_count: usize,
    pub command_buffer: vk::CommandBuffer,
}
#[derive(Default)]
pub struct StageContext {
    accesses: BTreeMap<u32, Access>,
}

impl StageContext {
    /// Declare a global memory write
    #[inline]
    pub fn write<T>(
        &mut self,
        res: &Res<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
    ) {
        let entry = self.accesses.entry(res.id).or_insert(Access {
            read_stages: vk::PipelineStageFlags2::NONE,
            read_access: vk::AccessFlags2::NONE,
            write_stages: vk::PipelineStageFlags2::NONE,
            write_access: vk::AccessFlags2::NONE,
        });
        entry.write_stages |= stages;
        entry.write_access |= accesses;
    }
    /// Declare a global memory read
    #[inline]
    pub fn read<T>(
        &mut self,
        res: &Res<T>,
        stages: vk::PipelineStageFlags2,
        accesses: vk::AccessFlags2,
    ) {
        let entry = self.accesses.entry(res.id).or_insert(Access {
            read_stages: vk::PipelineStageFlags2::NONE,
            read_access: vk::AccessFlags2::NONE,
            write_stages: vk::PipelineStageFlags2::NONE,
            write_access: vk::AccessFlags2::NONE,
        });
        entry.read_stages |= stages;
        entry.read_access |= accesses;
    }

    pub fn merge(&mut self, mut other: Self) {
        // TODO: merge accesses. Do we actually need to merge access within each resource?
        self.accesses.append(&mut other.accesses);
    }
}

pub trait GPUCommandFutureRecordAll: GPUCommandFuture + Sized {
    #[inline]
    fn record_all(mut self, command_buffer: vk::CommandBuffer) -> Self::Output {
        let mut this = unsafe { std::pin::Pin::new_unchecked(&mut self) };
        this.as_mut().init();
        let result = loop {
            if let Poll::Ready(result) = this.as_mut().record(command_buffer) {
                break result;
            }
            println!("-----pipeline barrier-------")
        };
        println!("End");
        result
    }
}
impl<T> GPUCommandFutureRecordAll for T where T: GPUCommandFuture {}
