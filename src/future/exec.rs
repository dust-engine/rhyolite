use ash::vk;

#[derive(Default, Clone)]
pub struct GPUCommandFutureContext {
    write_stages: vk::PipelineStageFlags2,
    write_accesses: vk::AccessFlags2,
    read_stages: vk::PipelineStageFlags2,
    read_accesses: vk::AccessFlags2,
}
impl GPUCommandFutureContext {
    /// Declare a global memory write
    pub fn write(&mut self, stages: vk::PipelineStageFlags2, accesses: vk::AccessFlags2) {
        self.write_stages |= stages;
        self.write_accesses |= accesses;
    }
    /// Declare a global memory read
    pub fn read(&mut self, stages: vk::PipelineStageFlags2, accesses: vk::AccessFlags2) {
        self.read_stages |= stages;
        self.read_accesses |= accesses;
    }
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            write_stages: self.write_stages | other.write_stages,
            write_accesses: self.write_accesses | other.write_accesses,
            read_stages: self.read_stages | other.read_stages,
            read_accesses: self.read_accesses | other.read_accesses,
        }
    }
}
