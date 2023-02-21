use std::collections::HashMap;

use ash::vk;
pub struct SpirvShader<T> {
    pub data: T,
    pub entry_points: HashMap<String, SpirvEntryPoint>
}

pub struct SpirvEntryPoint {
    descriptor_sets: Vec<vk::DescriptorSetLayoutCreateInfo>
}