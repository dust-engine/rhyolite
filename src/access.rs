use ash::vk;

use crate::access::utils::compare_pipeline_stages;

#[derive(Debug, Clone, Default)]
pub struct Access {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryBarrier {
    pub src_stage_mask: vk::PipelineStageFlags2,
    pub src_access_mask: vk::AccessFlags2,
    pub dst_stage_mask: vk::PipelineStageFlags2,
    pub dst_access_mask: vk::AccessFlags2,
}

impl Access {
    pub fn is_writeonly(&self) -> bool {
        if self.access == vk::AccessFlags2::empty() {
            return false;
        }
        // Clear all the write bits. If nothing is left, that means there's no read bits.
        self.access & !utils::ALL_WRITE_BITS == vk::AccessFlags2::NONE
    }

    pub fn is_readonly(&self) -> bool {
        if self.access == vk::AccessFlags2::empty() {
            return false;
        }
        // Clear all the read bits. If nothing is left, that means there's no write bits.
        self.access & !utils::ALL_READ_BITS == vk::AccessFlags2::NONE
    }
}

#[derive(Clone, Default)]
pub struct ResourceState {
    pub(crate) read: Access,
    pub(crate) write: Access,
}
impl ResourceState {
    pub fn transition(&mut self, next: Access, with_layout_transition: bool) -> MemoryBarrier {
        let mut barrier = MemoryBarrier {
            src_stage_mask: self.write.stage,
            src_access_mask: self.write.access,
            dst_stage_mask: next.stage,
            dst_access_mask: next.access,
        };
        if self.write.access == vk::AccessFlags2::empty()
            && self.write.stage == vk::PipelineStageFlags2::empty()
        {
            // Resource was never accessed before.
            if with_layout_transition {
                barrier.src_access_mask = vk::AccessFlags2::empty();
                barrier.src_stage_mask = vk::PipelineStageFlags2::empty();
            } else {
                barrier = MemoryBarrier::default();
            }
        } else if next.is_readonly() {
            if let Some(ordering) = compare_pipeline_stages(self.read.stage, next.stage) {
                if ordering.is_gt() {
                    barrier.src_stage_mask = self.read.stage;
                    barrier.src_access_mask = vk::AccessFlags2::empty();
                    barrier.dst_access_mask = vk::AccessFlags2::empty();
                } else {
                    // it has already been made visible at the desired stage
                    barrier = MemoryBarrier::default();
                }
            }
        } else {
            // The next stage will be writing.
            if self.read.stage != vk::PipelineStageFlags2::empty() {
                // This is a WAR hazard, which you would usually only need an execution dependency for.
                // meaning you wouldn't need to supply any memory barriers.
                barrier.src_stage_mask = self.read.stage;
                barrier.src_access_mask = vk::AccessFlags2::empty();
                if !with_layout_transition {
                    // When we do have a layout transition, you still need a memory barrier,
                    // but you don't need any access types in the src access mask. The layout transition
                    // itself is considered a write operation though, so you do need the destination
                    // access mask to be correct - or there would be a WAW hazard between the layout
                    // transition and the color attachment write.
                    barrier.dst_access_mask = vk::AccessFlags2::empty();
                }
            }
        }
        // the info is now available for all stages after next.stage, but not for stages before next.stage.
        if next.is_readonly() {
            self.read.stage = utils::earlier_stage(self.read.stage, next.stage);
        } else {
            self.write = next.clone();
            self.read = Access::default();
        }
        barrier
    }
}

pub mod utils {
    use ash::vk;
    use std::cmp::Ordering;
    pub const ALL_WRITE_BITS: vk::AccessFlags2 = vk::AccessFlags2::from_raw(
        vk::AccessFlags2::SHADER_WRITE.as_raw()
            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw()
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
            | vk::AccessFlags2::TRANSFER_WRITE.as_raw()
            | vk::AccessFlags2::HOST_WRITE.as_raw()
            | vk::AccessFlags2::MEMORY_WRITE.as_raw()
            | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw()
            | vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR.as_raw()
            | vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR.as_raw()
            | vk::AccessFlags2::TRANSFORM_FEEDBACK_WRITE_EXT.as_raw()
            | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT.as_raw()
            | vk::AccessFlags2::COMMAND_PREPROCESS_WRITE_NV.as_raw()
            | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR.as_raw()
            | vk::AccessFlags2::MICROMAP_WRITE_EXT.as_raw()
            | vk::AccessFlags2::OPTICAL_FLOW_WRITE_NV.as_raw(),
    );
    pub const ALL_READ_BITS: vk::AccessFlags2 = vk::AccessFlags2::from_raw(
        vk::AccessFlags2::INDIRECT_COMMAND_READ.as_raw()
            | vk::AccessFlags2::INDEX_READ.as_raw()
            | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ.as_raw()
            | vk::AccessFlags2::UNIFORM_READ.as_raw()
            | vk::AccessFlags2::INPUT_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags2::SHADER_READ.as_raw()
            | vk::AccessFlags2::COLOR_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags2::TRANSFER_READ.as_raw()
            | vk::AccessFlags2::HOST_READ.as_raw()
            | vk::AccessFlags2::MEMORY_READ.as_raw()
            | vk::AccessFlags2::SHADER_SAMPLED_READ.as_raw()
            | vk::AccessFlags2::SHADER_STORAGE_READ.as_raw()
            | vk::AccessFlags2::VIDEO_DECODE_READ_KHR.as_raw()
            | vk::AccessFlags2::VIDEO_ENCODE_READ_KHR.as_raw()
            | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_READ_EXT.as_raw()
            | vk::AccessFlags2::CONDITIONAL_RENDERING_READ_EXT.as_raw()
            | vk::AccessFlags2::COMMAND_PREPROCESS_READ_NV.as_raw()
            | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR.as_raw()
            | vk::AccessFlags2::FRAGMENT_DENSITY_MAP_READ_EXT.as_raw()
            | vk::AccessFlags2::COLOR_ATTACHMENT_READ_NONCOHERENT_EXT.as_raw()
            | vk::AccessFlags2::DESCRIPTOR_BUFFER_READ_EXT.as_raw()
            | vk::AccessFlags2::INVOCATION_MASK_READ_HUAWEI.as_raw()
            | vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR.as_raw()
            | vk::AccessFlags2::MICROMAP_READ_EXT.as_raw()
            | vk::AccessFlags2::OPTICAL_FLOW_READ_NV.as_raw(),
    );
    const GRAPHICS_PIPELINE_ORDER: [vk::PipelineStageFlags2; 13] = [
        vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::PipelineStageFlags2::INDEX_INPUT,
        vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
        vk::PipelineStageFlags2::VERTEX_SHADER,
        vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
        vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
        vk::PipelineStageFlags2::GEOMETRY_SHADER,
        vk::PipelineStageFlags2::TRANSFORM_FEEDBACK_EXT,
        vk::PipelineStageFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR,
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
    ];
    const GRAPHICS_MESH_PIPELINE_ORDER: [vk::PipelineStageFlags2; 8] = [
        vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::PipelineStageFlags2::TASK_SHADER_EXT,
        vk::PipelineStageFlags2::MESH_SHADER_EXT,
        vk::PipelineStageFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR,
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
    ];
    const COMPUTE_PIPELINE_ORDER: [vk::PipelineStageFlags2; 2] = [
        vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::PipelineStageFlags2::COMPUTE_SHADER,
    ];
    const FRAGMENT_DENSITY_ORDER: [vk::PipelineStageFlags2; 2] = [
        vk::PipelineStageFlags2::FRAGMENT_DENSITY_PROCESS_EXT,
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
    ];
    const RAYTRACING_PIPELINE_ORDER: [vk::PipelineStageFlags2; 2] = [
        vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
    ];
    const ALL_ORDERS: [&[vk::PipelineStageFlags2]; 5] = [
        &GRAPHICS_PIPELINE_ORDER,
        &GRAPHICS_MESH_PIPELINE_ORDER,
        &COMPUTE_PIPELINE_ORDER,
        &FRAGMENT_DENSITY_ORDER,
        &RAYTRACING_PIPELINE_ORDER,
    ];
    /// Compare two pipeline stages. Returns [`Some(Ordering::Less)`] if `a` is earlier than `b`,
    /// [`Ordering::Equal`] if they are the same, [`Ordering::Greater`] if `a` is later than `b`,
    /// and [`None`] if they are not in the same time and are not mutually ordered.
    pub fn compare_pipeline_stages(
        a: vk::PipelineStageFlags2,
        b: vk::PipelineStageFlags2,
    ) -> Option<Ordering> {
        if a == b {
            return Some(std::cmp::Ordering::Equal);
        }
        for order in ALL_ORDERS.iter() {
            let first_index: Option<usize> = order.iter().position(|&x| a.contains(x));
            let second_index: Option<usize> = order.iter().position(|&x| b.contains(x));
            if let Some(first_index) = first_index {
                if let Some(second_index) = second_index {
                    return first_index.partial_cmp(&second_index);
                }
            }
        }
        None
    }
    pub fn earlier_stage(
        a: vk::PipelineStageFlags2,
        b: vk::PipelineStageFlags2,
    ) -> vk::PipelineStageFlags2 {
        if let Some(ordering) = compare_pipeline_stages(a, b) {
            if ordering.is_le() {
                a
            } else {
                b
            }
        } else {
            a | b
        }
    }
    pub fn later_stage(
        a: vk::PipelineStageFlags2,
        b: vk::PipelineStageFlags2,
    ) -> vk::PipelineStageFlags2 {
        if let Some(ordering) = compare_pipeline_stages(a, b) {
            if ordering.is_ge() {
                a
            } else {
                b
            }
        } else {
            a | b
        }
    }
}

#[cfg(test)]
mod tests {
    use super::utils::*;
    use super::*;
    #[test]
    fn test_earlier_stage() {
        assert_eq!(
            earlier_stage(
                vk::PipelineStageFlags2::INDEX_INPUT,
                vk::PipelineStageFlags2::INDEX_INPUT
            ),
            vk::PipelineStageFlags2::INDEX_INPUT
        );
        assert_eq!(
            earlier_stage(
                vk::PipelineStageFlags2::VERTEX_SHADER,
                vk::PipelineStageFlags2::FRAGMENT_SHADER
            ),
            vk::PipelineStageFlags2::VERTEX_SHADER
        );
        assert_eq!(
            earlier_stage(
                vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags2::FRAGMENT_SHADER
            ),
            vk::PipelineStageFlags2::FRAGMENT_SHADER
        );
        assert_eq!(
            earlier_stage(
                vk::PipelineStageFlags2::VERTEX_SHADER,
                vk::PipelineStageFlags2::TRANSFER
            ),
            vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::TRANSFER
        );
    }
    #[test]
    fn test_compare_pipeline_stages() {
        assert!(compare_pipeline_stages(
            vk::PipelineStageFlags2::INDEX_INPUT,
            vk::PipelineStageFlags2::INDEX_INPUT
        )
        .unwrap()
        .is_eq());
        assert!(compare_pipeline_stages(
            vk::PipelineStageFlags2::INDEX_INPUT,
            vk::PipelineStageFlags2::FRAGMENT_SHADER
        )
        .unwrap()
        .is_lt());
        assert!(compare_pipeline_stages(
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
            vk::PipelineStageFlags2::FRAGMENT_SHADER
        )
        .unwrap()
        .is_lt());
        assert!(compare_pipeline_stages(
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
        )
        .unwrap()
        .is_lt());
        assert!(compare_pipeline_stages(
            vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::PipelineStageFlags2::FRAGMENT_SHADER
        )
        .unwrap()
        .is_gt());
        assert!(compare_pipeline_stages(
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::PipelineStageFlags2::FRAGMENT_SHADER
        )
        .is_none());
        assert!(compare_pipeline_stages(
            vk::PipelineStageFlags2::TASK_SHADER_EXT,
            vk::PipelineStageFlags2::VERTEX_SHADER
        )
        .is_none());
    }

    /// Write - Read - Read, reads are ordered-later
    #[test]
    fn test_wrr() {
        let mut access = ResourceState::default();
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                dst_access_mask: vk::AccessFlags2::SHADER_READ,
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                src_access_mask: vk::AccessFlags2::empty(),
                dst_stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
                dst_access_mask: vk::AccessFlags2::empty(),
            }
        );
    }

    /// Write - Read - Read, reads are ordered-earlier
    #[test]
    fn test_wrr2() {
        let mut access = ResourceState::default();
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
                dst_access_mask: vk::AccessFlags2::SHADER_READ,
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        // The fragment shader resource usage would've been synced by the previous barrier too.
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
    }
    /// Write - Read - Read, reads are ordered the same
    #[test]
    fn test_wrr3() {
        let mut access = ResourceState::default();
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                dst_access_mask: vk::AccessFlags2::SHADER_READ,
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        // The fragment shader resource usage would've been synced by the previous barrier too.
        assert_eq!(barrier, MemoryBarrier::default());
    }
    /// Write - Read - Read, reads are not ordered
    #[test]
    fn test_wrr4() {
        let mut access = ResourceState::default();
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
                dst_access_mask: vk::AccessFlags2::SHADER_READ,
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::TASK_SHADER_EXT,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::TASK_SHADER_EXT,
                dst_access_mask: vk::AccessFlags2::SHADER_READ,
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::MESH_SHADER_EXT,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
    }

    #[test]
    fn test_wrw() {
        let mut access = ResourceState::default();
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
                dst_access_mask: vk::AccessFlags2::SHADER_READ,
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_WRITE,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src_stage_mask: vk::PipelineStageFlags2::VERTEX_SHADER,
                src_access_mask: vk::AccessFlags2::empty(),
                dst_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                dst_access_mask: vk::AccessFlags2::empty(),
            }
        );
    }

    #[test]
    fn test_wrw2() {
        let mut access = ResourceState::default();
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_WRITE,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                src_access_mask: vk::AccessFlags2::SHADER_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_WRITE,
            },
            false,
        );
        // We need to wait on the read from the second operation to finish.
        // We also need to wait on the write from the first operation but this is done through an indirect barrier.
        assert_eq!(
            barrier,
            MemoryBarrier {
                src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: vk::AccessFlags2::empty(),
                dst_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                dst_access_mask: vk::AccessFlags2::empty(),
            }
        );
    }

    #[test]
    fn test_www() {
        let mut access = ResourceState::default();
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_WRITE,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                src_access_mask: vk::AccessFlags2::SHADER_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_WRITE,
            },
            false,
        );
        // We need to wait on the read from the second operation to finish.
        // We also need to wait on the write from the first operation but this is done through an indirect barrier.
        assert_eq!(
            barrier,
            MemoryBarrier {
                src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                dst_access_mask: vk::AccessFlags2::SHADER_WRITE,
            }
        );
    }

    #[test]
    fn test_wrw_layout_transition() {
        // First draw samples a texture in the fragment shader. Second draw writes to that texture as a color attachment.
        let mut access = ResourceState::default();
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::TRANSFER,
                access: vk::AccessFlags2::TRANSFER_WRITE,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_SAMPLED_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
                src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                dst_access_mask: vk::AccessFlags2::SHADER_SAMPLED_READ,
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            },
            true,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                src_access_mask: vk::AccessFlags2::empty(),
                dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            }
        );
    }
}
