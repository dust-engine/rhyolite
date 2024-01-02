use std::collections::BTreeSet;

use bevy_ecs::{
    component::ComponentId,
    schedule::{NodeId, ReportCycles, ScheduleBuildPass},
    world::World,
};
use bevy_utils::petgraph::{graphmap::NodeTrait, Direction::Incoming};

use crate::{ecs::QueueAssignment, queue::{QueuesRouter, QueueRef}};

use super::{RenderResRegistry, RenderSystemConfig};

#[derive(Debug)]
pub struct RenderSystemPass {}

impl ScheduleBuildPass for RenderSystemPass {
    type EdgeOptions = ();

    type NodeOptions = RenderSystemConfig;

    fn add_dependency(
        &mut self,
        from: bevy_ecs::schedule::NodeId,
        to: bevy_ecs::schedule::NodeId,
        options: Option<&Self::EdgeOptions>,
    ) {
    }

    type CollapseSetIterator = std::iter::Empty<(NodeId, NodeId)>;

    fn collapse_set(
        &mut self,
        set: bevy_ecs::schedule::NodeId,
        systems: &[bevy_ecs::schedule::NodeId],
        dependency_flattened: &bevy_utils::petgraph::prelude::GraphMap<
            bevy_ecs::schedule::NodeId,
            (),
            bevy_utils::petgraph::prelude::Directed,
        >,
    ) -> Self::CollapseSetIterator {
        std::iter::empty()
    }

    fn build(
        &mut self,
        world: &mut World,
        graph: &mut bevy_ecs::schedule::ScheduleGraph,
        dependency_flattened: &mut bevy_utils::petgraph::prelude::GraphMap<
            bevy_ecs::schedule::NodeId,
            (),
            bevy_utils::petgraph::prelude::Directed,
        >,
    ) -> Result<(), bevy_ecs::schedule::ScheduleBuildError> {
        let mut render_graph = bevy_utils::petgraph::graphmap::DiGraphMap::<usize, ()>::new();
        
        #[derive(Clone)]
        struct RenderGraphNodeMeta {
            config: RenderSystemConfig,
            selected_queue: QueueRef,
            stage_index: u32,
            ancestor_colors: u8
        }
        let mut render_graph_meta: Vec<Option<RenderGraphNodeMeta>> = vec![None; graph.systems.len()];
        let queue_router = world.resource::<QueuesRouter>();
        
        // Step 1: Queue coloring.
        // Step 1.1: Generate render graph
        let mut num_queues: u8 = 0;
        for node in dependency_flattened.nodes() {
            let NodeId::System(node_id) = node else {
                continue;
            };
            let system = graph.get_system_at(node).unwrap();
            let Some(render_system_config) = system.default_configs().and_then(|map| map.get::<RenderSystemConfig>()) else {
                println!("skipped.");
                continue; // not a render system
            };
            let selected_queue = match render_system_config.queue {
                QueueAssignment::MinOverhead(queue) | QueueAssignment::MaxAsync(queue) => {
                    queue_router.of_type(queue)
                },
                QueueAssignment::Manual(queue) => queue,
            };
            num_queues = num_queues.max(selected_queue.0 + 1);

            render_graph_meta[node_id] = Some(RenderGraphNodeMeta {
                config: render_system_config.clone(),
                selected_queue,
                stage_index: u32::MAX,
                ancestor_colors: 0
            });
            render_graph.add_node(node_id);
            for neighbor in dependency_flattened.neighbors(node) {
                let NodeId::System(neighbor_node_id) = neighbor else {
                    continue;
                };
                let neighbor_system = graph.get_system_at(neighbor).unwrap();
                if Some(true) != neighbor_system.default_configs().map(|map| map.has::<RenderSystemConfig>()) {
                    continue; // neighbor not a render system
                };
                render_graph.add_edge(node_id, neighbor_node_id, ());
            }
        }
        
        // In Vulkan, cross-queue syncronizations are heavily weight and entails
        // expensive vkQueueSubmit calls. We therefore try to merge queue nodes agressively
        // as much as we could.
        // In some APIs and platforms, it might be possible to have light-weight, frequent
        // cross-queue events. For example, D3D12 has Fences and Metal has MTLEvent.
        // This is unfortunately not exposed in Vulkan, so we're stuck agressively merging nodes.
        // https://github.com/KhronosGroup/Vulkan-Docs/issues/771

        // Step 1.2: Agressively merge nodes
        // (queue_node_buffer, mask of dependent colors of this queue node, current stage index, stages)
        // indexed by color
        let mut queue_nodes: Vec<(Vec<usize>, u32, Vec<Vec<usize>>)> = vec![
            (Vec::new(), 0, Vec::new());
            num_queues as usize
        ];
        let topo = bevy_utils::petgraph::algo::toposort(&render_graph, None).unwrap(); // TODO: better to be bredth first
        for node in topo {
            let meta = render_graph_meta[node].as_ref().unwrap();

            let (queue_node_buffer, current_stage_index, stages) = &mut queue_nodes[meta.selected_queue.0 as usize];

            let mut ancestor_mask: u8 = 0;
            let mut parent_mask: u8 = 0;
            for parent_node in render_graph.neighbors_directed(node, Incoming) {
                let parent_meta = render_graph_meta[parent_node].as_ref().unwrap();
                ancestor_mask |= parent_meta.ancestor_colors;
                parent_mask |= 1 << parent_meta.selected_queue.0;
            }
            let meta = render_graph_meta[node].as_mut().unwrap();
            meta.ancestor_colors = ancestor_mask | parent_mask;
            if parent_mask & (1 << meta.selected_queue.0) == 0 {
                // has no parent with same color
                if ancestor_mask & (1 << meta.selected_queue.0) != 0 {
                    // ancestor has same color
                    // flush node
                    let queue_node = std::mem::take(queue_node_buffer);
                    stages.push(queue_node);
                    *current_stage_index += 1;
                    println!("will have cycle");
                    meta.ancestor_colors = 0;
                }
            }
            meta.stage_index = *current_stage_index;
            queue_node_buffer.push(node);
        }

        // Step 1.3: Build queue graph
        #[derive(Hash, PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Debug)]
        struct QueueGraphNode {
            stage_index: u32,
            queue: QueueRef
        }
        let mut queue_graph = bevy_utils::petgraph::graphmap::DiGraphMap::<QueueGraphNode, ()>::new();
        // Flush all colors
        for (queue_node_buffer, _current_stage_index, stages) in queue_nodes.iter_mut() {
            if !queue_node_buffer.is_empty() {
                // Flush remaining nodes
                stages.push(std::mem::take(queue_node_buffer));
            }
            for stage in stages.iter() {
                for node in stage.iter() {
                    let meta = render_graph_meta[*node].as_ref().unwrap();
                    let queue_graph_node = QueueGraphNode {
                        stage_index: meta.stage_index,
                        queue: meta.selected_queue
                    };
                    queue_graph.add_node(queue_graph_node);
                    for neighbor in render_graph.neighbors(*node) {
                        let neighbor_meta = render_graph_meta[neighbor].as_ref().unwrap();
                        let neighbor_queue_graph_node = QueueGraphNode {
                            stage_index: neighbor_meta.stage_index,
                            queue: neighbor_meta.selected_queue
                        };
                        queue_graph.add_edge(queue_graph_node, neighbor_queue_graph_node, ());
                    }
                }
            }
        }
        println!("{:#?}", queue_graph);


        // Step 2: inside each queue, insert pipeline barriers.

        Ok(())
    }
}
