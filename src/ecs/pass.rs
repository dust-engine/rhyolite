use std::collections::BTreeMap;

use ash::vk;
use bevy_ecs::{
    schedule::{NodeId, ScheduleBuildPass},
    world::World,
};
use bevy_utils::{petgraph::{
    visit::{Dfs, Walker},
    Direction::{Incoming, Outgoing},
}, ConfigMap};

use crate::{
    ecs::{QueueAssignment, QueueSystemState},
    queue::{QueueRef, QueuesRouter}, Device,
};

use super::RenderSystemConfig;

#[derive(Debug)]
pub struct RenderSystemPass {}

impl ScheduleBuildPass for RenderSystemPass {
    type EdgeOptions = ();

    type NodeOptions = RenderSystemConfig;

    fn add_dependency(
        &mut self,
        _from: bevy_ecs::schedule::NodeId,
        _to: bevy_ecs::schedule::NodeId,
        _options: Option<&Self::EdgeOptions>,
    ) {
    }

    type CollapseSetIterator = std::iter::Empty<(NodeId, NodeId)>;

    fn collapse_set(
        &mut self,
        _set: bevy_ecs::schedule::NodeId,
        _systems: &[bevy_ecs::schedule::NodeId],
        _dependency_flattened: &bevy_utils::petgraph::prelude::GraphMap<
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
        }
        let mut render_graph_meta: Vec<Option<RenderGraphNodeMeta>> =
            vec![None; graph.systems.len()];
        let queue_router = world.resource::<QueuesRouter>();

        // Step 1: Queue coloring.
        // Step 1.1: Generate render graph
        let mut num_queues: u8 = 0;
        for node in dependency_flattened.nodes() {
            let NodeId::System(node_id) = node else {
                continue;
            };
            let config = &graph.systems[node_id].config;
            let Some(render_system_config) = config.get::<RenderSystemConfig>()
            else {
                println!("skipped.");
                continue; // not a render system
            };
            let selected_queue = match render_system_config.queue {
                QueueAssignment::MinOverhead(queue) | QueueAssignment::MaxAsync(queue) => {
                    queue_router.of_type(queue)
                }
                QueueAssignment::Manual(queue) => queue,
            };
            num_queues = num_queues.max(selected_queue.0 + 1);

            render_graph_meta[node_id] = Some(RenderGraphNodeMeta {
                config: render_system_config.clone(),
                selected_queue,
                stage_index: u32::MAX,
            });
            render_graph.add_node(node_id);
            for neighbor in dependency_flattened.neighbors(node) {
                let NodeId::System(neighbor_node_id) = neighbor else {
                    continue;
                };
                let neighbor_config = &graph.systems[neighbor_node_id].config;
                if !neighbor_config.has::<RenderSystemConfig>() {
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

        // The algorithm tries to find a topological sorting for each queue while minimizing the number
        // of semaphore syncronizations. On each iteration, we produce 0 or 1 "stage" for each queue.
        // We does this by popping nodes with no incoming edges as much as possible, as long as doing so
        // will not cause a cycle between queue nodes in the current stage. We do this by maintaining a
        // tiny graph with up to X nodes, where X = the total number of queues. Every time we pop a node,
        // we ensure that doing so will not cause a cycle in this tiny graph. Otherwise, we defer
        // this node to the next stage.

        // Step 1.2: Agressively merge nodes
        let mut heap: Vec<usize> = Vec::new();

        // First, find all nodes with no incoming edges
        for node in render_graph.nodes() {
            if render_graph
                .neighbors_directed(node, Incoming)
                .next()
                .is_none()
            {
                // Has no incoming edges
                let _meta = render_graph_meta[node].as_mut().unwrap();
                heap.push(node);
            }
        }
        let mut stage_index = 0;
        // (buffer, stages)
        let mut colors: Vec<(Vec<usize>, Vec<Vec<usize>>)> =
            vec![Default::default(); num_queues as usize];
        let mut tiny_graph = bevy_utils::petgraph::graphmap::DiGraphMap::<u8, ()>::new();
        let mut current_graph = render_graph.clone();
        let mut heap_next_stage: Vec<usize> = Vec::new(); // nodes to be deferred to the next stage
        while let Some(node) = heap.pop() {
            let meta = render_graph_meta[node].as_ref().unwrap();
            let node_color = meta.selected_queue;
            let mut should_defer = false;
            for parent in render_graph.neighbors_directed(node, Incoming) {
                let parent_meta = render_graph_meta[parent].as_ref().unwrap();
                if parent_meta.selected_queue != node_color ||
                // In the case that they're the same color, the parent node can't also be a queue operation.
                // Queue operations cannot be merged like this.
                (parent_meta.config.is_queue_op || meta.config.is_queue_op) 
                 {
                    let start = parent_meta.selected_queue.0;
                    let end = node_color.0;
                    let has_path = Dfs::new(&tiny_graph, end)
                        .iter(&tiny_graph)
                        .any(|x| x == start);
                    if has_path {
                        // There is already a path from end to start, so adding a node from start to end causes a cycle.
                        should_defer = true;
                        break;
                    }
                }
            }
            if should_defer {
                // Adding this node causes a cycle in the tiny graph.
                heap_next_stage.push(node);
            } else {
                let meta = render_graph_meta[node].as_mut().unwrap();
                meta.stage_index = stage_index;
                colors[meta.selected_queue.0 as usize].0.push(node);

                for parent in render_graph.neighbors_directed(node, Incoming) {
                    // Update the tiny graph.
                    let parent_meta = render_graph_meta[parent].as_mut().unwrap();
                    if parent_meta.selected_queue != node_color && parent_meta.stage_index == stage_index {
                        let start = parent_meta.selected_queue.0;
                        let end = node_color.0;
                        tiny_graph.add_edge(start, end, ());
                    }
                }

                for child in current_graph.neighbors_directed(node, Outgoing) {
                    let mut other_parents = current_graph.neighbors_directed(child, Incoming);
                    other_parents.next().unwrap();
                    if other_parents.next().is_some() {
                        // other edges exist
                        continue;
                    }
                    // no other edges
                    heap.push(child);
                }
                current_graph.remove_node(node);
            }

            if heap.is_empty() {
                // Flush all colors
                for (queue_node_buffer, stages) in colors.iter_mut() {
                    if !queue_node_buffer.is_empty() {
                        // Flush remaining nodes
                        stages.push(std::mem::take(queue_node_buffer));
                    }
                }
                // Start a new stage
                stage_index += 1;
                tiny_graph.clear(); // Clear the tiny graph because we've flipped to a new stage.
                std::mem::swap(&mut heap, &mut heap_next_stage);
            }
        }

        // Step 1.3: Build queue graph
        #[derive(Hash, PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Debug)]
        struct QueueGraphNode {
            stage_index: u32,
            queue: QueueRef,
        }
        
        #[derive(Clone)]
        struct QueueGraphNodeMeta {
            force_binary_semaphore: bool,
            nodes: Vec<usize>
        }
        let mut queue_graph =
            bevy_utils::petgraph::graphmap::DiGraphMap::<QueueGraphNode, ()>::new();
        let mut queue_graph_nodes = BTreeMap::<QueueGraphNode, QueueGraphNodeMeta>::new();
        // Flush all colors
        for (queue_node_buffer, stages) in colors.iter_mut() {
            if !queue_node_buffer.is_empty() {
                // Flush remaining nodes
                stages.push(std::mem::take(queue_node_buffer));
            }
            for stage in stages.iter_mut() {
                let mut queue_graph_node: Option<QueueGraphNode> = None;
                let mut force_binary_semaphore = false;
                for node in stage.iter() {
                    let meta = render_graph_meta[*node].as_ref().unwrap();
                    if let Some(queue_graph_node) = &queue_graph_node {
                        // All nodes in here shall have the same stage index and selected queue
                        assert_eq!(queue_graph_node.stage_index, meta.stage_index);
                        assert_eq!(queue_graph_node.queue, meta.selected_queue);
                    }
                    let queue_graph_node = queue_graph_node.get_or_insert(QueueGraphNode {
                        stage_index: meta.stage_index,
                        queue: meta.selected_queue,
                    }).clone();
                    queue_graph.add_node(queue_graph_node);
                    force_binary_semaphore = meta.config.force_binary_semaphore;
                    for neighbor in render_graph.neighbors(*node) {
                        let neighbor_meta = render_graph_meta[neighbor].as_ref().unwrap();
                        let neighbor_queue_graph_node = QueueGraphNode {
                            stage_index: neighbor_meta.stage_index,
                            queue: neighbor_meta.selected_queue,
                        };
                        if queue_graph_node != neighbor_queue_graph_node {
                            // avoid self edges
                            queue_graph.add_edge(queue_graph_node, neighbor_queue_graph_node, ());
                        }
                    }
                }
                if let Some(queue_graph_node) = queue_graph_node {
                    queue_graph_nodes.insert(
                        queue_graph_node,
                        QueueGraphNodeMeta {
                            force_binary_semaphore,
                            nodes: std::mem::take(stage)
                        },
                    );
                }
            }
        }
        println!("{:#?}", queue_graph);

        // Step 1.4: Disperse semaphores
        let device = world.resource::<Device>();
        for (i, queue_node) in queue_graph_nodes.iter() {
            for j in queue_node.nodes.iter() {
                let node = render_graph_meta[*j].as_ref().unwrap();
                if node.config.is_queue_op {
                    // need to signal for this.
                    let system = &mut graph.systems[*j];
                    let queue = device.get_raw_queue(node.selected_queue);
                    let mut config = ConfigMap::new();
                    config.insert(QueueSystemState {
                        queue,
                        semaphore_waits: vec![],
                        semaphore_signals: vec![],
                    });
                    system.get_mut().unwrap().set_configs(&mut config);
                }
            }
            // add a node and signal for all command nodes.
        }

        // Step 1.5: Command buffer recording re-serialization

        // Step 2: inside each queue, insert pipeline barriers.

        Ok(())
    }
}
