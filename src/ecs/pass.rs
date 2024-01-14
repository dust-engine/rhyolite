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
    ecs::{QueueAssignment, QueueSystemInitialState, QueueSystemDependencyConfig, SemaphoreOp, SemaphoreOpType},
    queue::{QueueRef, QueuesRouter}, Device,
};

use super::RenderSystemConfig;

#[derive(Debug, Default)]
pub struct RenderSystemPass {
    edge_graph: BTreeMap<(NodeId, NodeId), QueueSystemDependencyConfig>,
}

impl ScheduleBuildPass for RenderSystemPass {
    type EdgeOptions = QueueSystemDependencyConfig;

    type NodeOptions = RenderSystemConfig;

    fn add_dependency(
        &mut self,
        from: bevy_ecs::schedule::NodeId,
        to: bevy_ecs::schedule::NodeId,
        options: Option<&Self::EdgeOptions>,
    ) {
        if let Some(edge) = options {
            let old = self.edge_graph.insert((from, to), edge.clone());
            assert!(old.is_none());
        }
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
        let mut render_graph = bevy_utils::petgraph::graphmap::DiGraphMap::<usize,QueueSystemDependencyConfig>::new();

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
            let system = &graph.systems[node_id];
            let Some(render_system_config) = system.config.get::<RenderSystemConfig>()
            else {
                println!("skipped. {}", system.get().unwrap().name());
                continue; // not a render system
            };
            println!("found. {}", system.get().unwrap().name());
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
                let neighbor_system = &graph.systems[neighbor_node_id];
                // TODO: This is kinda bad. With apply_deferred inserted inbetween, we lose dependency info.
                if !neighbor_system.config.has::<RenderSystemConfig>() {
                    continue; // neighbor not a render system
                };
                let edge = self.edge_graph.get(&(node, neighbor)).cloned().unwrap_or_default();
                render_graph.add_edge(node_id, neighbor_node_id, edge);
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
        let mut heap: Vec<usize> = Vec::new(); // nodes with no incoming edges

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
        let mut colors: Vec<((Vec<usize>, bool), Vec<(Vec<usize>, bool)>)> =
            vec![Default::default(); num_queues as usize];
        let mut tiny_graph = bevy_utils::petgraph::graphmap::DiGraphMap::<u8, ()>::new();
        let mut current_graph = render_graph.clone();
        let mut heap_next_stage: Vec<usize> = Vec::new(); // nodes to be deferred to the next stage
        while let Some(node) = heap.pop() {
            let meta = render_graph_meta[node].as_ref().unwrap();
            let node_color = meta.selected_queue;
            let mut should_defer = false;
            if meta.config.is_queue_op && !colors[meta.selected_queue.0 as usize].0.0.is_empty() {
                // Can only push a queue op when there aren't other ops in the buffer
                should_defer = true;
            } else if !meta.config.is_queue_op && colors[meta.selected_queue.0 as usize].0.1 { // not a queue op, but a queue op was already queued
                should_defer = true;
            } else {
                // not a queue op. push as many as we can
                for parent in render_graph.neighbors_directed(node, Incoming) {
                    let parent_meta = render_graph_meta[parent].as_ref().unwrap();
                    if parent_meta.selected_queue != node_color {
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
            }
            if should_defer {
                // Adding this node causes a cycle in the tiny graph.
                heap_next_stage.push(node);
            } else {
                let meta = render_graph_meta[node].as_mut().unwrap();
                meta.stage_index = stage_index;
                colors[meta.selected_queue.0 as usize].0.0.push(node);
                colors[meta.selected_queue.0 as usize].0.1 |= meta.config.is_queue_op;

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
                    if !queue_node_buffer.0.is_empty() {
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
            is_queue_op: bool,
            nodes: Vec<usize>
        }
        #[derive(Debug)]
        struct QueueGraphEdge {
            config: QueueSystemDependencyConfig,
            semaphore_id: Option<SemaphoreOpType>
        }
        let mut queue_graph =
            bevy_utils::petgraph::graphmap::DiGraphMap::<QueueGraphNode, QueueGraphEdge>::new();
        let mut queue_graph_nodes = BTreeMap::<QueueGraphNode, QueueGraphNodeMeta>::new();
        // Flush all colors
        for (queue_node_buffer, stages) in colors.iter_mut() {
            if !queue_node_buffer.0.is_empty() {
                // Flush remaining nodes
                stages.push(std::mem::take(queue_node_buffer));
            }
            for stage in stages.iter_mut() {
                let mut queue_graph_node: Option<QueueGraphNode> = None;
                let mut force_binary_semaphore = false;
                for node in stage.0.iter() {
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
                    for (_, neighbor, edge) in render_graph.edges(*node) {
                        let neighbor_meta = render_graph_meta[neighbor].as_ref().unwrap();
                        let neighbor_queue_graph_node = QueueGraphNode {
                            stage_index: neighbor_meta.stage_index,
                            queue: neighbor_meta.selected_queue,
                        };
                        if queue_graph_node != neighbor_queue_graph_node {
                            // avoid self edges
                            if let Some(existing) = queue_graph.edge_weight_mut(queue_graph_node, neighbor_queue_graph_node) {
                                existing.config.wait.stage |= edge.wait.stage;
                                existing.config.wait.access |= edge.wait.access;
                                existing.config.signal.stage |= edge.signal.stage;
                                existing.config.signal.access |= edge.signal.access;
                            } else {
                                queue_graph.add_edge(queue_graph_node, neighbor_queue_graph_node, QueueGraphEdge {
                                    config: edge.clone(),
                                    semaphore_id: None,
                                });
                            }
                        }
                    }
                }
                if let Some(queue_graph_node) = queue_graph_node {
                    queue_graph_nodes.insert(
                        queue_graph_node,
                        QueueGraphNodeMeta {
                            force_binary_semaphore,
                            nodes: std::mem::take(&mut stage.0),
                            is_queue_op: stage.1,
                        },
                    );
                }
            }
        }
        println!("{:#?}", queue_graph);

        // Step 1.4: Disperse semaphores
        // Step 1.4.1: Assign semaphore IDs
        let mut binary_semaphore_id = 0;
        for (src, dst, edge) in queue_graph.all_edges_mut() {
            let config_src = &queue_graph_nodes[&src];
            let config_dst = &queue_graph_nodes[&dst];
            if config_src.force_binary_semaphore || config_dst.force_binary_semaphore {
                edge.semaphore_id = Some(SemaphoreOpType::Binary(binary_semaphore_id));
                binary_semaphore_id += 1;
            } else {
                todo!()
            }
        }
        let device = world.resource::<Device>();
        for (i, queue_node) in queue_graph_nodes.iter() {
            let mut queue_op_node: Option<usize> = None;
            for j in queue_node.nodes.iter() {
                let node = render_graph_meta[*j].as_ref().unwrap();
                assert!(node.selected_queue == i.queue);
                assert!(node.stage_index == i.stage_index);
                if node.config.is_queue_op {
                    assert!(queue_op_node.is_none(), "one queue op node per queue");
                    queue_op_node = Some(*j);
                }
            }
            let queue_op_node = queue_op_node.expect("one queue op node per queue");
            
            // need to signal for this.
            let signals:Vec<SemaphoreOp> = queue_graph.edges_directed(*i, Outgoing)
            .map(|(_, _, edge)| {
                SemaphoreOp {
                    ty: edge.semaphore_id.clone().unwrap(),
                    access: edge.config.signal.clone(),
                }
            }).collect();
            let waits:Vec<SemaphoreOp> = queue_graph.edges_directed(*i, Incoming)
            .map(|(_, _, edge)| {
                SemaphoreOp {
                    ty: edge.semaphore_id.clone().unwrap(),
                    access: edge.config.wait.clone(),
                }
            }).collect();

            let system = &mut graph.systems[queue_op_node];
            let queue = device.get_raw_queue(i.queue);
            let mut config = ConfigMap::new();
            config.insert(QueueSystemInitialState {
                queue,
                signals,
                waits
            });
            system.get_mut().unwrap().set_configs(&mut config);
        }

        // Step 1.5: Command buffer recording re-serialization

        // Step 2: inside each queue, insert pipeline barriers.

        Ok(())
    }
}
