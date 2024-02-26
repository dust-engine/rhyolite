use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use bevy_ecs::{
    query::Access,
    schedule::{IntoSystemConfigs, NodeId, ScheduleBuildPass, SystemNode},
    system::{BoxedSystem, IntoSystem, System},
    world::World,
};
use bevy_utils::petgraph::{
    graph::DiGraph,
    graphmap::DiGraphMap,
    visit::{Dfs, EdgeRef, IntoEdgeReferences, IntoNeighbors, IntoNeighborsDirected, Walker},
    Direction::{Incoming, Outgoing},
};

use crate::{
    ecs::{
        BinarySemaphoreOp, QueueSystemDependencyConfig, QueueSystemInitialState,
        RenderSystemInitialState, RenderSystemsBinarySemaphoreTracker, TimelineSemaphoreOp,
    },
    queue::{QueueRef, QueuesRouter},
    semaphore::TimelineSemaphore,
    Device, QueueType,
};

use super::RenderSystemConfig;

#[derive(Debug)]
pub struct RenderSystemPass {
    edge_graph: BTreeMap<(NodeId, NodeId), QueueSystemDependencyConfig>,
    queue_graph: bevy_utils::petgraph::graphmap::DiGraphMap<u32, QueueGraphEdge>,
    queue_graph_nodes: Vec<QueueGraphNodeMeta>,
    num_binary_semaphores: u32,
}

impl RenderSystemPass {
    pub fn new() -> Self {
        Self {
            edge_graph: BTreeMap::new(),
            queue_graph: Default::default(),
            queue_graph_nodes: Vec::new(),
            num_binary_semaphores: 0,
        }
    }
}

#[derive(Debug)]
enum QueueGraphEdgeSemaphoreType {
    Binary(u32),
    Timeline(u32),
}

#[derive(Debug)]
struct QueueGraphEdge {
    config: QueueSystemDependencyConfig,
    semaphore_id: Option<QueueGraphEdgeSemaphoreType>,
}

#[derive(Clone, Debug)]
struct QueueGraphNodeMeta {
    force_binary_semaphore: bool,
    is_queue_op: bool,
    /// for the queue node, this is the system id of the "queue op" system?
    queue_node_index: usize,
    nodes: Vec<usize>,
    selected_queue: QueueRef,
    queue_type: QueueType,
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
        let mut render_graph =
            bevy_utils::petgraph::graphmap::DiGraphMap::<usize, QueueSystemDependencyConfig>::new();

        struct RenderGraphNodeMeta {
            queue: QueueType,
            force_binary_semaphore: bool,
            is_queue_op: bool,

            selected_queue: QueueRef,
            stage_index: u32,
            queue_graph_node: u32,
            queue_type: QueueType,
        }
        let mut render_graph_meta: Vec<Option<RenderGraphNodeMeta>> =
            (0..graph.systems.len()).map(|_| None).collect();
        let queue_router = world.resource::<QueuesRouter>();

        // Step 1: Queue coloring.
        // Step 1.1: Generate render graph
        let mut num_queues: u8 = 0;
        for node in dependency_flattened.nodes() {
            let NodeId::System(node_id) = node else {
                continue;
            };
            let system = &graph.systems[node_id];
            let Some(render_system_config) = system.config.get::<RenderSystemConfig>() else {
                continue; // not a render system
            };
            let selected_queue = queue_router.of_type(render_system_config.queue);
            num_queues = num_queues.max(selected_queue.0 + 1);

            render_graph_meta[node_id] = Some(RenderGraphNodeMeta {
                queue: render_system_config.queue,
                force_binary_semaphore: render_system_config.force_binary_semaphore,
                is_queue_op: render_system_config.is_queue_op,

                selected_queue,
                stage_index: u32::MAX,
                queue_graph_node: 0,
                queue_type: render_system_config.queue,
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
                let edge = self
                    .edge_graph
                    .get(&(node, neighbor))
                    .cloned()
                    .unwrap_or_default();
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
            if meta.is_queue_op && !colors[meta.selected_queue.0 as usize].0 .0.is_empty() {
                // Can only push a queue op when there aren't other ops in the buffer
                should_defer = true;
            } else if !meta.is_queue_op && colors[meta.selected_queue.0 as usize].0 .1 {
                // not a queue op, but a queue op was already queued
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
                colors[meta.selected_queue.0 as usize].0 .0.push(node);
                colors[meta.selected_queue.0 as usize].0 .1 |= meta.is_queue_op;

                for parent in render_graph.neighbors_directed(node, Incoming) {
                    // Update the tiny graph.
                    let parent_meta = render_graph_meta[parent].as_mut().unwrap();
                    if parent_meta.selected_queue != node_color
                        && parent_meta.stage_index == stage_index
                    {
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
            queue_type: QueueType,
        }

        let mut queue_graph =
            bevy_utils::petgraph::graphmap::DiGraphMap::<u32, QueueGraphEdge>::new();
        let mut queue_graph_nodes = Vec::<QueueGraphNodeMeta>::new();
        // Flush all colors
        for (queue_node_buffer, stages) in colors.iter_mut() {
            if !queue_node_buffer.0.is_empty() {
                // Flush remaining nodes
                stages.push(std::mem::take(queue_node_buffer));
            }
            for stage in stages.iter_mut() {
                let mut queue_graph_node_info: Option<QueueGraphNode> = None;
                let mut force_binary_semaphore = false;
                let queue_graph_node = queue_graph_nodes.len() as u32;
                queue_graph.add_node(queue_graph_node);

                for node in stage.0.iter() {
                    let meta = render_graph_meta[*node].as_mut().unwrap();
                    meta.queue_graph_node = queue_graph_node;
                    if let Some(queue_graph_node_info) = &queue_graph_node_info {
                        // All nodes in here shall have the same stage index and selected queue
                        assert_eq!(queue_graph_node_info.stage_index, meta.stage_index);
                        assert_eq!(queue_graph_node_info.queue, meta.selected_queue);
                    }
                    let _queue_graph_node = queue_graph_node_info
                        .get_or_insert(QueueGraphNode {
                            stage_index: meta.stage_index,
                            queue: meta.selected_queue,
                            queue_type: meta.queue_type,
                        })
                        .clone();
                    force_binary_semaphore |= meta.force_binary_semaphore;
                }
                if let Some(queue_graph_node_info) = queue_graph_node_info {
                    queue_graph_nodes.push(QueueGraphNodeMeta {
                        force_binary_semaphore,
                        nodes: std::mem::take(&mut stage.0),
                        is_queue_op: stage.1,
                        queue_node_index: 0,
                        selected_queue: queue_graph_node_info.queue,
                        queue_type: queue_graph_node_info.queue_type,
                    });
                }
            }
        }
        // queue graph connectivity
        for (from, to, edge) in render_graph.all_edges() {
            let from_meta = render_graph_meta[from].as_ref().unwrap();
            let to_meta = render_graph_meta[to].as_ref().unwrap();
            if from_meta.queue_graph_node != to_meta.queue_graph_node {
                // avoid self edges
            }

            if let Some(existing) =
                queue_graph.edge_weight_mut(from_meta.queue_graph_node, to_meta.queue_graph_node)
            {
                existing.config.wait.stage |= edge.wait.stage;
                existing.config.wait.access |= edge.wait.access;
                existing.config.signal.stage |= edge.signal.stage;
                existing.config.signal.access |= edge.signal.access;
            } else {
                queue_graph.add_edge(
                    from_meta.queue_graph_node,
                    to_meta.queue_graph_node,
                    QueueGraphEdge {
                        config: edge.clone(),
                        semaphore_id: None,
                    },
                );
            }
        }
        // Step 1.4: Insert queue submission systems
        for (_i, queue_node) in queue_graph_nodes.iter_mut().enumerate() {
            if queue_node.is_queue_op {
                assert!(queue_node.nodes.len() == 1);
                queue_node.queue_node_index = queue_node.nodes[0];
                continue;
            }
            let id_num = graph.systems.len();
            let id = NodeId::System(id_num);
            let mut system: BoxedSystem = match queue_node.queue_type {
                QueueType::Graphics => Box::new(IntoSystem::into_system(
                    crate::ecs::flush_system_graph::<'g'>,
                )),
                QueueType::Compute => Box::new(IntoSystem::into_system(
                    crate::ecs::flush_system_graph::<'c'>,
                )),
                QueueType::Transfer => Box::new(IntoSystem::into_system(
                    crate::ecs::flush_system_graph::<'t'>,
                )),
                _ => unimplemented!(),
            };

            system.initialize(world);
            graph
                .systems
                .push(SystemNode::new(system, Default::default()));
            graph.system_conditions.push(Vec::new());
            graph.ambiguous_with_all.insert(id);

            queue_node.queue_node_index = id_num;
        }
        // Connect queue submission systems with parents and children
        for (i, queue_node) in queue_graph_nodes.iter().enumerate() {
            if queue_node.is_queue_op {
                continue;
            }
            let new_node = NodeId::System(queue_node.queue_node_index);
            for parent_command_node in queue_node.nodes.iter() {
                // After all nodes within the corrent queue node
                let command_node_id = NodeId::System(*parent_command_node);
                dependency_flattened.add_edge(command_node_id, new_node, ());
            }
            for parent in queue_graph.neighbors_directed(i as u32, Outgoing) {
                // Before all command nodes in the queue nodes following the current
                let child_meta = &queue_graph_nodes[parent as usize];
                for child_command_node in child_meta.nodes.iter() {
                    let command_node_id = NodeId::System(*child_command_node);
                    dependency_flattened.add_edge(new_node, command_node_id, ());
                }
            }
        }
        // Remove redundant edges
        let queue_nodes_topo_sorted =
            bevy_utils::petgraph::algo::toposort(&queue_graph, None).unwrap();
        let (queue_nodes_tred_list, _) =
            bevy_utils::petgraph::algo::tred::dag_to_toposorted_adjacency_list::<_, u32>(
                &queue_graph,
                &queue_nodes_topo_sorted,
            );
        let (reduction, _) = bevy_utils::petgraph::algo::tred::dag_transitive_reduction_closure(
            &queue_nodes_tred_list,
        );

        // Step 1.5: Disperse semaphores
        // Step 1.5.1: Assign semaphore IDs
        let mut binary_semaphore_id = 0;
        let mut timeline_semaphore_id = 0;
        let mut queue_graph_reduced =
            bevy_utils::petgraph::graphmap::DiGraphMap::<u32, QueueGraphEdge>::new();
        for edge in reduction.edge_references() {
            let src = queue_nodes_topo_sorted[edge.source() as usize];
            let dst = queue_nodes_topo_sorted[edge.target() as usize];
            let config_src = &queue_graph_nodes[src as usize];
            let config_dst = &queue_graph_nodes[dst as usize];
            let edge_ref = queue_graph.edge_weight(src, dst).unwrap();
            let mut new_edge = QueueGraphEdge {
                config: edge_ref.config.clone(),
                semaphore_id: None,
            };
            if config_src.force_binary_semaphore || config_dst.force_binary_semaphore {
                new_edge.semaphore_id =
                    Some(QueueGraphEdgeSemaphoreType::Binary(binary_semaphore_id));
                binary_semaphore_id += 1;
            } else {
                new_edge.semaphore_id =
                    Some(QueueGraphEdgeSemaphoreType::Timeline(timeline_semaphore_id));
                timeline_semaphore_id += 1;
            }
            queue_graph_reduced.add_edge(edge.source(), edge.target(), new_edge);
        }
        self.queue_graph = queue_graph_reduced;
        self.queue_graph_nodes = queue_graph_nodes;
        self.num_binary_semaphores = binary_semaphore_id;
        let device = world.resource::<Device>();
        let timeline_semaphores: Vec<_> = {
            (0..timeline_semaphore_id)
                .map(|_| Arc::new(TimelineSemaphore::new(device.clone()).unwrap()))
                .collect()
        };
        // distribute timeline semaphores
        for (i, queue_node) in self.queue_graph_nodes.iter().enumerate() {
            let mut signals: Vec<TimelineSemaphoreOp> = Vec::new();
            let mut binary_signals: Vec<BinarySemaphoreOp> = Vec::new();
            self.queue_graph
                .edges_directed(i as u32, Outgoing)
                .for_each(|(_, _, edge)| match edge.semaphore_id.as_ref().unwrap() {
                    QueueGraphEdgeSemaphoreType::Binary(u32) => {
                        binary_signals.push(BinarySemaphoreOp {
                            index: *u32,
                            access: edge.config.signal.clone(),
                        })
                    }
                    QueueGraphEdgeSemaphoreType::Timeline(u32) => {
                        signals.push(TimelineSemaphoreOp {
                            semaphore: timeline_semaphores[*u32 as usize].clone(),
                            access: edge.config.signal.clone(),
                        })
                    }
                });
            let mut waits: Vec<TimelineSemaphoreOp> = Vec::new();
            let mut binary_waits: Vec<BinarySemaphoreOp> = Vec::new();
            self.queue_graph
                .edges_directed(i as u32, Incoming)
                .for_each(|(_, _, edge)| match edge.semaphore_id.as_ref().unwrap() {
                    QueueGraphEdgeSemaphoreType::Binary(u32) => {
                        binary_waits.push(BinarySemaphoreOp {
                            index: *u32,
                            access: edge.config.wait.clone(),
                        })
                    }
                    QueueGraphEdgeSemaphoreType::Timeline(u32) => {
                        waits.push(TimelineSemaphoreOp {
                            semaphore: timeline_semaphores[*u32 as usize].clone(),
                            access: edge.config.wait.clone(),
                        });
                    }
                });
            for i in queue_node.nodes.iter() {
                if *i == queue_node.queue_node_index {
                    continue;
                }
                if signals.is_empty() {
                    // Make sure we signal at least one timeline semaphore.
                    let device = world.resource::<Device>();
                    signals.push(TimelineSemaphoreOp {
                        access: Default::default(),
                        semaphore: Arc::new(TimelineSemaphore::new(device.clone()).unwrap()),
                    })
                }
                // Set config for all nodes in system
                let node = &mut graph.systems[*i];
                node.get_mut().unwrap().configurate(
                    &mut RenderSystemInitialState {
                        queue: queue_node.selected_queue,
                        timeline_signal: signals[0].semaphore.clone(),
                    },
                    world,
                );
            }
            let system = &mut graph.systems[queue_node.queue_node_index];
            system.get_mut().unwrap().configurate(
                &mut QueueSystemInitialState {
                    queue: queue_node.selected_queue,
                    timeline_signals: signals,
                    timeline_waits: waits,
                    binary_signals,
                    binary_waits,
                },
                world,
            );
        }
        let device: &Device = world.resource();
        let binary_semaphores =
            RenderSystemsBinarySemaphoreTracker::new(device.clone(), binary_semaphore_id as usize);
        world.insert_resource(binary_semaphores);

        // Step 2: inside each queue, insert pipeline barriers.
        for queue_node in self.queue_graph_nodes.iter() {
            if queue_node.is_queue_op {
                continue;
            }
            // TODO: do a DFS, detect system meta compatibilities, then group them into stages.
            let nodes = BTreeSet::from_iter(queue_node.nodes.iter());
            let mut queue_node_graph = DiGraphMap::<usize, ()>::new();
            // Produce a subgraph
            for i in queue_node.nodes.iter() {
                queue_node_graph.add_node(*i);
                for next in render_graph.neighbors(*i) {
                    if nodes.contains(&next) {
                        queue_node_graph.add_edge(*i, next, ());
                    }
                }
            }
            let mut heap: Vec<usize> = queue_node_graph
                .nodes()
                .filter(|node| {
                    queue_node_graph
                        .neighbors_directed(*node, Incoming)
                        .next()
                        .is_none()
                })
                .collect();
            let mut current_access = Access::new();
            let mut next_stage_heap: Vec<usize> = Vec::new();
            let mut current_stage: Vec<usize> = Vec::new();
            let mut all_stages: Vec<Vec<usize>> = Vec::new();
            while let Some(node) = heap.pop() {
                let system = graph.systems[node].get().unwrap();
                let access = system.archetype_component_access();
                if current_access.is_compatible(access) {
                    current_access.extend(access);
                    let revealed_nodes = queue_node_graph
                        .neighbors_directed(node, Outgoing)
                        .filter(|node| {
                            let mut iter = queue_node_graph.neighbors_directed(*node, Incoming);
                            iter.next().unwrap();
                            iter.next().is_none()
                        });
                    heap.extend(revealed_nodes);
                    queue_node_graph.remove_node(node);
                    current_stage.push(node);
                } else {
                    // defer
                    next_stage_heap.push(node);
                }
                if heap.is_empty() {
                    // Try enter the next stage
                    std::mem::swap(&mut heap, &mut next_stage_heap);
                    all_stages.push(std::mem::take(&mut current_stage));
                    current_access = Access::new();
                }
            }
            if !current_stage.is_empty() {
                all_stages.push(std::mem::take(&mut current_stage));
            }
            println!("{:?}", all_stages);
            let mut prev_stage: Vec<usize> = Vec::new();
            for stage in all_stages.into_iter() {
                // Create a pipeline flush system
                let id_num = graph.systems.len();
                let id = NodeId::System(id_num);
                let mut system: BoxedSystem = match queue_node.queue_type {
                    QueueType::Graphics => Box::new(IntoSystem::into_system(
                        crate::ecs::InsertPipelineBarrier::<'g'>::new(),
                    )),
                    QueueType::Compute => Box::new(IntoSystem::into_system(
                        crate::ecs::InsertPipelineBarrier::<'c'>::new(),
                    )),
                    QueueType::Transfer => Box::new(IntoSystem::into_system(
                        crate::ecs::InsertPipelineBarrier::<'t'>::new(),
                    )),
                    _ => unimplemented!(),
                };

                let mut barrier_producers: Vec<_> = stage
                    .iter()
                    .map(|&i| {
                        graph.systems[i]
                            .config
                            .get_mut::<RenderSystemConfig>()
                            .unwrap()
                            .barrier_producer
                            .take()
                            .unwrap()
                    })
                    .collect();
                system.configurate(&mut barrier_producers, world);
                assert!(barrier_producers.is_empty());
                system.initialize(world);
                graph
                    .systems
                    .push(SystemNode::new(system, Default::default()));
                graph.system_conditions.push(Vec::new());
                graph.ambiguous_with_all.insert(id);

                for i in prev_stage.iter() {
                    dependency_flattened.add_edge(NodeId::System(*i), id, ());
                }
                for i in stage.iter() {
                    dependency_flattened.add_edge(id, NodeId::System(*i), ());
                }
                let _ = std::mem::replace(&mut prev_stage, stage);
            }
        }
        Ok(())
    }
}
