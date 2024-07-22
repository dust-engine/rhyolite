use std::{
    collections::BTreeSet,
    ffi::CString,
    sync::{Arc, Mutex},
};

use bevy::ptr::OwningPtr;
use bevy::{
    ecs::{
        schedule::{NodeId, ScheduleBuildPass, SystemNode},
        system::{BoxedSystem, IntoSystem},
        world::World,
    },
    utils::ConfigMap,
};

use crate::{
    debug::DebugObject,
    ecs::{
        BarrierProducerOutConfig, DefaultCommandPool, PerFrame, QueueSubmissionInfo,
        RenderSystemInitialState, ResInstanceConfig,
    },
    queue::{QueueRef, Queues},
    semaphore::TimelineSemaphore,
    Device,
};
use petgraph::{
    graphmap::DiGraphMap,
    visit::{Dfs, EdgeRef, IntoEdgeReferences, Walker},
    Direction::{self, Incoming, Outgoing},
};
use smallvec::SmallVec;

use super::RenderSystemConfig;

pub struct RenderSystemPass {
    queue_graph: petgraph::graphmap::DiGraphMap<u32, ()>,
    queue_graph_nodes: Vec<QueueGraphNodeMeta>,
}

impl RenderSystemPass {
    pub fn new() -> Self {
        Self {
            queue_graph: Default::default(),
            queue_graph_nodes: Vec::new(),
        }
    }
}

#[derive(Debug)]
enum QueueGraphEdgeSemaphoreType {
    Binary(u32),
    Timeline(u32),
}

struct SubmitSystemMarker;

#[derive(Clone)]
struct QueueGraphNodeMeta {
    is_queue_op: bool,
    /// for the queue node, this is the system id of the "queue op" system?
    queue_node_index: usize,
    flush_node_index: usize,
    nodes: Vec<usize>,
    selected_queue: QueueRef,
    submission_info: Arc<Mutex<QueueSubmissionInfo>>,
    /// Indexed by QueueRef
    prev_stage_submission_info: SmallVec<[Option<Arc<Mutex<QueueSubmissionInfo>>>; 4]>,
}

impl ScheduleBuildPass for RenderSystemPass {
    type EdgeOptions = ();

    type NodeOptions = RenderSystemConfig;

    fn add_dependency(
        &mut self,
        _from: bevy::ecs::schedule::NodeId,
        _to: bevy::ecs::schedule::NodeId,
        _options: Option<&Self::EdgeOptions>,
    ) {
    }

    type CollapseSetIterator = std::iter::Empty<(NodeId, NodeId)>;

    fn collapse_set(
        &mut self,
        _set: bevy::ecs::schedule::NodeId,
        _systems: &[bevy::ecs::schedule::NodeId],
        _dependency_flattened: &petgraph::prelude::GraphMap<
            bevy::ecs::schedule::NodeId,
            (),
            petgraph::prelude::Directed,
        >,
    ) -> Self::CollapseSetIterator {
        std::iter::empty()
    }

    fn build(
        &mut self,
        world: &mut World,
        graph: &mut bevy::ecs::schedule::ScheduleGraph,
        dependency_flattened: &mut petgraph::prelude::GraphMap<
            bevy::ecs::schedule::NodeId,
            (),
            petgraph::prelude::Directed,
        >,
    ) -> Result<(), bevy::ecs::schedule::ScheduleBuildError> {
        let mut render_graph = petgraph::graphmap::DiGraphMap::<usize, ()>::new();

        struct RenderGraphNodeMeta {
            force_binary_semaphore: bool,
            is_queue_op: bool,

            selected_queue: QueueRef,
            stage_index: u32,
            queue_graph_node: u32,
        }
        let mut render_graph_meta: Vec<Option<RenderGraphNodeMeta>> =
            (0..graph.systems.len()).map(|_| None).collect();
        let queue_router = world.resource::<Queues>().clone();

        // Step 1: Queue coloring.
        // Step 1.1: Generate render graph
        let mut num_queues: u32 = 0;
        for node in dependency_flattened.nodes() {
            let NodeId::System(node_id) = node else {
                continue;
            };
            let system = &graph.systems[node_id];
            let Some(render_system_config) = system.config.get::<RenderSystemConfig>() else {
                continue; // not a render system
            };
            let selected_queue = queue_router
                .with_caps(
                    render_system_config.required_queue_flags,
                    render_system_config.preferred_queue_flags,
                )
                .unwrap();
            num_queues = num_queues.max(selected_queue.index + 1);

            render_graph_meta[node_id] = Some(RenderGraphNodeMeta {
                force_binary_semaphore: render_system_config.force_binary_semaphore,
                is_queue_op: render_system_config.is_queue_op,

                selected_queue,
                stage_index: u32::MAX,
                queue_graph_node: 0,
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
        let mut cmd_op_colors: Vec<(Vec<usize>, Vec<Vec<usize>>)> =
            vec![Default::default(); num_queues as usize];
        let mut queue_op_colors: Vec<(Option<usize>, Vec<usize>)> =
            vec![Default::default(); num_queues as usize];
        let mut tiny_graph = petgraph::graphmap::DiGraphMap::<(QueueRef, bool), ()>::new();
        let mut current_graph = render_graph.clone();
        let mut heap_next_stage: Vec<usize> = Vec::new(); // nodes to be deferred to the next stage
        while let Some(node) = heap.pop() {
            let meta = render_graph_meta[node].as_ref().unwrap();
            let node_color = meta.selected_queue;
            let is_queue_op = meta.is_queue_op;
            let mut should_defer = false;
            if meta.is_queue_op
                && queue_op_colors[meta.selected_queue.index as usize]
                    .0
                    .is_some()
            {
                // A queue op of this color was already queued
                should_defer = true;
            }
            for parent in render_graph.neighbors_directed(node, Incoming) {
                let parent_meta = render_graph_meta[parent].as_ref().unwrap();
                if (parent_meta.selected_queue, parent_meta.is_queue_op)
                    != (node_color, meta.is_queue_op)
                {
                    let start = (parent_meta.selected_queue, parent_meta.is_queue_op);
                    let end = (node_color, meta.is_queue_op);
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
                if is_queue_op {
                    assert!(queue_op_colors[meta.selected_queue.index as usize]
                        .0
                        .is_none());
                    queue_op_colors[meta.selected_queue.index as usize].0 = Some(node);
                } else {
                    cmd_op_colors[meta.selected_queue.index as usize]
                        .0
                        .push(node);
                }

                for parent in render_graph.neighbors_directed(node, Incoming) {
                    // Update the tiny graph.
                    let parent_meta = render_graph_meta[parent].as_mut().unwrap();
                    if parent_meta.selected_queue != node_color
                        && parent_meta.stage_index == stage_index
                    {
                        let start = (parent_meta.selected_queue, parent_meta.is_queue_op);
                        let end = (node_color, is_queue_op);
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
                for (queue_node_buffer, stages) in cmd_op_colors.iter_mut() {
                    if !queue_node_buffer.is_empty() {
                        // Flush remaining nodes
                        stages.push(std::mem::take(queue_node_buffer));
                    }
                }
                for (queue_node_buffer, stages) in queue_op_colors.iter_mut() {
                    if let Some(a) = queue_node_buffer.take() {
                        // Flush remaining nodes
                        stages.push(a);
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

        let mut queue_graph = petgraph::graphmap::DiGraphMap::<u32, ()>::new();
        let mut queue_graph_nodes = Vec::<QueueGraphNodeMeta>::new();

        // Flush all colors
        let device = world.resource::<Device>().clone();
        for (queue_node_buffer, stages) in queue_op_colors.iter_mut() {
            if let Some(a) = queue_node_buffer.take() {
                // Flush remaining nodes
                stages.push(a);
            }
            for stage in stages.iter_mut() {
                let queue_graph_node = queue_graph_nodes.len() as u32;
                queue_graph.add_node(queue_graph_node);
                let meta = render_graph_meta[*stage].as_mut().unwrap();
                meta.queue_graph_node = queue_graph_node;
                queue_graph_nodes.push(QueueGraphNodeMeta {
                    nodes: vec![*stage],
                    is_queue_op: true,
                    queue_node_index: 0,
                    flush_node_index: 0,
                    selected_queue: meta.selected_queue,
                    submission_info: Arc::new(Mutex::new(if meta.force_binary_semaphore {
                        QueueSubmissionInfo::default()
                    } else {
                        QueueSubmissionInfo {
                            signal_semaphore: Some(Arc::new(
                                TimelineSemaphore::new(device.clone()).unwrap().with_name(
                                    CString::new(format!(
                                        "TimelineSemaphore for QueueOp {:?} Stage {}",
                                        meta.selected_queue, meta.stage_index
                                    ))
                                    .unwrap()
                                    .as_c_str(),
                                ),
                            )),
                            signal_semaphore_value: 1,
                            ..Default::default()
                        }
                    })),
                    prev_stage_submission_info: SmallVec::from_iter(
                        std::iter::repeat(None).take(num_queues as usize),
                    ),
                });
            }
        }
        for (queue_node_buffer, stages) in cmd_op_colors.iter_mut() {
            if !queue_node_buffer.is_empty() {
                // Flush remaining nodes
                stages.push(std::mem::take(queue_node_buffer));
            }
            for stage in stages.iter_mut() {
                let mut queue_graph_node_info: Option<QueueGraphNode> = None;
                let queue_graph_node = queue_graph_nodes.len() as u32;
                queue_graph.add_node(queue_graph_node);

                for node in stage.iter() {
                    let meta = render_graph_meta[*node].as_mut().unwrap();
                    meta.queue_graph_node = queue_graph_node;
                    if let Some(queue_graph_node_info) = &queue_graph_node_info {
                        // All nodes in here shall have the same stage index and selected queue
                        assert_eq!(queue_graph_node_info.stage_index, meta.stage_index);
                        assert_eq!(queue_graph_node_info.queue, meta.selected_queue);
                    }
                    queue_graph_node_info.get_or_insert(QueueGraphNode {
                        stage_index: meta.stage_index,
                        queue: meta.selected_queue,
                    });
                }
                if let Some(queue_graph_node_info) = queue_graph_node_info {
                    queue_graph_nodes.push(QueueGraphNodeMeta {
                        nodes: std::mem::take(stage),
                        is_queue_op: false,
                        queue_node_index: 0,
                        flush_node_index: 0,
                        selected_queue: queue_graph_node_info.queue,
                        submission_info: Arc::new(Mutex::new(QueueSubmissionInfo {
                            signal_semaphore: Some(Arc::new(
                                TimelineSemaphore::new(device.clone()).unwrap().with_name(
                                    CString::new(format!(
                                        "TimelineSemaphore for QueueOp {:?} Stage {}",
                                        queue_graph_node_info.queue,
                                        queue_graph_node_info.stage_index
                                    ))
                                    .unwrap()
                                    .as_c_str(),
                                ),
                            )),
                            signal_semaphore_value: 1,
                            ..Default::default()
                        })),
                        prev_stage_submission_info: SmallVec::from_iter(
                            std::iter::repeat(None).take(num_queues as usize),
                        ),
                    });
                }
            }
        }
        // queue graph connectivity
        for (from, to, _) in render_graph.all_edges() {
            let from_meta = render_graph_meta[from].as_ref().unwrap();
            let to_meta = render_graph_meta[to].as_ref().unwrap();

            if from_meta.queue_graph_node != to_meta.queue_graph_node {
                queue_graph.add_edge(from_meta.queue_graph_node, to_meta.queue_graph_node, ());
            }
        }
        // populate prev_stage_submission_info
        for (from, to, _meta) in queue_graph.all_edges() {
            let from = &queue_graph_nodes[from as usize];
            if from.is_queue_op {
                continue;
            }
            let submission_info = from.submission_info.clone();
            let from_queue_type = from.selected_queue;

            let to = &mut queue_graph_nodes[to as usize];
            assert!(to.prev_stage_submission_info[from_queue_type.index as usize].is_none());
            to.prev_stage_submission_info[from_queue_type.index as usize] = Some(submission_info);
        }

        // Step 1.4: Insert queue submission systems
        let queue_id_to_component_id = (0..num_queues)
            .map(|i| unsafe {
                let id = world.init_instanced_resource::<PerFrame<DefaultCommandPool>>();
                let queue = &queue_router[i];
                OwningPtr::make(
                    PerFrame::<DefaultCommandPool>::new(|_| {
                        DefaultCommandPool::new(device.clone(), queue.family)
                    }),
                    |value| {
                        world.insert_resource_by_id(id, value);
                    },
                );
                id
            })
            .collect::<Vec<_>>();

        for (_i, queue_node) in queue_graph_nodes.iter_mut().enumerate() {
            if queue_node.is_queue_op {
                assert!(queue_node.nodes.len() == 1);
                queue_node.queue_node_index = queue_node.nodes[0];
                queue_node.flush_node_index = queue_node.nodes[0];
                continue;
            }
            let id_num = graph.systems.len();
            let id: NodeId = NodeId::System(id_num);
            let mut system: BoxedSystem =
                Box::new(IntoSystem::into_system(crate::ecs::flush_system_graph));

            let mut config = RenderSystemInitialState {
                queue: queue_node.selected_queue,
                queue_submission_info: queue_node.submission_info.clone(),
                prev_stage_queue_submission_info: queue_node.prev_stage_submission_info.clone(),
            };
            system.initialize(world);
            system.configurate(&mut config, world);

            system.configurate(
                &mut ResInstanceConfig::for_type::<PerFrame<DefaultCommandPool>>(
                    queue_id_to_component_id[queue_node.selected_queue.index as usize],
                ),
                world,
            );

            let config_map = ConfigMap::new();
            graph.systems.push(SystemNode::new(system, config_map));
            graph.system_conditions.push(Vec::new());
            graph.ambiguous_with_all.insert(id);

            queue_node.flush_node_index = id_num;

            let id_num = graph.systems.len();
            let id: NodeId = NodeId::System(id_num);
            let mut system: BoxedSystem =
                Box::new(IntoSystem::into_system(crate::ecs::submit_system_graph));

            let mut config = RenderSystemInitialState {
                queue: queue_node.selected_queue,
                queue_submission_info: queue_node.submission_info.clone(),
                prev_stage_queue_submission_info: queue_node.prev_stage_submission_info.clone(),
            };
            system.initialize(world);
            system.configurate(&mut config, world);

            system.configurate(
                &mut ResInstanceConfig::for_type::<PerFrame<DefaultCommandPool>>(
                    queue_id_to_component_id[queue_node.selected_queue.index as usize],
                ),
                world,
            );
            let mut config_map = ConfigMap::new();
            config_map.insert::<SubmitSystemMarker>(SubmitSystemMarker);
            graph.systems.push(SystemNode::new(system, config_map));
            graph.system_conditions.push(Vec::new());
            graph.ambiguous_with_all.insert(id);

            queue_node.queue_node_index = id_num;
        }
        // Connect queue submission systems with parents and children
        for (i, queue_node) in queue_graph_nodes.iter().enumerate() {
            if queue_node.is_queue_op {
                continue;
            }
            let submit_node = NodeId::System(queue_node.queue_node_index);
            let flush_node = NodeId::System(queue_node.flush_node_index);
            dependency_flattened.add_edge(flush_node, submit_node, ());
            for parent_command_node in queue_node.nodes.iter() {
                // After all nodes within the corrent queue node
                let command_node_id = NodeId::System(*parent_command_node);
                dependency_flattened.add_edge(command_node_id, flush_node, ());
                dependency_flattened.add_edge(command_node_id, submit_node, ());
            }
            for child in queue_graph.neighbors_directed(i as u32, Outgoing) {
                let child_meta = &queue_graph_nodes[child as usize];
                if child_meta.is_queue_op {
                    assert_eq!(child_meta.queue_node_index, child_meta.flush_node_index);
                    dependency_flattened.add_edge(
                        flush_node,
                        NodeId::System(child_meta.queue_node_index),
                        (),
                    );
                    dependency_flattened.add_edge(
                        submit_node,
                        NodeId::System(child_meta.queue_node_index),
                        (),
                    );
                } else {
                    for child_command_node in child_meta.nodes.iter() {
                        let command_node_id = NodeId::System(*child_command_node);
                        dependency_flattened.add_edge(flush_node, command_node_id, ());
                        dependency_flattened.add_edge(command_node_id, submit_node, ());
                    }
                    dependency_flattened.add_edge(
                        submit_node,
                        NodeId::System(child_meta.flush_node_index),
                        (),
                    );
                }
            }
        }
        // Remove redundant edges
        let queue_nodes_topo_sorted = petgraph::algo::toposort(&queue_graph, None).unwrap();
        let (queue_nodes_tred_list, _) = petgraph::algo::tred::dag_to_toposorted_adjacency_list::<
            _,
            u32,
        >(&queue_graph, &queue_nodes_topo_sorted);
        let (reduction, _) =
            petgraph::algo::tred::dag_transitive_reduction_closure(&queue_nodes_tred_list);

        // Step 1.5: Disperse semaphores
        // Step 1.5.1: Assign semaphore IDs
        let mut queue_graph_reduced = petgraph::graphmap::DiGraphMap::<u32, ()>::new();
        for edge in reduction.edge_references() {
            let src = queue_nodes_topo_sorted[edge.source() as usize];
            let dst = queue_nodes_topo_sorted[edge.target() as usize];
            let config_src = &queue_graph_nodes[src as usize];
            let config_dst = &queue_graph_nodes[dst as usize];
            let edge_ref = queue_graph.edge_weight(src, dst).unwrap();
            queue_graph_reduced.add_edge(src, dst, ());
        }
        self.queue_graph = queue_graph_reduced;
        self.queue_graph_nodes = queue_graph_nodes;

        // distribute timeline semaphores
        for (i, queue_node) in self.queue_graph_nodes.iter().enumerate() {
            for i in queue_node.nodes.iter() {
                if *i == queue_node.queue_node_index {
                    continue;
                }
                // Set config for all nodes in system
                let node = &mut graph.systems[*i];
                node.get_mut().unwrap().configurate(
                    &mut RenderSystemInitialState {
                        queue: queue_node.selected_queue,
                        queue_submission_info: queue_node.submission_info.clone(),
                        prev_stage_queue_submission_info: queue_node
                            .prev_stage_submission_info
                            .clone(),
                    },
                    world,
                );
                node.get_mut().unwrap().configurate(
                    &mut ResInstanceConfig::for_type::<PerFrame<DefaultCommandPool>>(
                        queue_id_to_component_id[queue_node.selected_queue.index as usize],
                    ),
                    world,
                );
            }
        }

        // Step 2: inside each queue, insert pipeline barriers.
        for (queue_node_i, queue_node) in self.queue_graph_nodes.iter().enumerate() {
            let all_stages = if queue_node.is_queue_op {
                vec![vec![queue_node.queue_node_index]]
            } else {
                // do a DFS, detect system meta compatibilities, then group them into stages.
                let nodes = BTreeSet::from_iter(queue_node.nodes.iter());
                let mut queue_node_graph = DiGraphMap::<usize, ()>::new();
                // Produce a subgraph
                for i in queue_node.nodes.iter() {
                    queue_node_graph.add_node(*i);
                    let mut neighbors = dependency_flattened
                        .neighbors(NodeId::System(*i))
                        .map(|next| (*i, next.index()))
                        .collect::<Vec<_>>();
                    while let Some((from, to)) = neighbors.pop() {
                        if nodes.contains(&to) {
                            queue_node_graph.add_edge(from, to, ()); // Note: In here, we also need to connect indirect dependencies.
                        } else {
                            let system = &graph.systems[to];
                            if system.config.get::<RenderSystemConfig>().is_none() {
                                // not a render system. Add its childrens too.
                                neighbors.extend(
                                    dependency_flattened
                                        .neighbors(NodeId::System(to))
                                        .map(|to| (*i, to.index())),
                                );
                            }
                            continue;
                        };
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
                let mut next_stage_heap: Vec<usize> = Vec::new();
                let mut current_stage: Vec<usize> = Vec::new();
                let mut all_stages: Vec<Vec<usize>> = Vec::new();
                while let Some(node) = heap.pop() {
                    let revealed_nodes = queue_node_graph
                        .neighbors_directed(node, Outgoing)
                        .filter(|node| {
                            let mut iter = queue_node_graph.neighbors_directed(*node, Incoming);
                            iter.next().unwrap();
                            iter.next().is_none()
                        });
                    next_stage_heap.extend(revealed_nodes);
                    queue_node_graph.remove_node(node);
                    current_stage.push(node);
                    if heap.is_empty() {
                        // Try enter the next stage
                        std::mem::swap(&mut heap, &mut next_stage_heap);
                        all_stages.push(std::mem::take(&mut current_stage));
                    }
                }
                if !current_stage.is_empty() {
                    all_stages.push(std::mem::take(&mut current_stage));
                }
                all_stages
            };
            for (i, stage) in all_stages.into_iter().enumerate() {
                // Create a pipeline flush system
                let id_num = graph.systems.len();
                let id = NodeId::System(id_num);

                let barrier_producers: Vec<_> = stage
                    .iter()
                    .filter_map(|&i| {
                        let Some(barrier_producer_config) = graph.systems[i]
                            .config
                            .get_mut::<RenderSystemConfig>()
                            .unwrap()
                            .barrier_producer_config
                            .take() else {
                            return None;
                            };
                        let mut config = Some(BarrierProducerOutConfig {
                            barrier_producer_output_cell: barrier_producer_config.barrier_producer_output_cell,
                            barrier_producer_output_type: barrier_producer_config.barrier_producer_output_type,
                        });
                        graph.systems[i].get_mut().unwrap().configurate(&mut config, world);
                        if let Some(config) = config {
                            // If we still have the config, that means nobody took it.
                            // In this case, we need to drop the Arc so that it doesn't leak.
                            if !config.barrier_producer_output_cell.is_null() {
                                unsafe { Arc::from_raw(config.barrier_producer_output_cell) };
                                tracing::warn!("Barrier producer output was not consumed by the render system.");
                            }
                        }

                        Some(barrier_producer_config.barrier_producer)
                    })
                    .collect();
                if barrier_producers.is_empty() {
                    continue;
                }
                let mut system: BoxedSystem = Box::new(IntoSystem::into_system(
                    crate::ecs::InsertPipelineBarrier::new(),
                ));

                let mut config = crate::ecs::InsertPipelineBarrierConfig {
                    producers: barrier_producers,
                    record_to_next: i == 0 && !queue_node.is_queue_op,
                };
                // This is fine to be before .initialize, because InsertPipelineBarrierConfig was sent to the system, not the system configs
                system.configurate(&mut config, world);
                assert!(config.producers.is_empty());
                system.initialize(world);
                let mut config = RenderSystemInitialState {
                    queue: queue_node.selected_queue,
                    queue_submission_info: queue_node.submission_info.clone(),
                    prev_stage_queue_submission_info: queue_node.prev_stage_submission_info.clone(),
                };
                system.configurate(&mut config, world);
                system.configurate(
                    &mut ResInstanceConfig::for_type::<PerFrame<DefaultCommandPool>>(
                        queue_id_to_component_id[queue_node.selected_queue.index as usize],
                    ),
                    world,
                );

                graph
                    .systems
                    .push(SystemNode::new(system, Default::default()));
                graph.system_conditions.push(Vec::new());
                graph.ambiguous_with_all.insert(id);

                for i in stage.iter() {
                    for parent in dependency_flattened
                        .neighbors_directed(NodeId::System(*i), Direction::Incoming)
                        .collect::<Vec<_>>()
                    {
                        let is_submit_system = graph.systems[parent.index()]
                            .config
                            .has::<SubmitSystemMarker>();
                        if !is_submit_system {
                            // skip if the parent is a submit system we just added.
                            dependency_flattened.add_edge(parent, id, ());
                        }
                    }
                    dependency_flattened.add_edge(id, NodeId::System(*i), ());
                }
            }
        }
        Ok(())
    }
}
