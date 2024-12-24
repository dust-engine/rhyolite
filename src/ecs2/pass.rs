use std::{collections::BTreeMap, sync::Arc};

use ash::vk;
use bevy::{
    ecs::{
        component::ComponentId,
        schedule::{NodeId, ScheduleBuildError, ScheduleBuildPass, ScheduleGraph},
        world::World,
    },
    prelude::{IntoSystem, System},
};

use petgraph::{
    graphmap::GraphMap,
    visit::{EdgeRef, IntoEdgeReferences},
    Directed,
};

use crate::{
    command::Timeline,
    ecs2::{system::TimelineDependencies, IntoRenderSystem},
    QueueInner,
};

use super::system::{RenderSystemIdentifierConfig, RenderSystemSharedState};

pub struct RenderSystemsPass {}
impl RenderSystemsPass {
    pub fn new() -> Self {
        Self {}
    }
}

impl RenderSystemsPass {
    fn add_system<Marker, T: IntoSystem<(), (), Marker>>(
        &mut self,
        graph: &mut ScheduleGraph,
        world: &mut World,
        system: T,
    ) -> NodeId {
        let id = NodeId::System(graph.systems.len());
        let mut system: T::System = IntoSystem::into_system(system);
        system.initialize(world);
        let mut configs = Default::default();
        system.default_configs(&mut configs);

        graph.systems.push(bevy::ecs::schedule::SystemNode::new(
            Box::new(system),
            configs,
        ));
        graph.system_conditions.push(Vec::new());

        // ignore ambiguities with auto sync points
        // They aren't under user control, so no one should know or care.
        graph.ambiguous_with_all.insert(id);

        id
    }
}

impl ScheduleBuildPass for RenderSystemsPass {
    type EdgeOptions = ();

    type NodeOptions = ();

    fn add_dependency(&mut self, from: NodeId, to: NodeId, options: Option<&Self::EdgeOptions>) {}

    type CollapseSetIterator = std::iter::Empty<(NodeId, NodeId)>;

    fn collapse_set(
        &mut self,
        set: NodeId,
        systems: &[NodeId],
        dependency_flattened: &GraphMap<NodeId, (), Directed>,
    ) -> Self::CollapseSetIterator {
        std::iter::empty()
    }

    fn build(
        &mut self,
        world: &mut World,
        graph: &mut ScheduleGraph,
        dependency_flattened: &mut GraphMap<NodeId, (), Directed>,
    ) -> Result<(), ScheduleBuildError> {
        let mut render_subgraph = dependency_flattened.clone();
        let mut queue_component_id_to_color: BTreeMap<ComponentId, u32> = BTreeMap::new();
        let mut color_to_queue_component_id: Vec<ComponentId> = Vec::new();
        // Build a subgraph from the full graph with only render nodes.
        // Indirect dependency (render node -> any node -> render node) are translated into dependencies in render node.
        for node in dependency_flattened.nodes() {
            let NodeId::System(node_id) = node else {
                // This should've been flattened out.
                panic!();
            };
            let system = &graph.systems[node_id];
            if let Some(config) = system.config.get::<RenderSystemIdentifierConfig>() {
                if !queue_component_id_to_color.contains_key(&config.queue_component_id) {
                    let color = color_to_queue_component_id.len() as u32;
                    queue_component_id_to_color.insert(config.queue_component_id, color);
                    color_to_queue_component_id.push(config.queue_component_id);
                }
                continue; // is a render system
            };

            // remove the node, and add all-pair dependencies between parents and child
            graph_remove_node_with_transitive_dependency(&mut render_subgraph, node);
        }

        // Next, we perform clustering
        let (queue_graph, queue_nodes) = graph_clustering(
            &render_subgraph,
            queue_component_id_to_color.len(),
            |node| {
                let NodeId::System(node_id) = node else {
                    // This should've been flattened out.
                    panic!();
                };
                let Some(render_system_config) = graph.systems[*node_id]
                    .config
                    .get::<RenderSystemIdentifierConfig>()
                else {
                    // Non render nodes shouldn't go into the render graph
                    panic!();
                };
                GraphClusteringNodeInfo {
                    color: *queue_component_id_to_color
                        .get(&render_system_config.queue_component_id)
                        .unwrap(),
                    is_standalone: render_system_config.is_standalone,
                }
            },
        );
        assert_eq!(queue_graph.node_count(), queue_nodes.len());

        let device: crate::Device = world.resource::<crate::Device>().clone();
        struct QueueNode {
            queue_component_id: ComponentId,
            info: GraphClusteringNodeInfo,
            nodes: Vec<NodeId>,

            /// The node responsible for actually performing the queue operation.
            /// Can be submission node or the standalone queue node itself.
            queue_node: NodeId,

            timeline_dependencies: TimelineDependencies,
        }
        // For each non standalone queue graph node, create prelude system and submission system.
        let mut queue_nodes: Vec<QueueNode> = queue_nodes
            .into_iter()
            .map(|mut n| {
                let queue_component_id = color_to_queue_component_id[n.info.color as usize];
                let queue_node = if n.info.is_standalone {
                    assert_eq!(n.nodes.len(), 1);
                    n.nodes[0]
                } else {
                    let prelude_system_id =
                        self.add_system(graph, world, crate::ecs2::system::prelude_system);
                    let submission_system_id = self.add_system(
                        graph,
                        world,
                        crate::ecs2::system::submission_system.with_queue(queue_component_id),
                    );
                    for node in n.nodes.iter() {
                        // for all nodes, they run before submission system and after prelude systems.
                        dependency_flattened.add_edge(prelude_system_id, *node, ());
                        dependency_flattened.add_edge(*node, submission_system_id, ());
                    }
                    n.nodes.push(prelude_system_id);
                    n.nodes.push(submission_system_id);
                    submission_system_id
                };
                QueueNode {
                    queue_node,
                    queue_component_id: queue_component_id,
                    info: n.info,
                    nodes: n.nodes,
                    timeline_dependencies: TimelineDependencies {
                        this: Arc::new(Timeline::new(device.clone()).unwrap()),
                        dependencies: Vec::new(),
                    },
                }
            })
            .collect();
        drop(color_to_queue_component_id);
        drop(queue_component_id_to_color);

        // Simplify the graph, then build dependency between queue nodes based on queue graph
        let queue_nodes_topo_sorted = petgraph::algo::toposort(&queue_graph, None).unwrap();
        let (queue_nodes_tred_list, _) = petgraph::algo::tred::dag_to_toposorted_adjacency_list::<
            _,
            u32,
        >(&queue_graph, &queue_nodes_topo_sorted);
        let (reduction, _) =
            petgraph::algo::tred::dag_transitive_reduction_closure(&queue_nodes_tred_list);
        for edge in reduction.edge_references() {
            let src = queue_nodes_topo_sorted[edge.source() as usize];
            let dst = queue_nodes_topo_sorted[edge.target() as usize];
            let start_node = &queue_nodes[src as usize];
            let end_node = &queue_nodes[dst as usize];
            dependency_flattened.add_edge(start_node.queue_node, end_node.queue_node, ());
            let timeline = start_node.timeline_dependencies.this.clone();
            let end_node = &mut queue_nodes[dst as usize];

            // TODO: allow stage flags
            end_node
                .timeline_dependencies
                .dependencies
                .push((timeline, vk::PipelineStageFlags2::ALL_COMMANDS));
        }

        // Distribute timeline semaphores
        for node in queue_nodes.iter_mut() {
            assert!(node.queue_node.is_system());
            graph.systems[node.queue_node.index()]
                .get_mut()
                .unwrap()
                .configurate(&mut node.timeline_dependencies, world);
        }

        // For each non standalone queue graph node, create shared states.
        // This runs after prelude and postlude systems so that these extra systems also get the shared states.
        for node_info in queue_nodes.iter() {
            if node_info.info.is_standalone {
                continue;
            }
            let queue_family = unsafe {
                world
                    .get_resource_by_id(node_info.queue_component_id)
                    .unwrap()
                    .deref::<QueueInner>()
                    .queue_family
            };
            let component_id = world.init_component_with_descriptor(
                bevy::ecs::component::ComponentDescriptor::new_resource::<RenderSystemSharedState>(
                ),
            );
            bevy::ptr::OwningPtr::make(
                RenderSystemSharedState::new(
                    device.clone(),
                    queue_family,
                    node_info.timeline_dependencies.this.clone(),
                ),
                |ptr| unsafe {
                    // SAFETY: component_id was just initialized and corresponds to resource of type R.
                    world.insert_resource_by_id(component_id, ptr);
                },
            );
            for node in node_info.nodes.iter() {
                let NodeId::System(node_id) = node else {
                    // This should've been flattened out.
                    panic!();
                };
                graph.systems[*node_id].get_mut().unwrap().configurate(
                    &mut super::system::RenderSystemInputConfig {
                        shared_state: component_id,
                        queue: node_info.queue_component_id,
                    },
                    world,
                );
            }
        }
        Ok(())
    }
}

fn graph_remove_node_with_transitive_dependency<N, Ty, S>(
    graph: &mut GraphMap<N, (), Ty, S>,
    node: N,
) where
    N: petgraph::graphmap::NodeTrait,
    Ty: petgraph::EdgeType,
    S: std::hash::BuildHasher,
{
    let parents: Vec<N> = graph
        .neighbors_directed(node, petgraph::Direction::Incoming)
        .collect();
    let children: Vec<N> = graph
        .neighbors_directed(node, petgraph::Direction::Outgoing)
        .collect();
    for parent in parents.iter() {
        for child in children.iter() {
            graph.add_edge(*parent, *child, ());
        }
    }
    graph.remove_node(node);
}

#[derive(PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Hash, Debug)]
struct GraphClusteringNodeInfo {
    color: u32,
    is_standalone: bool,
}
struct ClusteredNode<N> {
    info: GraphClusteringNodeInfo,
    nodes: Vec<N>,
}

/// Returns (clustered graph, clustered graph node info)
fn graph_clustering<N: petgraph::graphmap::NodeTrait + Clone>(
    render_graph: &GraphMap<N, (), Directed>,
    num_colors: usize,
    get_node_info: impl Fn(&N) -> GraphClusteringNodeInfo,
) -> (GraphMap<u32, (), Directed>, Vec<ClusteredNode<N>>) {
    let mut heap: Vec<N> = Vec::new(); // nodes with no incoming edges

    let mut node_stage_indexes: BTreeMap<N, usize> = BTreeMap::new();

    // First, find all nodes with no incoming edges
    for node in render_graph.nodes() {
        if render_graph
            .neighbors_directed(node, petgraph::Direction::Incoming)
            .next()
            .is_none()
        {
            // Has no incoming edges
            heap.push(node);
        }
    }
    let mut stage_index = 0;
    // (buffer, stages)
    let mut cmd_op_colors: Vec<(Vec<N>, Vec<Vec<N>>)> = vec![Default::default(); num_colors];
    let mut queue_op_colors: Vec<(Option<N>, Vec<N>)> = vec![Default::default(); num_colors];
    let mut tiny_graph = petgraph::graphmap::DiGraphMap::<GraphClusteringNodeInfo, ()>::new();
    let mut current_graph = render_graph.clone();
    let mut heap_next_stage: Vec<N> = Vec::new(); // nodes to be deferred to the next stage
    while let Some(node) = heap.pop() {
        let node_info = get_node_info(&node);
        let mut should_defer = false;
        if node_info.is_standalone && queue_op_colors[node_info.color as usize].0.is_some() {
            // A queue op of this color was already queued
            should_defer = true;
        }
        for parent in render_graph.neighbors_directed(node, petgraph::Direction::Incoming) {
            let parent_info = get_node_info(&parent);
            if parent_info != node_info {
                use petgraph::visit::Walker;
                let has_path = petgraph::visit::Dfs::new(&tiny_graph, node_info)
                    .iter(&tiny_graph)
                    .any(|x| x == parent_info);
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
            node_stage_indexes.insert(node, stage_index);
            if node_info.is_standalone {
                assert!(queue_op_colors[node_info.color as usize].0.is_none());
                queue_op_colors[node_info.color as usize].0 = Some(node);
            } else {
                cmd_op_colors[node_info.color as usize].0.push(node);
            }

            for parent in render_graph.neighbors_directed(node, petgraph::Direction::Incoming) {
                // Update the tiny graph.
                let parent_info = get_node_info(&parent);
                if parent_info.color != node_info.color
                    && *node_stage_indexes.get(&parent).unwrap() == stage_index
                {
                    tiny_graph.add_edge(parent_info, node_info, ());
                }
            }

            for child in current_graph.neighbors_directed(node, petgraph::Direction::Outgoing) {
                let mut other_parents =
                    current_graph.neighbors_directed(child, petgraph::Direction::Incoming);
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

    // Now, create the clustered graph.
    let mut clustered_graph = petgraph::graphmap::DiGraphMap::<u32, ()>::new();
    let mut clustered_graph_info: Vec<ClusteredNode<N>> = Vec::new();
    let mut node_to_clustered_nodes: BTreeMap<N, u32> = BTreeMap::new(); // mapping from render nodes to clustered nodes

    // Flush standalone nodes
    for (queue_node_buffer, stages) in queue_op_colors.iter_mut() {
        if let Some(a) = queue_node_buffer.take() {
            // Flush remaining nodes
            stages.push(a);
        }
        for stage in stages.iter_mut() {
            let clustered_node = clustered_graph.node_count() as u32;
            clustered_graph.add_node(clustered_node);
            clustered_graph_info.push(ClusteredNode {
                info: get_node_info(stage),
                nodes: vec![*stage],
            });
            node_to_clustered_nodes.insert(*stage, clustered_node);
        }
    }

    // Flush clustered nodes
    for (queue_node_buffer, mut stages) in cmd_op_colors.into_iter() {
        if !queue_node_buffer.is_empty() {
            // Flush remaining nodes
            stages.push(queue_node_buffer);
        }
        for stage in stages.into_iter() {
            let clustered_node = clustered_graph.node_count() as u32;
            clustered_graph.add_node(clustered_node);
            assert!(!stage.is_empty());
            let mut info: Option<GraphClusteringNodeInfo> = None;
            for node in stage.iter() {
                node_to_clustered_nodes.insert(*node, clustered_node);
                if let Some(info) = info {
                    assert_eq!(info, get_node_info(node));
                } else {
                    info = Some(get_node_info(node));
                }
            }

            clustered_graph_info.push(ClusteredNode {
                info: info.unwrap(),
                nodes: stage,
            });
        }
    }

    // clustered graph connectivity
    for (from, to, _) in render_graph.all_edges() {
        let from_clustered_node = *node_to_clustered_nodes.get(&from).unwrap();
        let to_clustered_node = *node_to_clustered_nodes.get(&to).unwrap();

        if from_clustered_node != to_clustered_node {
            clustered_graph.add_edge(from_clustered_node, to_clustered_node, ());
        }
    }

    (clustered_graph, clustered_graph_info)
}
