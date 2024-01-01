use std::collections::BTreeSet;

use bevy_ecs::{
    component::ComponentId,
    schedule::{NodeId, ReportCycles, ScheduleBuildPass},
    world::World,
};
use bevy_utils::petgraph::graphmap::NodeTrait;

use crate::{ecs::QueueAssignment, queue::QueuesRouter};

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
            config: RenderSystemConfig
        }
        let mut render_graph_meta: Vec<Option<RenderGraphNodeMeta>> = vec![None; graph.systems.len()];
        
        // Step 1: Queue coloring.
        // Step 1.1: Generate render graph
        for node in dependency_flattened.nodes() {
            let NodeId::System(node_id) = node else {
                continue;
            };
            let system = graph.get_system_at(node).unwrap();
            let Some(render_system_config) = system.default_configs().and_then(|map| map.get::<RenderSystemConfig>()) else {
                continue; // not a render system
            };
            render_graph_meta[node_id] = Some(RenderGraphNodeMeta {
                config: render_system_config.clone()
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

        // Step 2.2: Agressively merge nodes
        // queue_nodes[queue_index] = list of nodes on that queue
        let queue_router = world.resource::<QueuesRouter>();
        let mut queue_nodes: Vec<Vec<usize>> = Vec::default();
        let topo = bevy_utils::petgraph::algo::toposort(&render_graph, None).unwrap();
        for node in topo {
            let meta = render_graph_meta[node].as_ref().unwrap();
            match meta.config.queue {
                QueueAssignment::MinOverhead(queue) | QueueAssignment::MaxAsync(queue) => {
                    let selected_queue = queue_router.of_type(queue);
                    queue_nodes[selected_queue.0 as usize].push(node);
                },
                QueueAssignment::Manual(queue) => todo!(),
            }
        }
        println!("{:?}", queue_nodes);


        // Step 2: inside each queue, insert pipeline barriers.

        Ok(())
    }
}
