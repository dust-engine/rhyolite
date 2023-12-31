use std::collections::BTreeSet;

use bevy_ecs::{
    component::ComponentId,
    schedule::{NodeId, ReportCycles, ScheduleBuildPass},
    world::World,
};

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
        let topo = graph.topsort_graph(dependency_flattened, ReportCycles::Dependency)?;

        let mut stage_access = BTreeSet::<ComponentId>::new();
        let registry = world.get_resource_or_insert_with(RenderResRegistry::default);
        let registry = registry.as_ref();

        // Step 1: Queue coloring.

        // Step 2: inside each queue, insert pipeline barriers.

        Ok(())
    }
}
