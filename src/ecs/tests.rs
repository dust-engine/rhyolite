use crate::queue::{QueueRef, QueuesRouter};

use super::*;
use bevy_app::Update;
use bevy_ecs::{
    schedule::{IntoSystemConfigs, Schedule},
    system::Resource,
    world::World,
};

#[derive(Resource)]
struct Image;

fn system_g1(_commands: RenderCommands<'g'>, _render: RenderResMut<Image>) {}
fn system_g2(_commands: RenderCommands<'g'>, _render: RenderRes<Image>) {}
fn system_g3(_commands: RenderCommands<'g'>, _render: RenderRes<Image>) {}
fn system_c1(_commands: RenderCommands<'c'>, _render: RenderRes<Image>) {}
fn system_c2(_commands: RenderCommands<'c'>, _render: RenderRes<Image>) {}
fn system_c3(_commands: RenderCommands<'c'>, _render: RenderRes<Image>) {}
fn system_t1(_commands: RenderCommands<'t'>, _render: RenderRes<Image>) {}
fn system_t2(_commands: RenderCommands<'t'>, _render: RenderRes<Image>) {}

#[test]
#[should_panic(
    expected = "RenderRes<rhyolite::ecs::tests::Image> can only be used in a render system. RenderCommands must be the first parameter of a render system."
)]
fn render_res_panics_on_non_render_system() {
    fn bad_system(_res: RenderRes<Image>) {}
    let mut schedule = Schedule::new(Update);
    schedule.add_systems(bad_system);
    let mut world = World::default();
    assert!(schedule.initialize(&mut world).is_err());
}

#[test]
#[should_panic(
    expected = "RenderResMut<rhyolite::ecs::tests::Image> can only be used in a render system. RenderCommands must be the first parameter of a render system."
)]
fn render_res_mut_panics_on_non_render_system() {
    fn bad_system(_res: RenderResMut<Image>) {}
    let mut schedule = Schedule::new(Update);
    schedule.add_systems(bad_system);
    let mut world = World::default();
    assert!(schedule.initialize(&mut world).is_err());
}

const ROUTER: QueuesRouter = QueuesRouter {
    queue_type_to_index: [QueueRef(0), QueueRef(1), QueueRef(2), QueueRef(3)],
    queue_type_to_family: [0, 0, 0, 0],
    queue_family_to_types: Vec::new(),
};

#[test]
fn test0() {
    let mut schedule = Schedule::new(Update);
    schedule.add_build_pass(RenderSystemPass::new());

    schedule.add_systems((
        system_g1,
        system_g2,
        system_c1.after(system_g1).after(system_g2),
        system_t1,
        system_g3.after(system_g2).after(system_t1),
        system_t2.after(system_c1).after(system_g3),
    ));

    let mut world = World::default();
    world.insert_resource(ROUTER);
    schedule.initialize(&mut world).unwrap();
}

#[test]
fn test1() {
    let mut schedule = Schedule::new(Update);
    schedule.add_build_pass(RenderSystemPass::new());

    schedule.add_systems((
        system_g1,
        system_c1.after(system_g1),
        system_g2.after(system_c1),
        system_t2.after(system_g2),
        system_g3.after(system_t2),
    ));

    let mut world = World::default();
    world.insert_resource(ROUTER);
    schedule.initialize(&mut world).unwrap();
}
