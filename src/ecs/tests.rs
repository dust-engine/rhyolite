use super::*;
use bevy_app::Update;
use bevy_ecs::{
    schedule::{IntoSystemConfigs, Schedule},
    system::Resource,
    world::World,
};

#[derive(Resource)]
struct Image;

fn system1(_commands: RenderCommands, _render: RenderResMut<Image>) {}
fn system2(_commands: RenderCommands, _render: RenderRes<Image>) {}

#[test]
#[should_panic(
    expected = "RenderRes<rhyolite::ecs::tests::Image> can only be used in a render system, but rhyolite::ecs::tests::render_res_panics_on_non_render_system::bad_system is not. RenderCommands must be the first parameter of a render system."
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
    expected = "RenderResMut<rhyolite::ecs::tests::Image> can only be used in a render system, but rhyolite::ecs::tests::render_res_mut_panics_on_non_render_system::bad_system is not. RenderCommands must be the first parameter of a render system."
)]
fn render_res_mut_panics_on_non_render_system() {
    fn bad_system(_res: RenderResMut<Image>) {}
    let mut schedule = Schedule::new(Update);
    schedule.add_systems(bad_system);
    let mut world = World::default();
    assert!(schedule.initialize(&mut world).is_err());
}

#[test]
fn my_test() {
    let mut schedule = Schedule::new(Update);
    schedule.add_build_pass(RenderSystemPass {});

    schedule.add_systems((system1, system2.before(system1)));

    let mut world = World::default();
    schedule.initialize(&mut world).unwrap();
    println!("Done");
}
