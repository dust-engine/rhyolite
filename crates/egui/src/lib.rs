use bevy::app::{App, Plugin, Update};
pub use bevy_egui::*;

pub struct EguiPlugin;
impl Plugin for EguiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, ui_example_system);
    }
}

fn ui_example_system(mut contexts: EguiContexts) {
    egui::Window::new("Hello").show(contexts.ctx_mut(), |ui| {
        ui.label("world");
    });
}
