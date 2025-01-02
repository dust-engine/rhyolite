mkdir -p ./imported_assets/Default
touch ./imported_assets/Default/egui.vert
touch ./imported_assets/Default/egui.frag
RUST_LOG=debug cargo run --bin process-assets
