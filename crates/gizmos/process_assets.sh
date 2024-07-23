mkdir -p ./imported_assets/Default
touch ./imported_assets/Default/gizmo.vert
touch ./imported_assets/Default/gizmo.frag
RUST_LOG=debug cargo run --bin process-assets
