use std::path::Path;

fn main() {
    // Pass the target triple to the compiled crate so it can find the sidecar.
    let triple = std::env::var("TARGET").unwrap_or_else(|_| "unknown".into());
    println!("cargo:rustc-env=SR_ENGINE_TARGET_TRIPLE={triple}");

    // Create a placeholder sidecar binary if the real one hasn't been built
    // yet. This allows `cargo tauri dev` to compile without the sidecar.
    // The placeholder is tiny (< 100 bytes) so the Rust runtime can detect
    // it and fall back to `uv run uvicorn`.
    let binary_path = Path::new("binaries").join(format!("sr-engine-{triple}"));

    if !binary_path.exists() {
        std::fs::create_dir_all("binaries").ok();
        std::fs::write(&binary_path, b"x").ok();
    }

    tauri_build::build();
}