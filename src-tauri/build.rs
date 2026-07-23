use std::path::Path;
use std::process::Command;

fn main() {
    // Build Python wheel for bundling. Used by the first-run wizard to install
    // the SR Engine package into ~/.sr-tuner/. In dev mode, only builds if
    // the wheel doesn't exist yet; in release, always rebuilds.
    let project_dir = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let dist_dir = project_dir.join("dist");
    let fixed_wheel = dist_dir.join("sr_engine.whl");

    let profile = std::env::var("PROFILE").unwrap_or_default();
    let needs_build = profile == "release" || !fixed_wheel.exists();

    if needs_build {
        std::fs::create_dir_all(&dist_dir).ok();
        let wheel_status = Command::new("uv")
            .args(["build", "--wheel", "--out-dir", &dist_dir.to_string_lossy()])
            .current_dir(project_dir)
            .status()
            .expect("uv build --wheel failed — is uv installed?");
        assert!(wheel_status.success(), "Python wheel build failed");
    }

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