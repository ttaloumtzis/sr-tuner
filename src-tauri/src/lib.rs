use std::path::Path;
use std::sync::Mutex;
use std::process::{Child, Command, Stdio};
use tauri::{Emitter, Manager};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

mod env_installer;

// ── Managed state ─────────────────────────────────────────────────────────

struct ServerState(Mutex<Option<Child>>);

// ── Process group kill helper ────────────────────────────────────────────

fn kill_process_group(pid: u32) {
    #[cfg(unix)]
    {
        // Send SIGTERM to the entire process group (-pid).
        // setpgid in start_python_server ensures uv + uvicorn are both in this group.
        unsafe { libc::kill(-(pid as i32), libc::SIGTERM); }

        // Wait up to 5 seconds for graceful shutdown, but break early
        // as soon as the process group is empty (checked via kill(2) with sig=0).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(100));
            // kill with sig=0 only checks existence — 0 means alive, ESRCH means gone
            if unsafe { libc::kill(-(pid as i32), 0) } != 0 {
                return; // process group is empty, no need for SIGKILL
            }
        }

        // Force-kill the entire group if still alive after the grace period
        let _ = unsafe { libc::kill(-(pid as i32), libc::SIGKILL) };
    }

    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .spawn();
    }
}

// ── Sidecar binary resolution ────────────────────────────────────────────
// Tries the bundled sidecar (release) or the dev build output (debug).
// Falls back to None so the caller can use `uv run uvicorn` instead.

#[allow(unused_variables)]
fn find_sidecar_binary(app: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    #[cfg(not(debug_assertions))]
    {
        // Release mode: bundled in the app's resource directory via externalBin.
        let dir = app.path().resource_dir().ok()?;
        let path = dir.join("bin").join("sr-engine");
        if is_valid_sidecar(&path) {
            return Some(path);
        }
        // Windows variant
        let path_exe = dir.join("bin").join("sr-engine.exe");
        if is_valid_sidecar(&path_exe) {
            return Some(path_exe);
        }
    }

    #[cfg(debug_assertions)]
    {
        // Dev mode: check the project's build output.
        // The sidecar binary is built by scripts/build-sidecar.sh and placed at
        // src-tauri/binaries/sr-engine-<target-triple>.
        // build.rs creates a 1-byte placeholder if the real binary is missing;
        // we reject anything smaller than 1 KB.
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let triple = env!("SR_ENGINE_TARGET_TRIPLE");
        let path = manifest_dir
            .join("binaries")
            .join(format!("sr-engine-{triple}"));
        if is_valid_sidecar(&path) {
            return Some(path);
        }
        let path_exe = manifest_dir
            .join("binaries")
            .join(format!("sr-engine-{triple}.exe"));
        if is_valid_sidecar(&path_exe) {
            return Some(path_exe);
        }
    }

    None
}

/// Returns `true` if the path is a valid sidecar.
/// build.rs creates a 1-byte placeholder when the real binary hasn't been
/// built yet — we reject that so the caller falls back to `uv run uvicorn`.
///
/// Two build modes are accepted:
///   --onefile: a single binary file > 1 KB
///   --onedir:  a directory containing the binary + _internal/
fn is_valid_sidecar(path: &std::path::Path) -> bool {
    let meta = std::fs::metadata(path);
    match meta {
        Ok(m) if m.is_dir() => true,          // --onedir directory
        Ok(m) => m.len() > 1024,              // --onefile real binary
        _ => false,
    }
}

/// Resolve the actual executable path for a sidecar.
///   --onefile: path is the binary itself
///   --onedir:  path is a directory, binary is at path / sr-engine
fn resolve_sidecar_exe(path: &std::path::Path) -> std::path::PathBuf {
    if path.is_dir() {
        path.join("sr-engine")
    } else {
        path.to_path_buf()
    }
}

// ── Python server lifecycle ───────────────────────────────────────────────

#[tauri::command]
fn start_python_server(app: tauri::AppHandle) -> Result<(), String> {
    let state = app.state::<ServerState>();

    // Kill any previously-spawned instance to prevent orphaned processes
    // from React Strict Mode double-mount, HMR, or user re-requesting.
    if let Ok(mut guard) = state.0.lock() {
        if let Some(mut old) = guard.take() {
            kill_process_group(old.id());
            let _ = old.wait();
        }
    }

    // Priority 1: installed env from wizard (~/.sr-tuner/env/env.json)
    if let Some(meta) = env_installer::read_env_meta() {
        let current = env!("CARGO_PKG_VERSION");
        if meta.app_version != current {
            env_installer::maybe_update_env(&app, &meta)
                .map_err(|e| format!("Env update failed: {e}"))?;
            // Re-read updated meta
            let meta = env_installer::read_env_meta()
                .ok_or_else(|| "Failed to read updated env.json".to_string())?;
            let child = env_installer::launch_from_env(&meta)?;
            if let Ok(mut guard) = state.0.lock() {
                *guard = Some(child);
            }
            return Ok(());
        }
        let child = env_installer::launch_from_env(&meta)?;
        if let Ok(mut guard) = state.0.lock() {
            *guard = Some(child);
        }
        return Ok(());
    }

    // Priority 2: bundled sidecar (release) or dev sidecar build
    let mut cmd = if let Some(sidecar_path) = find_sidecar_binary(&app) {
        let exe = resolve_sidecar_exe(&sidecar_path);
        Command::new(exe)
    } else {
        // Priority 3: fall back to `uv run uvicorn` (dev mode).
        let mut c = Command::new("uv");
        c.args([
            "run",
            "uvicorn",
            "sr_engine.api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
            "--log-level",
            "info",
        ]);
        c
    };

    cmd.stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    #[cfg(unix)]
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    let child = cmd.spawn()
        .map_err(|e| format!("Failed to start Python server: {e}"))?;

    if let Ok(mut guard) = state.0.lock() {
        *guard = Some(child);
    }
    Ok(())
}

#[tauri::command]
fn stop_python_server(app: tauri::AppHandle) -> Result<(), String> {
    let state = app.state::<ServerState>();
    let Ok(mut guard) = state.0.lock() else { return Ok(()) };
    if let Some(mut child) = guard.take() {
        kill_process_group(child.id());
        child.wait().ok();
    }
    Ok(())
}

#[tauri::command]
fn get_server_url() -> String {
    "http://localhost:8765".to_string()
}

// ── Filesystem commands ──────────────────────────────────────────────────

#[tauri::command]
fn list_dir(path: String) -> Result<Vec<String>, String> {
    let dir = Path::new(&path);
    if !dir.is_dir() {
        return Err(format!("Not a directory: {path}"));
    }
    let entries = std::fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                format!("{name}/")
            } else {
                name
            }
        })
        .collect();
    Ok(entries)
}

#[tauri::command]
fn read_text_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| e.to_string())
}

#[tauri::command]
fn write_text_file(path: String, contents: String) -> Result<(), String> {
    if let Some(parent) = Path::new(&path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&path, &contents).map_err(|e| e.to_string())
}

#[tauri::command]
fn create_dir_all(path: String) -> Result<(), String> {
    std::fs::create_dir_all(&path).map_err(|e| e.to_string())
}

#[tauri::command]
fn path_exists(path: String) -> bool {
    Path::new(&path).exists()
}

#[tauri::command]
fn delete_file(path: String) -> Result<(), String> {
    std::fs::remove_file(&path).map_err(|e| e.to_string())
}

#[tauri::command]
fn open_in_file_manager(path: String) -> Result<(), String> {
    let p = Path::new(&path);
    let dir = if p.is_dir() { p } else { p.parent().unwrap_or(p) };
    #[cfg(target_os = "linux")]
    { Command::new("xdg-open").arg(dir).spawn().ok(); }
    #[cfg(target_os = "macos")]
    { Command::new("open").arg(dir).spawn().ok(); }
    #[cfg(target_os = "windows")]
    { Command::new("explorer").arg(dir).spawn().ok(); }
    Ok(())
}

// ── Dataset file operations ─────────────────────────────────────────────

#[tauri::command]
fn delete_directory(path: String) -> Result<(), String> {
    std::fs::remove_dir_all(&path).map_err(|e| e.to_string())
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    if src.is_dir() {
        std::fs::create_dir_all(dst)?;
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            let child_src = entry.path();
            let child_dst = dst.join(entry.file_name());
            if entry.file_type()?.is_dir() {
                copy_dir_recursive(&child_src, &child_dst)?;
            } else {
                std::fs::copy(&child_src, &child_dst)?;
            }
        }
    }
    Ok(())
}

#[tauri::command]
fn copy_directory(src: String, dst: String) -> Result<(), String> {
    copy_dir_recursive(Path::new(&src), Path::new(&dst))
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn list_image_files(path: String) -> Result<Vec<String>, String> {
    let dir = Path::new(&path);
    if !dir.is_dir() {
        return Err(format!("Not a directory: {path}"));
    }
    let mut files: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let p = entry.path();
        if p.is_file() {
            if let Some(ext) = p.extension() {
                match ext.to_str().unwrap_or("").to_lowercase().as_str() {
                    "png" | "jpg" | "jpeg" | "webp" | "bmp" => {
                        files.push(p.to_string_lossy().to_string());
                    }
                    _ => {}
                }
            }
        }
    }
    files.sort();
    Ok(files)
}

// ── Environment installer commands ───────────────────────────────────────

#[tauri::command]
fn check_first_run() -> bool {
    env_installer::is_first_run()
}

#[tauri::command]
fn probe_system() -> env_installer::SystemInfo {
    env_installer::probe_system()
}

#[tauri::command]
fn install_env(
    app: tauri::AppHandle,
    backend: String,
    env_type: String,
    rocm_venv_path: Option<String>,
) -> Result<(), String> {
    let app2 = app.clone();
    std::thread::spawn(move || {
        let result =
            env_installer::install_env(app2.clone(), &app2, backend, env_type, rocm_venv_path);
        if let Err(e) = result {
            let _ = app2.emit("install-error", &e);
        }
    });
    Ok(())
}

#[tauri::command]
fn cancel_install(app: tauri::AppHandle) {
    env_installer::cancel_install(&app);
}

#[tauri::command]
fn get_env_dir() -> String {
    env_installer::get_env_dir().to_string_lossy().to_string()
}

#[tauri::command]
fn verify_rocm_venv(venv_path: String) -> env_installer::RocmVenvInfo {
    env_installer::verify_rocm_venv(&venv_path)
}

// ── App entry point ───────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(ServerState(Mutex::new(None)))
        .manage(env_installer::InstallState::new())
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();

                let state = window.state::<ServerState>();
                if let Ok(mut guard) = state.0.lock() {
                    if let Some(mut child) = guard.take() {
                        std::thread::spawn(move || {
                            kill_process_group(child.id());
                            child.wait().ok();
                        });
                    }
                }

                let win = window.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_secs(10));
                    let _ = win.destroy();
                });
            }
        })
        .invoke_handler(tauri::generate_handler![
            start_python_server,
            stop_python_server,
            get_server_url,
            check_first_run,
            probe_system,
            install_env,
            cancel_install,
            get_env_dir,
            verify_rocm_venv,
            list_dir,
            read_text_file,
            write_text_file,
            create_dir_all,
            path_exists,
            delete_file,
            open_in_file_manager,
            delete_directory,
            copy_directory,
            list_image_files,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
