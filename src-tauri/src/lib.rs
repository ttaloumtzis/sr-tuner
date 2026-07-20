use std::path::Path;
use std::sync::Mutex;
use std::process::{Child, Command, Stdio};
use tauri::Manager;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

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

    let mut cmd = Command::new("uv");
    cmd.args([
            "run", "uvicorn", "sr_engine.api.app:app",
            "--host", "127.0.0.1", "--port", "8765", "--log-level", "info",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());

    #[cfg(unix)]
    unsafe {
        cmd.pre_exec(|| {
            // Isolate this process into its own process group so we can
            // kill *every* descendant (uv → uvicorn) in one shot later.
            libc::setpgid(0, 0);
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
    "http://127.0.0.1:8765".to_string()
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

// ── App entry point ───────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(ServerState(Mutex::new(None)))
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                // Keep the window alive so JS cleanup (project save, training
                // cancel) can complete asynchronously.
                api.prevent_close();

                // Kill the server in a background thread so the Tauri event
                // loop stays responsive for JS cleanup.
                let state = window.state::<ServerState>();
                if let Ok(mut guard) = state.0.lock() {
                    if let Some(mut child) = guard.take() {
                        std::thread::spawn(move || {
                            kill_process_group(child.id());
                            child.wait().ok();
                        });
                    }
                }

                // Give JS (useSaveTrigger etc.) 10 seconds to finish its own
                // cleanup and call destroy(). If JS never responds, force-close.
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
