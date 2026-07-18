use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;
use std::process::{Child, Command, Stdio};
use tauri::Manager;

// ── Managed state ─────────────────────────────────────────────────────────

struct ServerState(Mutex<Option<Child>>);

// ── Python server lifecycle ───────────────────────────────────────────────

#[tauri::command]
fn start_python_server(app: tauri::AppHandle) -> Result<(), String> {
    let child = Command::new("uv")
        .args(["run", "uvicorn", "sr_engine.api.app:app", "--host", "127.0.0.1", "--port", "8765", "--log-level", "info"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to start Python server: {e}"))?;

    let state = app.state::<ServerState>();
    *state.0.lock().unwrap() = Some(child);

    // Poll health endpoint for up to 30 seconds
    let start = std::time::Instant::now();
    while start.elapsed() < Duration::from_secs(30) {
        match ureq::get("http://127.0.0.1:8765/api/health").call() {
            Ok(r) if r.status() == 200 => return Ok(()),
            _ => std::thread::sleep(Duration::from_millis(500)),
        }
    }

    Err("Server failed to start within 30 seconds".to_string())
}

#[tauri::command]
fn stop_python_server(app: tauri::AppHandle) -> Result<(), String> {
    let state = app.state::<ServerState>();
    let mut guard = state.0.lock().unwrap();
    if let Some(mut child) = guard.take() {
        child.kill().ok();
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

// ── App entry point ───────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(ServerState(Mutex::new(None)))
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}