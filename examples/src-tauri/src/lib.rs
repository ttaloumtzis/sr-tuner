use std::fs;
use std::process::Command;
use tauri::Manager;
use sha2::{Digest, Sha256};

mod sidecar;
use sidecar::{SidecarState, StartupQueueState};

// ── Sidecar lifecycle ──────────────────────────────────────────────────────

/// Spawn the sidecar and register the startup queue as managed state.
/// §16.11: Messages sent before `sidecar.ready` are buffered in the queue
///         and flushed automatically once the ready signal is detected.
#[tauri::command]
async fn start_sidecar(app: tauri::AppHandle) -> Result<(), String> {
    sidecar::spawn_sidecar(&app)?;
    // Retrieve the queue Arc created inside spawn_sidecar and register it.
    if let Some(queue) = sidecar::take_startup_queue() {
        app.manage(StartupQueueState(queue));
    }
    Ok(())
}

/// §25.8 — Spawn a downloaded GPU-variant sidecar from an explicit path.
/// Called by the onboarding screen after the GPU variant download completes.
#[tauri::command]
async fn spawn_sidecar_from_path(
    app: tauri::AppHandle,
    queue_state: tauri::State<'_, StartupQueueState>,
    path: String,
) -> Result<(), String> {
    sidecar::spawn_sidecar_from_path(&app, &path, &queue_state)
}

/// Send a JSON-encoded IPC message to the sidecar's stdin.
/// §16.11: If sidecar.ready has not yet been received, the message is buffered
///         in the startup queue and flushed once the ready signal arrives.
#[tauri::command]
async fn send_to_sidecar(
    state: tauri::State<'_, SidecarState>,
    queue_state: tauri::State<'_, StartupQueueState>,
    payload: String,
) -> Result<(), String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    let child = guard.as_mut().ok_or("Sidecar is not running")?;
    sidecar::enqueue_or_send(child, &queue_state.0, payload)
}

#[tauri::command]
async fn kill_sidecar(state: tauri::State<'_, SidecarState>) -> Result<(), String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(child) = guard.take() {
        child.kill().map_err(|e| e.to_string())?;
    }
    Ok(())
}

// ── Deployment mode ────────────────────────────────────────────────────────

#[tauri::command]
fn get_deployment_mode() -> &'static str {
    if cfg!(feature = "bundled-sidecar") {
        "bundled"
    } else {
        "dev"
    }
}

// ── Disk space check (bundled mode, §6.3) ──────────────────────────────────

#[tauri::command]
fn check_disk_space(path: String) -> Result<f64, String> {
    let check_path = resolve_check_path(&path);
    available_disk_gb(&check_path)
}

fn resolve_check_path(path: &str) -> String {
    let p = std::path::Path::new(path);
    if p.exists() {
        return path.to_string();
    }
    if let Some(parent) = p.parent() {
        if parent.exists() {
            return parent.to_string_lossy().to_string();
        }
    }
    ".".to_string()
}

#[cfg(target_os = "windows")]
fn available_disk_gb(path: &str) -> Result<f64, String> {
    let drive = std::path::Path::new(path)
        .components()
        .next()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .unwrap_or_else(|| "C:".to_string());
    let script = format!(
        "(Get-PSDrive -Name '{}')[0].Free",
        drive.trim_end_matches(':')
    );
    let out = Command::new("powershell")
        .args(["-NoProfile", "-Command", &script])
        .output()
        .map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    text.parse::<u64>()
        .map(|b| b as f64 / (1024.0_f64.powi(3)))
        .map_err(|e| e.to_string())
}

#[cfg(not(target_os = "windows"))]
fn available_disk_gb(path: &str) -> Result<f64, String> {
    let out = Command::new("df")
        .arg("-k")
        .arg(path)
        .output()
        .map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().nth(1).ok_or("df produced no data line")?;
    let avail_kb: u64 = line
        .split_whitespace()
        .nth(3)
        .ok_or("unexpected df format")?
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())?;
    Ok(avail_kb as f64 / (1024.0 * 1024.0))
}

// ── Dev dependency checks (dev mode, §6.4) ─────────────────────────────────

#[derive(serde::Serialize)]
pub struct DevCheckResult {
    python_ok: bool,
    python_version: Option<String>,
    torch_ok: bool,
    basicsr_ok: bool,
    ffmpeg_ok: bool,
    errors: Vec<String>,
}

#[tauri::command]
fn check_dev_dependencies(app: tauri::AppHandle) -> DevCheckResult {
    let mut errors: Vec<String> = Vec::new();

    // Prefer the venv Python so torch/basicsr installed via `uv sync` are found.
    // Fall back to system python3/python if the venv doesn't exist yet.
    let venv_py = {
        let base = sidecar_project_dir(&app).join(".venv");
        let candidate = if cfg!(target_os = "windows") {
            base.join("Scripts").join("python.exe")
        } else {
            base.join("bin").join("python")
        };
        if candidate.exists() { Some(candidate) } else { None }
    };
    let py_cmd: std::ffi::OsString = venv_py
        .as_ref()
        .map(|p| p.as_os_str().to_owned())
        .unwrap_or_else(|| {
            if cfg!(target_os = "windows") { "python" } else { "python3" }.into()
        });

    let (python_ok, python_version) = match Command::new(&py_cmd).arg("--version").output() {
        Ok(out) => {
            let raw = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout).trim(),
                String::from_utf8_lossy(&out.stderr).trim()
            );
            let ok = python_version_ok(&raw);
            if !ok {
                errors.push(format!("Python >= 3.11 required, got: {raw}"));
            }
            (ok, Some(raw))
        }
        Err(e) => {
            errors.push(format!("Python not found on PATH: {e}"));
            (false, None)
        }
    };

    let torch_ok = Command::new(&py_cmd)
        .args(["-c", "import torch"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !torch_ok {
        errors.push("PyTorch is not importable".to_string());
    }

    // basicsr 1.4.2 imports torchvision.transforms.functional_tensor which was
    // removed in torchvision >= 0.17. Apply a shim before the import check so
    // the installed package is considered valid despite the upstream API change.
    let basicsr_check = concat!(
        "import sys\n",
        "try:\n",
        "    import torchvision.transforms.functional_tensor\n",
        "except ImportError:\n",
        "    import torchvision.transforms.functional as _f\n",
        "    sys.modules['torchvision.transforms.functional_tensor'] = _f\n",
        "import basicsr\n",
    );
    let basicsr_ok = Command::new(&py_cmd)
        .args(["-c", basicsr_check])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !basicsr_ok {
        errors.push("BasicSR is not importable".to_string());
    }

    let ffmpeg_ok = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ffmpeg_ok {
        errors.push("FFmpeg not found on PATH".to_string());
    }

    DevCheckResult { python_ok, python_version, torch_ok, basicsr_ok, ffmpeg_ok, errors }
}

fn python_version_ok(ver_str: &str) -> bool {
    if let Some(v) = ver_str.split_whitespace().nth(1) {
        let parts: Vec<u32> = v.split('.').filter_map(|x| x.parse().ok()).collect();
        if parts.len() >= 2 {
            return parts[0] > 3 || (parts[0] == 3 && parts[1] >= 11);
        }
    }
    false
}

// ── Filesystem helpers ──────────────────────────────────────────────────────

#[tauri::command]
fn list_dir(path: String) -> Result<Vec<String>, String> {
    let entries = fs::read_dir(&path).map_err(|e| e.to_string())?;
    let mut items: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if e.path().is_dir() {
                name + "/"
            } else {
                name
            }
        })
        .collect();
    items.sort();
    Ok(items)
}

#[tauri::command]
fn read_text_file(path: String) -> Result<String, String> {
    fs::read_to_string(&path).map_err(|e| e.to_string())
}

#[tauri::command]
fn write_text_file(path: String, content: String) -> Result<(), String> {
    if let Some(parent) = std::path::Path::new(&path).parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(&path, content).map_err(|e| e.to_string())
}

#[tauri::command]
fn create_dir_all(path: String) -> Result<(), String> {
    fs::create_dir_all(&path).map_err(|e| e.to_string())
}

#[tauri::command]
fn path_exists(path: String) -> bool {
    std::path::Path::new(&path).exists()
}

// ── GPU variant helpers (§19.8, §19.9) ────────────────────────────────────

/// Compute SHA-256 hex digest of a file.
#[tauri::command]
fn sha256_file(path: String) -> Result<String, String> {
    use std::io::Read;
    let mut file = fs::File::open(&path).map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65_536];
    loop {
        let n = file.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Delete a single file (used to clean up a corrupt download).
#[tauri::command]
fn delete_file(path: String) -> Result<(), String> {
    fs::remove_file(&path).map_err(|e| e.to_string())
}

/// Mark a file as executable on Unix; no-op on Windows.
#[tauri::command]
fn set_executable(path: String) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&path).map_err(|e| e.to_string())?.permissions();
        perms.set_mode(perms.mode() | 0o111);
        fs::set_permissions(&path, perms).map_err(|e| e.to_string())?;
    }
    let _ = path;
    Ok(())
}

/// Download a URL to a file.  Progress is coarse-grained (bytes written) because
/// Tauri commands are request/response and cannot stream events mid-invocation.
/// The frontend polls via a separate progress store updated by Tauri events if
/// a streaming download is needed in the future.
#[tauri::command]
async fn download_file(url: String, dest: String) -> Result<(), String> {
    let response = reqwest::get(&url).await.map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!("HTTP {}", response.status()));
    }
    let bytes = response.bytes().await.map_err(|e| e.to_string())?;
    fs::write(&dest, &bytes).map_err(|e| e.to_string())?;
    Ok(())
}

// ── Frontend IPC log (§19.12, §19.13) ────────────────────────────────────

/// Append a single log line to the frontend rolling log file.
/// §19.13: Rotates when the file exceeds 10 MB, keeping the last 5 files.
#[tauri::command]
fn append_frontend_log(app: tauri::AppHandle, line: String) -> Result<(), String> {
    use std::io::Write;

    let log_path = frontend_log_path(&app)?;

    // §19.13 — Rotate if the file has grown past 10 MB
    if let Ok(meta) = fs::metadata(&log_path) {
        if meta.len() > 10 * 1024 * 1024 {
            rotate_log_files(&log_path, 5);
        }
    }

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| e.to_string())?;

    writeln!(file, "{line}").map_err(|e| e.to_string())
}

fn frontend_log_path(app: &tauri::AppHandle) -> Result<std::path::PathBuf, String> {
    let data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let log_dir = data_dir.join("logs");
    fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
    let today = chrono_today();
    Ok(log_dir.join(format!("frontend-{today}.log")))
}

fn chrono_today() -> String {
    // Simple date without pulling chrono — format YYYY-MM-DD via std
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = now / 86_400;
    // Approximate Gregorian date (good enough for a log filename)
    let year = 1970 + days / 365;
    let day_of_year = days % 365;
    let month = day_of_year / 30 + 1;
    let day = day_of_year % 30 + 1;
    format!("{year:04}-{month:02}-{day:02}")
}

/// §19.13 — Rotate log files: shift old files down and keep at most `max_files`.
fn rotate_log_files(current: &std::path::Path, max_files: usize) {
    // Shift: frontend-DATE.log.4 → drop, .3 → .4, …, current → .1
    for i in (1..max_files).rev() {
        let old = current.with_extension(format!("log.{i}"));
        let new = current.with_extension(format!("log.{}", i + 1));
        let _ = fs::rename(&old, &new);
    }
    let rotated = current.with_extension("log.1");
    let _ = fs::rename(current, &rotated);
}

// ── §19.14 [Gap J] — Export Logs ─────────────────────────────────────────

/// Zip the last 5 frontend and sidecar log files from `<app_data>/logs/`
/// and write the archive to `dest_path`.
#[tauri::command]
fn export_logs(app: tauri::AppHandle, dest_path: String) -> Result<(), String> {
    use std::io::Write;

    let data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let log_dir = data_dir.join("logs");

    // Collect *.log* files, sort by name (names include dates), take newest 5.
    let mut log_files: Vec<std::path::PathBuf> = Vec::new();
    if log_dir.exists() {
        let entries = fs::read_dir(&log_dir).map_err(|e| e.to_string())?;
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            let ext = path.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_default();
            if ext.starts_with("log") {
                log_files.push(path);
            }
        }
    }
    log_files.sort();
    let take_from = if log_files.len() > 5 { log_files.len() - 5 } else { 0 };
    let log_files = &log_files[take_from..];

    let dest = std::path::Path::new(&dest_path);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let file = fs::File::create(dest).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    for log_path in log_files {
        let name = log_path.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        zip.start_file(&name, options).map_err(|e| e.to_string())?;
        let content = fs::read(log_path).map_err(|e| e.to_string())?;
        zip.write_all(&content).map_err(|e| e.to_string())?;
    }

    zip.finish().map_err(|e| e.to_string())?;
    Ok(())
}

// ── §20.2 / §20.11 — File manager and directory validation ───────────────

/// Open `path` in the platform's default file manager.
/// Returns Ok(()) immediately after launching the process (fire-and-forget).
#[tauri::command]
fn open_in_file_manager(path: String) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        Command::new("explorer")
            .arg(&path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(&path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open")
            .arg(&path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    let _ = path;
    Ok(())
}

/// Return true if `path` is an existing directory that this process can write to.
/// Used by ScreenInference for pre-flight output-directory validation (§20.11).
#[tauri::command]
fn validate_dir_writable(path: String) -> bool {
    let p = std::path::Path::new(&path);
    if !p.is_dir() {
        return false;
    }
    // Attempt to create and immediately delete a temp file to test writability.
    let probe = p.join(".srtuner_write_probe");
    match fs::File::create(&probe) {
        Ok(_) => {
            let _ = fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

// ── Dev-mode venv helpers (§19.5) ─────────────────────────────────────────

/// Return true if the sidecar's `.venv` directory already exists.
/// The sidecar directory is resolved relative to the app resource dir so it
/// works both when running `cargo tauri dev` and from a built AppImage.
#[tauri::command]
fn check_venv_exists(app: tauri::AppHandle) -> bool {
    let venv = venv_path(&app);
    venv.exists()
}

/// Run `uv sync` inside the sidecar project directory to create / update the
/// virtual environment.  Returns the combined stdout+stderr output so the
/// frontend can display progress text.
///
/// This is a blocking operation — the frontend should call it from a background
/// context and stream the resulting string into the onboarding step display.
#[tauri::command]
fn setup_venv(app: tauri::AppHandle) -> Result<String, String> {
    let sidecar_dir = sidecar_project_dir(&app);
    if !sidecar_dir.exists() {
        return Err(format!("Sidecar directory not found: {}", sidecar_dir.display()));
    }

    let out = Command::new("uv")
        .args(["sync"])
        .current_dir(&sidecar_dir)
        .output()
        .map_err(|e| format!("Failed to run `uv sync` in {}: {e}", sidecar_dir.display()))?;

    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let combined = format!("{stdout}{stderr}");

    if out.status.success() {
        Ok(combined)
    } else {
        Err(combined)
    }
}

fn venv_path(app: &tauri::AppHandle) -> std::path::PathBuf {
    sidecar_project_dir(app).join(".venv")
}

fn sidecar_project_dir(app: &tauri::AppHandle) -> std::path::PathBuf {
    // Compile-time path: CARGO_MANIFEST_DIR is src-tauri/, sidecar is one level up.
    // This is the most reliable probe in dev mode because it's baked in at build time.
    let dev_candidate = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../sidecar");
    if dev_candidate.join("pyproject.toml").exists() {
        return dev_candidate.canonicalize().unwrap_or(dev_candidate);
    }

    // Bundled / production: sidecar dir may be packaged alongside resources.
    if let Ok(res_dir) = app.path().resource_dir() {
        let candidate = res_dir.join("sidecar");
        if candidate.join("pyproject.toml").exists() {
            return candidate;
        }
    }

    // Walk up from the running binary to find sidecar/pyproject.toml.
    let mut dir = std::env::current_exe().unwrap_or_default();
    for _ in 0..6 {
        dir.pop();
        let candidate = dir.join("sidecar");
        if candidate.join("pyproject.toml").exists() {
            return candidate;
        }
    }

    // Last resort: assume sidecar/ is a sibling of cwd.
    std::path::PathBuf::from("../sidecar")
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_fs::init())
        .manage(SidecarState(std::sync::Mutex::new(None)))
        .manage(StartupQueueState(std::sync::Arc::new(std::sync::Mutex::new(None))))
        .invoke_handler(tauri::generate_handler![
            get_deployment_mode,
            check_disk_space,
            check_dev_dependencies,
            list_dir,
            read_text_file,
            write_text_file,
            start_sidecar,
            spawn_sidecar_from_path,
            send_to_sidecar,
            kill_sidecar,
            create_dir_all,
            path_exists,
            check_venv_exists,
            setup_venv,
            sha256_file,
            delete_file,
            set_executable,
            download_file,
            append_frontend_log,
            open_in_file_manager,
            validate_dir_writable,
            export_logs,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
