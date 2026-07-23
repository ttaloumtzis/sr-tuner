use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

// ── Public types ───────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
pub struct SystemInfo {
    pub os: String,
    pub os_distro: Option<String>,
    pub cuda_available: bool,
    pub rocm_available: bool,
    pub mps_available: bool,
    pub has_ffmpeg: bool,
    pub has_uv: bool,
    pub has_python3: bool,
    pub supported_backends: Vec<String>,
    pub default_backend: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct EnvMeta {
    pub app_version: String,
    pub backend: String,
    pub env_type: String,
    pub env_path: String,
    pub installed_at: String,
}

#[derive(Serialize, Clone)]
pub struct RocmVenvInfo {
    pub valid: bool,
    pub hip_version: Option<String>,
    pub python_version: Option<String>,
    pub error: Option<String>,
}

pub struct InstallState {
    pub child_pid: Mutex<Option<u32>>,
    pub cancelled: AtomicBool,
}

impl InstallState {
    pub fn new() -> Self {
        Self {
            child_pid: Mutex::new(None),
            cancelled: AtomicBool::new(false),
        }
    }
}

// ── Platform-dependent helpers ─────────────────────────────────────────

fn command_exists(cmd: &str) -> bool {
    #[cfg(unix)]
    {
        Command::new("sh")
            .args(["-c", &format!("command -v {cmd}")])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
    #[cfg(windows)]
    {
        Command::new("where")
            .arg(cmd)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
}

fn kill_pid(pid: u32) {
    #[cfg(unix)]
    unsafe {
        libc::kill(pid as i32, libc::SIGTERM);
        std::thread::sleep(std::time::Duration::from_millis(300));
        libc::kill(pid as i32, libc::SIGKILL);
    }
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .spawn();
    }
}

// ── Env directory resolution ──────────────────────────────────────────

pub fn get_env_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SRTUNER_ENV_DIR") {
        return PathBuf::from(dir).join("env");
    }
    #[cfg(debug_assertions)]
    {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let project = manifest.parent().unwrap();
        return project.join(".test-sr-tuner").join("env");
    }
    #[cfg(not(debug_assertions))]
    {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        let dir_name = if cfg!(windows) { "sr-tuner" } else { ".sr-tuner" };
        PathBuf::from(home).join(dir_name).join("env")
    }
}

// ── First-run detection ───────────────────────────────────────────────

pub fn is_first_run() -> bool {
    let forced = std::env::var("SRTUNER_FORCE_WIZARD").is_ok();
    if forced {
        return true;
    }
    let marker = get_env_dir().join("installed");
    !marker.exists()
}

// ── System probing ────────────────────────────────────────────────────

pub fn probe_system() -> SystemInfo {
    let os = std::env::consts::OS.to_string();
    let os_distro = probe_distro(&os);
    let (cuda, rocm, mps) = probe_gpus(&os);
    let has_ffmpeg = command_exists("ffmpeg");
    let has_uv = command_exists("uv");
    let has_python3 = command_exists("python3") || command_exists("python");

    // Support env override for testing different platform wizards
    let mock_os = std::env::var("SRTUNER_MOCK_OS").ok();
    let effective_os = mock_os.as_deref().unwrap_or(&os);

    let supported = supported_backends(effective_os, cuda, rocm, mps);
    let default = default_backend(effective_os, cuda, rocm, mps);

    SystemInfo {
        os: effective_os.to_string(),
        os_distro,
        cuda_available: cuda,
        rocm_available: rocm,
        mps_available: mps,
        has_ffmpeg,
        has_uv,
        has_python3,
        supported_backends: supported,
        default_backend: default,
    }
}

fn probe_distro(os: &str) -> Option<String> {
    if os != "linux" {
        return None;
    }
    let content = std::fs::read_to_string("/etc/os-release").ok()?;
    for line in content.lines() {
        if let Some(name) = line.strip_prefix("PRETTY_NAME=\"") {
            return Some(name.trim_end_matches('"').to_string());
        }
        if let Some(name) = line.strip_prefix("PRETTY_NAME=") {
            return Some(name.to_string());
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn probe_gpus(_os: &str) -> (bool, bool, bool) {
    let cuda = command_exists("nvidia-smi")
        || Path::new("/usr/lib/x86_64-linux-gnu/libcuda.so").exists()
        || Path::new("/usr/lib/x86_64-linux-gnu/libcuda.so.1").exists();
    let rocm = command_exists("rocm-smi") || Path::new("/opt/rocm").is_dir();
    (cuda, rocm, false)
}

#[cfg(target_os = "macos")]
fn probe_gpus(_os: &str) -> (bool, bool, bool) {
    let mps = Command::new("sysctl")
        .args(["-n", "hw.optional.arm64"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "1")
        .unwrap_or(false);
    (false, false, mps)
}

#[cfg(target_os = "windows")]
fn probe_gpus(_os: &str) -> (bool, bool, bool) {
    let cuda = command_exists("nvidia-smi.exe") || command_exists("nvidia-smi");
    let rocm = has_amd_gpu_windows();
    (cuda, rocm, false)
}

#[cfg(target_os = "windows")]
fn has_amd_gpu_windows() -> bool {
    // Tier 1: PowerShell WMI query for GPU vendor name
    if let Ok(output) = Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "& { (Get-CimInstance Win32_VideoController).Name -join '|' }",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
    {
        let name = String::from_utf8_lossy(&output.stdout).to_lowercase();
        if name.contains("amd") || name.contains("radeon") || name.contains("advanced micro") {
            return true;
        }
    }
    // Tier 2: AMD display driver file (no PowerShell needed)
    if Path::new(r"C:\Windows\System32\Drivers\amdkmdap.sys").exists() {
        return true;
    }
    // Tier 3: AMD ROCm registry key
    Command::new("reg")
        .args(["query", "HKLM\\SOFTWARE\\AMD\\ROCm"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn probe_gpus(_os: &str) -> (bool, bool, bool) {
    (false, false, false)
}

fn supported_backends(os: &str, cuda: bool, rocm: bool, mps: bool) -> Vec<String> {
    let mut v = Vec::new();
    match os {
        "linux" => {
            if cuda { v.push("cuda".into()); }
            if rocm { v.push("rocm".into()); }
            v.push("cpu".into());
        }
        "macos" => {
            if mps { v.push("mps".into()); }
            v.push("cpu".into());
        }
        "windows" => {
            if cuda { v.push("cuda".into()); }
            if rocm { v.push("rocm".into()); }
            v.push("cpu".into());
        }
        _ => v.push("cpu".into()),
    }
    v
}

fn default_backend(os: &str, cuda: bool, rocm: bool, mps: bool) -> String {
    match os {
        "linux" => {
            if cuda { return "cuda".into(); }
            if rocm { return "rocm".into(); }
            "cpu".into()
        }
        "macos" => {
            if mps { return "mps".into(); }
            "cpu".into()
        }
        "windows" => {
            if cuda { return "cuda".into(); }
            if rocm { return "rocm".into(); }
            "cpu".into()
        }
        _ => "cpu".into(),
    }
}

// ── Verify ROCm venv (Windows) ──────────────────────────────────────

pub fn verify_rocm_venv(venv_path: &str) -> RocmVenvInfo {
    let python = if cfg!(windows) {
        Path::new(venv_path).join("Scripts").join("python.exe")
    } else {
        Path::new(venv_path).join("bin").join("python")
    };

    if !python.exists() {
        return RocmVenvInfo {
            valid: false,
            hip_version: None,
            python_version: None,
            error: Some(format!(
                "Python not found at {} — create the venv first",
                python.display()
            )),
        };
    }

    let py_ver = Command::new(&python)
        .args(["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string()));

    let hip_ver = Command::new(&python)
        .args(["-c", "import torch; v = torch.version.hip; print(v or '')"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8(o.stdout).ok()?;
            let v = s.trim().to_string();
            if v.is_empty() { None } else { Some(v) }
        });

    let is_valid = hip_ver.is_some();

    RocmVenvInfo {
        valid: is_valid,
        hip_version: hip_ver,
        python_version: py_ver,
        error: if !is_valid {
            Some("ROCm PyTorch not found — run AMD Adrenalin to create the venv with PyTorch support".into())
        } else {
            None
        },
    }
}

// ── Torch index URLs ──────────────────────────────────────────────────

fn torch_index_url(backend: &str) -> Option<&'static str> {
    match backend {
        "cpu" => Some("https://download.pytorch.org/whl/cpu"),
        "cuda" => Some("https://download.pytorch.org/whl/cu121"),
        "rocm" => Some("https://download.pytorch.org/whl/rocm6.3"),
        "mps" => None, // macOS — standard PyPI
        _ => None,
    }
}

// ── Bundled resource resolution ───────────────────────────────────────

fn find_bundled_wheel(app: &AppHandle) -> Option<PathBuf> {
    // Glob for any wheel matching sr_engine-*.whl in the resource dir.
    if let Ok(dir) = app.path().resource_dir() {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("sr_engine") && name.ends_with(".whl") {
                    return Some(entry.path());
                }
            }
        }
    }
    #[cfg(debug_assertions)]
    {
        let project = Path::new(env!("CARGO_MANIFEST_DIR")).parent()?;
        let dist = project.join("dist");
        if let Ok(entries) = std::fs::read_dir(&dist) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("sr_engine") && name.ends_with(".whl") {
                    return Some(entry.path());
                }
            }
        }
    }
    None
}

fn find_bundled_sidecar_entry(app: &AppHandle) -> Option<PathBuf> {
    if let Ok(dir) = app.path().resource_dir() {
        let entry = dir.join("sidecar_entry.py");
        if entry.exists() {
            return Some(entry);
        }
    }
    #[cfg(debug_assertions)]
    {
        let project = Path::new(env!("CARGO_MANIFEST_DIR")).parent()?;
        let entry = project.join("scripts").join("sidecar_entry.py");
        if entry.exists() {
            return Some(entry);
        }
    }
    None
}

// ─── Helper: run a shell command with progress events ───────────────

/// On Unix, run the child connected to a PTY so `uv` uses line buffering
/// and its stdout lines are delivered to the frontend in real time.
#[cfg(unix)]
fn run_step(
    app: &AppHandle,
    state: &InstallState,
    cmd: &mut Command,
    label: &str,
) -> Result<(), String> {
    use std::ffi::CString;
    use std::os::unix::io::FromRawFd;

    app.emit("install-progress-label", label).ok();
    app.emit("install-log", format!("──── {label} ────")).ok();

    // Create a pseudo-terminal pair.
    let master = unsafe { libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY) };
    if master < 0 {
        return Err(format!(
            "{label}: PTY creation failed — {}",
            std::io::Error::last_os_error()
        ));
    }

    unsafe {
        if libc::grantpt(master) < 0 || libc::unlockpt(master) < 0 {
            let err = std::io::Error::last_os_error();
            let _ = libc::close(master);
            return Err(format!("{label}: PTY grant/unlock failed — {err}"));
        }
    }

    let slave_name = unsafe {
        let ptr = libc::ptsname(master);
        if ptr.is_null() {
            let err = std::io::Error::last_os_error();
            let _ = libc::close(master);
            return Err(format!("{label}: ptsname failed — {err}"));
        }
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    };
    let slave_cstr = CString::new(slave_name.as_str()).unwrap();

    // Suppress uv progress spinners and ANSI colours so the log is clean text.
    cmd.env("CI", "1").env("NO_COLOR", "1").env("TERM", "dumb");

    let mut child = unsafe {
        cmd.stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .pre_exec(move || {
                let slave = libc::open(slave_cstr.as_ptr(), libc::O_RDWR);
                if slave < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                libc::dup2(slave, 1); // stdout → PTY slave
                libc::dup2(slave, 2); // stderr → PTY slave
                if slave > 2 {
                    libc::close(slave);
                }
                Ok(())
            })
            .spawn()
    }
    .map_err(|e| format!("{label}: failed to start — {e}"))?;

    {
        let mut guard = state.child_pid.lock().unwrap();
        *guard = Some(child.id());
    }

    // Read lines from the PTY master as they are flushed by `uv`.
    let master_file = unsafe { std::fs::File::from_raw_fd(master) };
    let reader = std::io::BufReader::new(master_file);
    for line in reader.lines() {
        if state.cancelled.load(Ordering::Relaxed) {
            let _ = child.kill();
            return Err("Installation cancelled".into());
        }
        // On Linux, after the PTY slave is closed, read() returns EIO instead of
        // 0/EOF.  Treat any read error as end-of-output to avoid an infinite loop.
        match line {
            Ok(l) => { let _ = app.emit("install-log", &l); }
            Err(_) => break,
        }
    }
    // reader (and its master fd) is dropped here.

    {
        let mut guard = state.child_pid.lock().unwrap();
        *guard = None;
    }

    let status = child
        .wait()
        .map_err(|e| format!("{label}: wait failed — {e}"))?;

    if !status.success() {
        let code = status.code().unwrap_or(-1);
        return Err(format!("{label} failed (exit code {code})"));
    }

    app.emit("install-log", format!("✓ {label}")).ok();
    Ok(())
}

/// Fallback for Windows (and any non-Unix platform): use piped stdio.
/// The child’s output is block-buffered but captured after each command.
#[cfg(not(unix))]
fn run_step(
    app: &AppHandle,
    state: &InstallState,
    cmd: &mut Command,
    label: &str,
) -> Result<(), String> {
    app.emit("install-progress-label", label).ok();
    app.emit("install-log", format!("──── {label} ────")).ok();

    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{label}: failed to start — {e}"))?;

    {
        let mut guard = state.child_pid.lock().unwrap();
        *guard = Some(child.id());
    }

    let stdout = child.stdout.take().unwrap();
    for line in std::io::BufReader::new(stdout).lines() {
        if state.cancelled.load(Ordering::Relaxed) {
            let _ = child.kill();
            return Err("Installation cancelled".into());
        }
        if let Ok(l) = line {
            let _ = app.emit("install-log", &l);
        }
    }

    let stderr = child.stderr.take().unwrap();
    for line in std::io::BufReader::new(stderr).lines() {
        if let Ok(l) = line {
            let _ = app.emit("install-log", &l);
        }
    }

    {
        let mut guard = state.child_pid.lock().unwrap();
        *guard = None;
    }

    let status = child
        .wait()
        .map_err(|e| format!("{label}: wait failed — {e}"))?;

    if !status.success() {
        let code = status.code().unwrap_or(-1);
        return Err(format!("{label} failed (exit code {code})"));
    }

    app.emit("install-log", format!("✓ {label}")).ok();
    Ok(())
}

// ── Sidecar build ────────────────────────────────────────────────────

const SIDECAR_HIDDEN_IMPORTS: &[&str] = &[
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "starlette",
    "sr_engine.api.app",
    "sr_engine.api.routes",
    "sr_engine.models.archs",
    "torchvision",
    "lpips",
    "safetensors",
    "cv2",
    "pydantic",
];

fn install_sidecar(
    app: &AppHandle,
    state: &InstallState,
    env_dir: &Path,
    python_bin: &Path,
) -> Result<(), String> {
    run_step(
        app,
        state,
        Command::new("uv").args([
            "pip",
            "install",
            "--python",
            &python_bin.to_string_lossy(),
            "pyinstaller",
        ]),
        "Installing PyInstaller",
    )?;

    let sidecar_entry = find_bundled_sidecar_entry(app)
        .ok_or_else(|| "sidecar_entry.py not found in app resources".to_string())?;

    let build_dir = env_dir.join("build-tmp");
    std::fs::create_dir_all(&build_dir).ok();

    let mut py_cmd = Command::new(python_bin);
    py_cmd.args([
        "-m",
        "PyInstaller",
        "--name",
        "sr-engine",
        "--onedir",
        "-y",
        "--distpath",
        &build_dir.to_string_lossy(),
        "--workpath",
        &env_dir.join("pyi-work").to_string_lossy(),
        "--specpath",
        &env_dir.to_string_lossy(),
    ]);
    for hi in SIDECAR_HIDDEN_IMPORTS {
        py_cmd.args(["--hidden-import", hi]);
    }
    // Include YAML config files from the sr_engine package (not collected
    // automatically because they are non-Python data files).
    py_cmd.args(["--collect-data", "sr_engine"]);
    py_cmd.arg(&*sidecar_entry.to_string_lossy());

    run_step(app, state, &mut py_cmd, "Building sidecar binary")?;

    let built = build_dir.join("sr-engine");
    let final_dir = env_dir.join("sidecar");
    if built.is_dir() {
        if final_dir.exists() {
            std::fs::remove_dir_all(&final_dir).ok();
        }
        std::fs::rename(&built, &final_dir)
            .map_err(|e| format!("Failed to move sidecar: {e}"))?;
    } else {
        return Err("PyInstaller did not produce expected output directory".into());
    }

    let _ = std::fs::remove_dir_all(&build_dir);
    let _ = std::fs::remove_dir_all(env_dir.join("pyi-work"));
    let _ = std::fs::remove_file(env_dir.join("sr-engine.spec"));

    Ok(())
}

// ── Public: main install entry point ──────────────────────────────────

/// Returns the project root directory. Only meaningful in dev mode (when the
/// repo exists); in release builds it may point to a directory without pyproject.toml.
fn project_root() -> Option<PathBuf> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().map(|p| p.to_path_buf())
}

/// Decide how to install the sr_engine package into the venv.
///
/// - **Dev mode** (pyproject.toml exists): `uv pip install -e .` — reads
///   pyproject.toml, installs the project in editable mode plus all deps
///   (except torch which comes in a separate step).
/// - **Release mode** (no pyproject.toml): `uv pip install <bundled-wheel>`
///   — the wheel carries the same dependency metadata.
fn install_sr_package(
    app: &AppHandle,
    istate: &InstallState,
    python_bin: &Path,
) -> Result<(), String> {
    let project = project_root();
    let pyproject = project.as_ref().map(|p| p.join("pyproject.toml"));

    if let Some(ref pp) = pyproject {
        if pp.exists() {
            return run_step(
                app,
                istate,
                Command::new("uv")
                    .args([
                        "pip",
                        "install",
                        "--python",
                        &python_bin.to_string_lossy(),
                        "-e",
                        ".",
                    ])
                    .current_dir(pp.parent().unwrap()),
                "Installing SR Engine package (editable)",
            );
        }
    }

    let wheel = find_bundled_wheel(app)
        .ok_or_else(|| "Bundled sr_engine wheel not found (dist/sr_engine.whl missing)".to_string())?;
    run_step(
        app,
        istate,
        Command::new("uv").args([
            "pip",
            "install",
            "--python",
            &python_bin.to_string_lossy(),
            &wheel.to_string_lossy(),
        ]),
        "Installing SR Engine package",
    )
}

// ── ROCm install for Windows (user pre-created venv via AMD Adrenalin) ──

#[cfg(windows)]
fn install_rocm_windows(
    app: &AppHandle,
    istate: &InstallState,
    env_dir: &Path,
    venv_path: &str,
    backend: &str,
    env_type: &str,
) -> Result<(), String> {
    let venv_dir = Path::new(venv_path);
    let python_bin = venv_dir.join("Scripts").join("python.exe");

    // Re-verify the venv (belt-and-suspenders — frontend already checked)
    let info = verify_rocm_venv(venv_path);
    if !info.valid {
        return Err(info.error.unwrap_or_else(|| "Venv verification failed".into()));
    }
    if let Some(ref pv) = info.python_version {
        match pv.as_str() {
            "3.11" | "3.12" => {}
            _ => return Err(format!("Python {pv} is not supported (need 3.11 or 3.12)")),
        }
    }

    // If installed marker exists but venv is valid, it's a retry — clear marker
    if env_dir.join("installed").exists() {
        std::fs::remove_file(env_dir.join("installed")).ok();
        std::fs::remove_file(env_dir.join("env.json")).ok();
    }
    std::fs::create_dir_all(env_dir)
        .map_err(|e| format!("Cannot create env dir: {e}"))?;

    // Install sr_engine package (skips torch — already present in the venv)
    install_sr_package(app, istate, &python_bin)?;

    // Install lpips
    run_step(
        app,
        istate,
        Command::new("uv").args([
            "pip",
            "install",
            "--python",
            &python_bin.to_string_lossy(),
            "lpips",
        ]),
        "Installing LPIPS",
    )?;

    // Write metadata + marker
    let meta = EnvMeta {
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        backend: backend.to_string(),
        env_type: env_type.to_string(),
        env_path: venv_dir.to_string_lossy().to_string(),
        installed_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "unknown".to_string()),
    };

    let json =
        serde_json::to_string_pretty(&meta).map_err(|e| format!("Serialize env meta: {e}"))?;
    std::fs::write(env_dir.join("env.json"), &json)
        .map_err(|e| format!("Write env.json: {e}"))?;
    std::fs::write(env_dir.join("installed"), "")
        .map_err(|e| format!("Write installed marker: {e}"))?;

    app.emit("install-done", &meta).ok();
    Ok(())
}

#[allow(unused_variables)]
pub fn install_env(
    app: AppHandle,
    state: &AppHandle,
    backend: String,
    env_type: String,
    rocm_venv_path: Option<String>,
) -> Result<(), String> {
    let istate = state.state::<InstallState>();

    if istate.cancelled.swap(false, Ordering::SeqCst) {
        return Err("Installation cancelled".into());
    }

    let env_dir = get_env_dir();

    // ROCm on Windows: user pre-created the venv via AMD Adrenalin
    #[cfg(windows)]
    if backend == "rocm" {
        let venv_path = rocm_venv_path
            .unwrap_or_else(|| env_dir.join("venv").to_string_lossy().to_string());
        return install_rocm_windows(&app, &istate, &env_dir, &venv_path, &backend, &env_type);
    }

    // Clean any leftover state from a previous run so uv venv never sees an
    // existing directory (which would make it exit with code 2).
    if env_dir.join("installed").exists() {
        return Err("Environment is already installed. Delete the env dir or remove the marker first.".into());
    }
    if env_dir.exists() {
        std::fs::remove_dir_all(&env_dir)
            .map_err(|e| format!("Cannot clean previous env dir: {e}"))?;
    }

    std::fs::create_dir_all(&env_dir)
        .map_err(|e| format!("Cannot create env dir: {e}"))?;

    let venv_dir = env_dir.join("venv");

    // Step 1: create venv
    run_step(
        &app,
        &istate,
        Command::new("uv").args(["venv", &venv_dir.to_string_lossy()]),
        "Creating virtual environment",
    )?;

    let python_bin = if cfg!(windows) {
        venv_dir.join("Scripts").join("python.exe")
    } else {
        venv_dir.join("bin").join("python")
    };

    // Step 2: install sr_engine package + all non-torch deps
    install_sr_package(&app, &istate, &python_bin)?;

    // Step 3: install torch
    if let Some(index) = torch_index_url(&backend) {
        run_step(
            &app,
            &istate,
            Command::new("uv").args([
                "pip",
                "install",
                "--python",
                &python_bin.to_string_lossy(),
                "torch",
                "torchvision",
                "--index-url",
                index,
            ]),
            "Installing PyTorch",
        )?;
    } else {
        // MPS on macOS: standard PyPI index
        run_step(
            &app,
            &istate,
            Command::new("uv").args([
                "pip",
                "install",
                "--python",
                &python_bin.to_string_lossy(),
                "torch",
                "torchvision",
            ]),
            "Installing PyTorch",
        )?;
    }

    // Step 4: install lpips
    run_step(
        &app,
        &istate,
        Command::new("uv").args([
            "pip",
            "install",
            "--python",
            &python_bin.to_string_lossy(),
            "lpips",
        ]),
        "Installing LPIPS",
    )?;

    // Step 5: optional sidecar
    if env_type == "sidecar" {
        install_sidecar(&app, &istate, &env_dir, &python_bin)?;
    }

    // Step 6: write metadata + marker
    let meta = EnvMeta {
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        backend: backend.clone(),
        env_type: env_type.clone(),
        env_path: if env_type == "sidecar" {
            env_dir.join("sidecar").to_string_lossy().to_string()
        } else {
            venv_dir.to_string_lossy().to_string()
        },
        installed_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "unknown".to_string()),
    };

    let json =
        serde_json::to_string_pretty(&meta).map_err(|e| format!("Serialize env meta: {e}"))?;
    std::fs::write(env_dir.join("env.json"), &json)
        .map_err(|e| format!("Write env.json: {e}"))?;
    std::fs::write(env_dir.join("installed"), "")
        .map_err(|e| format!("Write installed marker: {e}"))?;

    app.emit("install-done", &meta).ok();
    Ok(())
}

// ── Public: cancel ─────────────────────────────────────────────────────

pub fn cancel_install(app: &AppHandle) {
    let state = app.state::<InstallState>();
    state.cancelled.store(true, Ordering::Relaxed);
    let pid = { *state.child_pid.lock().unwrap() };
    if let Some(pid) = pid {
        kill_pid(pid);
    }
}

// ── Public: read / update env meta ────────────────────────────────────

pub fn read_env_meta() -> Option<EnvMeta> {
    let path = get_env_dir().join("env.json");
    if !path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

pub fn maybe_update_env(app: &AppHandle, meta: &EnvMeta) -> Result<(), String> {
    let current = env!("CARGO_PKG_VERSION");
    if meta.app_version == current {
        return Ok(());
    }

    let wheel = find_bundled_wheel(app)
        .ok_or_else(|| "Bundled wheel not found for update".to_string())?;

    let env_path = Path::new(&meta.env_path);
    let python = if cfg!(windows) {
        env_path.join("Scripts").join("python.exe")
    } else {
        env_path.join("bin").join("python")
    };

    let output = Command::new("uv")
        .args([
            "pip",
            "install",
            "--python",
            &python.to_string_lossy(),
            "--reinstall",
            &wheel.to_string_lossy(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Update command failed: {e}"))?;

    if !output.status.success() {
        let msg = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Update failed: {msg}"));
    }

    let mut updated = meta.clone();
    updated.app_version = current.to_string();
    updated.installed_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let json = serde_json::to_string_pretty(&updated)
        .map_err(|e| format!("Serialize updated meta: {e}"))?;
    std::fs::write(get_env_dir().join("env.json"), &json)
        .map_err(|e| format!("Write updated env.json: {e}"))?;

    Ok(())
}

// ── Public: launch server from installed env ──────────────────────────

pub fn launch_from_env(meta: &EnvMeta) -> Result<Child, String> {
    let env_path = Path::new(&meta.env_path);

    match meta.env_type.as_str() {
        "venv" => {
            let python = if cfg!(windows) {
                env_path.join("Scripts").join("python.exe")
            } else {
                env_path.join("bin").join("python")
            };
            if !python.exists() {
                return Err(format!(
                    "Python not found at {} — environment may be corrupted",
                    python.display()
                ));
            }
            let mut cmd = Command::new(&python);
            cmd.args([
                "-m",
                "uvicorn",
                "sr_engine.api.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
                "--log-level",
                "info",
            ]);
            cmd.stdin(Stdio::null())
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit());

            #[cfg(unix)]
            unsafe {
                cmd.pre_exec(|| {
                    libc::setsid();
                    Ok(())
                });
            }

            cmd.spawn().map_err(|e| format!("Failed to launch venv server: {e}"))
        }

        "sidecar" => {
            let exe = if meta.env_path.ends_with("sidecar") {
                env_path.join("sr-engine")
            } else {
                env_path.to_path_buf()
            };
            if !exe.exists() {
                return Err(format!(
                    "Sidecar binary not found at {}",
                    exe.display()
                ));
            }
            let mut cmd = Command::new(&exe);
            cmd.stdin(Stdio::null())
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit());

            #[cfg(unix)]
            unsafe {
                cmd.pre_exec(|| {
                    libc::setsid();
                    Ok(())
                });
            }

            cmd.spawn().map_err(|e| format!("Failed to launch sidecar: {e}"))
        }

        other => Err(format!("Unknown env type: {other}")),
    }
}
