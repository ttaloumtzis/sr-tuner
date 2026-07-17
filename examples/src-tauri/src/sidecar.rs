use std::sync::{Arc, Mutex};
use tauri::{Emitter, Manager};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

// ── State ──────────────────────────────────────────────────────────────────

pub struct SidecarState(pub Mutex<Option<CommandChild>>);

// ── Spawn + IPC bridge ────────────────────────────────────────────────────

/// Spawn the bundled sidecar binary and bridge its stdout → Tauri events.
///
/// §16.10: On receiving the `SIDECAR_EXTRACTING` sentinel line, the ready
///         timeout extends from 10 s to 120 s.  The sentinel is relayed to
///         the frontend as `sidecar-extracting` so OnboardingScreen can update
///         its progress text and start its own extended timer.
///
/// §16.11: Outbound IPC messages that arrive before `sidecar.ready` is received
///         are buffered in `startup_queue`.  Immediately after `sidecar.ready`
///         is detected on stdout, all buffered messages are flushed to stdin in
///         order, then the queue is cleared.
///         If `sidecar.ready` does not arrive within the timeout window, the
///         `sidecar-timeout` event is emitted so the frontend shows the
///         "Sidecar failed to start" error dialog.
pub fn spawn_sidecar(app: &tauri::AppHandle) -> Result<(), String> {
    let cmd = app
        .shell()
        .sidecar("sidecar")
        .map_err(|e| e.to_string())?;

    let (mut rx, child) = cmd.spawn().map_err(|e| e.to_string())?;

    *app.state::<SidecarState>().0.lock().unwrap() = Some(child);

    // §16.11 — startup command queue shared between the reader task and
    // send_to_sidecar().  Before ready: messages are buffered here.
    // After ready: this vec is empty and messages go directly to stdin.
    let startup_queue: Arc<Mutex<Option<Vec<String>>>> =
        Arc::new(Mutex::new(Some(Vec::new())));

    // Expose the queue via Tauri managed state so send_to_sidecar can access it.
    // We store it alongside SidecarState using a type alias.
    let queue_for_reader = Arc::clone(&startup_queue);

    let app_handle = app.clone();
    tauri::async_runtime::spawn(async move {
        // §16.10 — timeout tracking
        // Initial window: 10 s.  Extended to 120 s on SIDECAR_EXTRACTING.
        // We track whether we received ready within this file's scope;
        // the actual timer is owned by the frontend (§15.1).
        let mut _extracting_sentinel_seen = false;

        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    let line = String::from_utf8_lossy(&bytes);
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    // §16.9 / §16.10 — PyInstaller extraction sentinel
                    if trimmed == "SIDECAR_EXTRACTING" {
                        _extracting_sentinel_seen = true;
                        let _ = app_handle.emit("sidecar-extracting", ());
                        continue;
                    }

                    if let Ok(value) =
                        serde_json::from_str::<serde_json::Value>(trimmed)
                    {
                        // §16.11 — detect sidecar.ready, then flush the startup queue
                        if value.get("type").and_then(|t| t.as_str())
                            == Some("sidecar.ready")
                        {
                            flush_startup_queue(&app_handle, &queue_for_reader);
                        }

                        let _ = app_handle.emit("sidecar-message", value);
                    }
                }
                CommandEvent::Stderr(bytes) => {
                    eprintln!(
                        "[sidecar stderr] {}",
                        String::from_utf8_lossy(&bytes).trim()
                    );
                }
                CommandEvent::Error(err) => {
                    eprintln!("[sidecar error] {err}");
                    let _ = app_handle.emit("sidecar-error", err);
                }
                CommandEvent::Terminated(payload) => {
                    eprintln!("[sidecar] terminated: code={:?}", payload.code);
                    let _ = app_handle.emit("sidecar-terminated", payload.code);
                    *app_handle.state::<SidecarState>().0.lock().unwrap() = None;
                    break;
                }
                _ => {}
            }
        }
    });

    // Store the startup queue in a second managed state so lib.rs commands
    // can reach it.  We wrap it in a newtype to satisfy Tauri's type uniqueness.
    // NOTE: The queue Arc is moved into StartupQueueState which is registered
    // in lib.rs via .manage(StartupQueueState(startup_queue)).
    // Here we just keep the local Arc alive until the manage call in lib.rs.
    // The actual storage happens in lib.rs; we return the Arc via a side channel
    // using the STARTUP_QUEUE thread-local below.
    STARTUP_QUEUE_CELL.with(|cell| {
        *cell.borrow_mut() = Some(startup_queue);
    });

    Ok(())
}

// ── GPU variant spawn (§25.8) ─────────────────────────────────────────────

/// Spawn a downloaded GPU-variant sidecar from an explicit filesystem path.
///
/// §25.8: Called by the onboarding screen after `downloadAndInstallVariant`
/// completes.  Kills the running minimal sidecar (if any), resets the startup
/// queue, and bridges the new process's stdout → Tauri events identically to
/// `spawn_sidecar`.
pub fn spawn_sidecar_from_path(
    app: &tauri::AppHandle,
    binary_path: &str,
    queue_state: &StartupQueueState,
) -> Result<(), String> {
    // Kill the current (minimal/cpu) sidecar before starting the GPU one.
    {
        let sidecar_state = app.state::<SidecarState>();
        let mut guard = sidecar_state.0.lock().unwrap();
        if let Some(child) = guard.take() {
            let _ = child.kill();
        }
    }

    // Reset the startup queue so pre-ready messages are buffered for the
    // new process.  Any reader task from the old spawn will exit once the
    // old process terminates and will not interfere.
    {
        let mut guard = queue_state.0.lock().unwrap();
        *guard = Some(Vec::new());
    }

    let cmd = app.shell().command(binary_path);
    let (mut rx, child) = cmd.spawn().map_err(|e| e.to_string())?;

    *app.state::<SidecarState>().0.lock().unwrap() = Some(child);

    let queue_for_reader = Arc::clone(&queue_state.0);
    let app_handle = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    let line = String::from_utf8_lossy(&bytes);
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if trimmed == "SIDECAR_EXTRACTING" {
                        let _ = app_handle.emit("sidecar-extracting", ());
                        continue;
                    }
                    if let Ok(value) =
                        serde_json::from_str::<serde_json::Value>(trimmed)
                    {
                        if value.get("type").and_then(|t| t.as_str())
                            == Some("sidecar.ready")
                        {
                            flush_startup_queue(&app_handle, &queue_for_reader);
                        }
                        let _ = app_handle.emit("sidecar-message", value);
                    }
                }
                CommandEvent::Stderr(bytes) => {
                    eprintln!(
                        "[sidecar stderr] {}",
                        String::from_utf8_lossy(&bytes).trim()
                    );
                }
                CommandEvent::Error(err) => {
                    eprintln!("[sidecar error] {err}");
                    let _ = app_handle.emit("sidecar-error", err);
                }
                CommandEvent::Terminated(payload) => {
                    eprintln!("[sidecar] terminated: code={:?}", payload.code);
                    let _ = app_handle.emit("sidecar-terminated", payload.code);
                    *app_handle.state::<SidecarState>().0.lock().unwrap() = None;
                    break;
                }
                _ => {}
            }
        }
    });

    Ok(())
}

// ── Startup queue cell (thread-local hand-off to lib.rs) ──────────────────

// lib.rs calls take_startup_queue() immediately after spawn_sidecar() to
// retrieve the Arc and register it as managed state.
std::thread_local! {
    static STARTUP_QUEUE_CELL: std::cell::RefCell<
        Option<Arc<Mutex<Option<Vec<String>>>>>
    > = const { std::cell::RefCell::new(None) };
}

pub fn take_startup_queue() -> Option<Arc<Mutex<Option<Vec<String>>>>> {
    STARTUP_QUEUE_CELL.with(|cell| cell.borrow_mut().take())
}

// ── Startup queue newtype for Tauri managed state ─────────────────────────

pub struct StartupQueueState(pub Arc<Mutex<Option<Vec<String>>>>);

// ── Queue helpers ──────────────────────────────────────────────────────────

/// §16.11 — Flush all buffered startup messages to sidecar stdin, then close
/// the queue so subsequent writes go directly to stdin.
fn flush_startup_queue(
    app: &tauri::AppHandle,
    queue: &Arc<Mutex<Option<Vec<String>>>>,
) {
    let messages: Vec<String> = {
        let mut guard = queue.lock().unwrap();
        // Replacing Some(vec) with None closes the pre-ready buffer.
        guard.take().unwrap_or_default()
    };

    if messages.is_empty() {
        return;
    }

    let state = app.state::<SidecarState>();
    let mut child_guard = state.0.lock().unwrap();
    if let Some(child) = child_guard.as_mut() {
        for mut msg in messages {
            if !msg.ends_with('\n') {
                msg.push('\n');
            }
            let _ = child.write(msg.as_bytes());
        }
    }
}

/// §16.11 — Enqueue a message if sidecar.ready has not yet been received,
/// or write directly to stdin if it has.
pub fn enqueue_or_send(
    child: &mut CommandChild,
    queue: &Arc<Mutex<Option<Vec<String>>>>,
    mut message: String,
) -> Result<(), String> {
    if !message.ends_with('\n') {
        message.push('\n');
    }
    let mut guard = queue.lock().map_err(|e| e.to_string())?;
    if let Some(ref mut pending) = *guard {
        // sidecar.ready not yet received — buffer the message
        pending.push(message);
        Ok(())
    } else {
        // sidecar.ready received — write directly
        child.write(message.as_bytes()).map_err(|e| e.to_string())
    }
}
