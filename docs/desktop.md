# Desktop Application (Tauri 2)

## Overview

sr-engine ships as a native desktop application built with **Tauri 2**. The Tauri shell (Rust) manages the Python backend server process, provides filesystem APIs, and hosts the React frontend as a webview.

The application follows a **two-process architecture**:

```
┌──────────────────────────────────────────────────┐
│                  Tauri Process                    │
│  (Rust — system tray, window, filesystem, IPC)   │
│                                                    │
│  ┌─────────────────┐  ┌────────────────────────┐  │
│  │  WebView         │  │  Python Server         │  │
│  │  (React/TS GUI)  │◄─┤  (uvicorn + FastAPI)   │  │
│  │  localhost:1420  │  │  localhost:8765         │  │
│  └─────────────────┘  └────────────────────────┘  │
│         │                       │                  │
│         └─────── HTTP ──────────┘                  │
└──────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Desktop Shell | Tauri 2 (Rust) | Window management, process lifecycle, filesystem API |
| Frontend | React 18 + TypeScript | UI rendered in webview |
| Backend | Python + FastAPI + Uvicorn | ML operations, data pipeline |
| Bundling | Tauri CLI | Native binary packaging |

---

## Directory Layout

```
src-tauri/
├── Cargo.toml              # Rust dependencies
├── Cargo.lock
├── build.rs                # Tauri build script
├── tauri.conf.json         # Window config, CSP, bundle settings
├── capabilities/
│   └── default.json        # Tauri 2 capability permissions
├── icons/                  # Application icons
├── gen/                    # Generated schemas
└── src/
    ├── main.rs             # Entry point (5 lines)
    └── lib.rs              # Tauri commands (196 lines)
```

---

## Rust Backend (lib.rs)

**File:** `src-tauri/src/lib.rs` (196 lines)

### Python Server Lifecycle

The Tauri shell manages the Python server as a child process:

```rust
#[tauri::command]
fn start_python_server(app_handle: tauri::AppHandle) -> Result<(), String> {
    // Spawns: uv run uvicorn sr_engine.api.app:app --host 127.0.0.1 --port 8765
    // Stores child process handle for later cleanup
}

#[tauri::command]
fn stop_python_server(app_handle: tauri::AppHandle) -> Result<(), String> {
    // Kills the Python server process
}
```

**Lifecycle:**
1. On app start: `start_python_server` is called, spawning `uv run uvicorn sr_engine.api.app:app` as a daemon child process
2. On window close: `stop_python_server` kills the child process
3. Auto-kill on crash: if the Tauri process exits unexpectedly, the child process is orphaned (OS-dependent)

### Filesystem Commands

Tauri commands exposed to the frontend for filesystem operations:

| Command | Signature | Description |
|---------|-----------|-------------|
| `list_dir` | `(path: String) -> Vec<FsEntry>` | List directory contents |
| `read_text_file` | `(path: String) -> String` | Read file as text |
| `write_text_file` | `(path: String, content: String)` | Write text to file |
| `create_dir_all` | `(path: String)` | Create directory (recursive) |
| `path_exists` | `(path: String) -> bool` | Check if path exists |
| `delete_file` | `(path: String)` | Delete a file |
| `delete_directory` | `(path: String)` | Delete a directory (recursive) |
| `copy_directory` | `(src: String, dst: String)` | Copy directory recursively |
| `list_image_files` | `(path: String) -> Vec<String>` | List image files in directory |
| `open_in_file_manager` | `(path: String)` | Open path in system file manager |

These commands are used by the frontend for:
- Browsing filesystem paths (dataset directories, video files, image files)
- Reading/writing project configuration files
- Creating workspace directories
- Opening outputs in the file manager

### Tauri 2 Plugins

| Plugin | Purpose |
|--------|---------|
| `tauri-plugin-dialog` | Native file open/save dialogs |
| `tauri-plugin-fs` | Extended filesystem access |

---

## Configuration (tauri.conf.json)

**File:** `src-tauri/tauri.conf.json`

Key settings:

```json
{
  "build": {
    "devUrl": "http://localhost:1420",
    "frontendDist": "../frontend/dist"
  },
  "app": {
    "windows": [
      {
        "title": "SR Tuner",
        "width": 1280,
        "height": 820,
        "resizable": true,
        "fullscreen": false
      }
    ]
  },
  "bundle": {
    "active": true,
    "icon": ["icons/32x32.png", "icons/128x128.png", "icons/128x128@2x.png", "icons/icon.icns", "icons/icon.ico"]
  },
  "security": {
    "csp": "default-src 'self'; connect-src 'self' http://localhost:8765; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:"
  }
}
```

### CSP Notes

The Content Security Policy allows:
- `connect-src 'self' http://localhost:8765` — API calls to the Python server
- `style-src 'self' 'unsafe-inline'` — Inline styles (used by React components)
- `img-src 'self' data: blob:` — Image display (inference results, validation frames)

---

## Building and Distribution

### Development

```bash
# Run in development mode (hot-reload for frontend, requires Python server separately)
cd frontend
npm run dev

# Or run Tauri dev mode (spawns both frontend and window)
npx tauri dev
```

### Production Build

```bash
# Build the Tauri desktop application
npx tauri build
```

This produces:
- **Linux:** `.deb`, `.AppImage`, or `.rpm` in `src-tauri/target/release/bundle/`
- **macOS:** `.dmg` in `src-tauri/target/release/bundle/`
- **Windows:** `.msi` or `.exe` in `src-tauri/target/release/bundle/`

The build process:
1. Compiles the React frontend (`npm run build` → `frontend/dist/`)
2. Compiles the Rust Tauri shell (via `cargo build`)
3. Embeds the frontend dist into the binary
4. Packages into platform-specific bundle format

### Building on Windows

**Prerequisites:**

- [Microsoft Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) or Visual Studio with "Desktop development with C++" workload
- [WebView2](https://developer.microsoft.com/en-us/microsoft-edge/webview2/) — ships with Windows 10 (version 1803+) and Windows 11
- Rust MSVC toolchain (installed automatically by `rustup` on Windows)
- Node.js >= 18

**Build command** (same as other platforms):

```powershell
npx tauri build
```

**Output:** `src-tauri/target/release/bundle/msi/SR Tuner_<version>_x64_en-US.msi` and/or an `.exe` installer.

**Note:** The Rust code already handles Windows-specific process management — the Tauri shell uses `taskkill /F /T /PID` to clean up the Python backend server, and `explorer` to open directories in File Explorer.

---

## Python Server Management

### How the Server is Started

The `start_python_server` Tauri command runs:

```bash
uv run uvicorn sr_engine.api.app:app --host 127.0.0.1 --port 8765
```

This requires:
- Python 3.11+ with `uv` installed
- `sr-engine` package installed in the `uv` environment
- All dependencies (PyTorch, etc.) available

### Port Conflicts

If port 8765 is already in use, the server will fail to start. The frontend's `useSSEConnection` hook detects the connection failure and shows a `ConnectionErrorDialog`.

### Cleanup

The server process is killed when:
- The user closes the application window
- The `stop_python_server` command is explicitly called
- The Tauri process exits

The Rust code handles child process cleanup for all platforms:
- **Linux/macOS:** Uses `libc::kill` with process group signals (SIGTERM → SIGKILL after 5s grace period)
- **Windows:** Uses `taskkill /F /T /PID` to forcefully terminate the process tree

---

## Security Considerations

### Network Isolation

- The Python server binds to `127.0.0.1` (localhost) by default
- The Tauri webview connects only to `localhost:8765` (enforced by CSP)
- No external network access is required for the API (the Python server does not accept external connections unless configured with `--host 0.0.0.0`)

### Filesystem Access

- Tauri commands use the `tauri-plugin-fs` plugin with capabilities defined in `capabilities/default.json`
- The frontend has access to the filesystem for workspace operations

### CSP

The Content Security Policy prevents XSS and data exfiltration by restricting:
- Script sources to the application itself
- Connection targets to localhost:8765
- Image sources to self, data URIs, and blob URIs

---

## See Also

- [Frontend Guide](frontend.md) — React GUI hosted in the webview
- [API Reference](api-reference.md) — REST API consumed by the frontend
- [Architecture Overview](architecture.md) — System-level architecture