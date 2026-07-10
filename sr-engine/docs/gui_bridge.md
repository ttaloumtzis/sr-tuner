# Server GUI Bridge

The TCP/JSON bridge between the sr-engine backend and your Godot (C#) GUI client.
Allows starting/stopping training and inference jobs, querying workspace state, and
receiving real-time progress and training metrics — all over a single persistent TCP
connection.

---

## Architecture

```
┌──────────────────────┐     TCP / NDJSON      ┌──────────────────────────────┐
│   Godot GUI (C#)     │◄──────────────────────►│   sr-engine Server           │
│                      │    persistent conn     │                              │
│  SrEngineClient      │    port 8765 (default) │  Server                      │
│    ↕ event-driven    │                        │    ├─ gui_listener :8765     │
│    ↕ _Process queue  │                        │    ├─ job_listener :random   │
│                      │                        │    ├─ ClientHandler(s)       │
│                      │                        │    ├─ ControlHandler(s)      │
│                      │                        │    └─ JobManager             │
└──────────────────────┘                        └──────────┬───────────────────┘
                                                           │
                                  TCP / NDJSON             │
                                  control socket           │
                                                           ▼
                                                  ┌──────────────────┐
                                                  │  Subprocess      │
                                                  │  (srengine       │
                                                  │   train/infer/   │
                                                  │   dataset.build) │
                                                  │                  │
                                                  │  SocketReporter  │
                                                  │  SocketCallback  │
                                                  └──────────────────┘
```

Three tiers:

- **GUI Client** (your Godot app) — opens one persistent TCP connection to the
  Server, sends JSON request lines, receives JSON response + event lines.
- **Server** — long-lived process (`srengine serve start`). Accepts GUI client
  connections, dispatches commands, spawns subprocesses for long-running work,
  and relays subprocess events back to all connected GUI clients.
- **Subprocess** — spawned by `JobManager` for each `train.start`, `infer.start`,
  or `dataset.build` command. Connects back to the Server's job listener via a
  control socket, sends progress and training events over it.

---

## Quick Start

```bash
# Terminal 1 — start the server
srengine serve start --port 8765

# Terminal 2 — test with netcat
echo '{"id":"1","command":"hello"}' | nc 127.0.0.1 8765
# Response:
# {"id":"1","type":"result","data":{"schema_version":1,"server_version":"0.1.0"}}
```

---

## Starting the Server

```bash
srengine serve start [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | `8765` | TCP port for GUI clients |
| `--host` | `127.0.0.1` | Bind address |

The server resolves the workspace automatically via:
1. `--workspace` CLI option
2. `SRENGINE_WORKSPACE` environment variable
3. Walking up from CWD looking for a `.sr_workspace` marker file

If no workspace is found, the server refuses to start.

---

## Wire Protocol

- **Transport:** Raw TCP.
- **Framing:** Newline-delimited JSON (NDJSON). Every message is a single line
  terminated by `\n`. Messages larger than 65,536 bytes are split across reads
  (the server reassembles them internally).
- **Encoding:** UTF-8.
- **TLS:** Not supported. The server binds to `127.0.0.1` by default for local
  consumption.

### Request Format (Client → Server)

```json
{"id": "<string>", "command": "<string>", "params": { ... }}
```

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Request identifier echoed back in the response. Use a unique value per request (e.g. `"req_1"`, GUID). |
| `command` | Yes | One of the command names below. |
| `params` | No | Command-specific parameters (may be omitted). |

### Response Format (Server → Client)

**Successful synchronous command:**
```json
{"id": "<request_id>", "type": "result", "data": { ... }}
```

**Accepted asynchronous command:**
```json
{"id": "<request_id>", "type": "accepted", "data": {"status": "accepted", "job_id": "<job_id>"}}
```

**Error response:**
```json
{"id": "<request_id>", "type": "error", "message": "<human-readable>", "error_type": "<ExceptionName>"}
```

### Unsolicited Events (Server → Client)

The server pushes events to **all** connected GUI clients without a corresponding
request. Every event includes a `type` field and a `job_id` field identifying
the originating job. These arrive at any time on the same connection.

---

## Command Reference

### Synchronous Commands — response type `result`

#### `hello`

Handshake. Should be the first command sent after connecting.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Any unique ID |
| `command` | `"hello"` | |

**Response data:**
```json
{"schema_version": 1, "server_version": "0.1.0"}
```

---

#### `workspace.info`

Returns the resolved workspace path.

**Params:** none

**Response data:**
```json
{"workspace": "/path/to/workspace"}
```

---

#### `workspace.check`

Checks whether the workspace directory exists on disk.

**Params:** none

**Response data:**
```json
{"exists": true, "workspace": "/path/to/workspace"}
```

---

#### `project.list`

Lists all projects in the workspace.

**Params:** none

**Response data:**
```json
{"projects": [{"name": "my_project", "path": "/path/to/workspace/projects/my_project"}, ...]}
```

---

#### `project.create`

Creates a new project in the workspace.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Project name |

**Response data:**
```json
{"project": "my_project", "status": "created"}
```

Or on error:
```json
{"status": "error", "message": "missing 'name' parameter"}
```

---

#### `dataset.validate`

Runs a deep validation scan on a dataset directory.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | yes | Dataset directory path (must contain `HR/` and `LR/`) |

**Response data:**
```json
{"ok": true, "num_pairs": 1000, "problems": []}
```

If validation finds issues:
```json
{"ok": false, "num_pairs": 998, "problems": ["Missing LR for frame_042.png", "HR file frame_123.png has mismatched dimensions"]}
```

---

#### `dataset.health`

Profiles a dataset's spatial properties and detects black/corrupt frames.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | yes | Dataset directory path (must contain `HR/`) |

**Response data:**
```json
{
  "total_images": 1000,
  "resolutions": {"1920x1080": 800, "1280x720": 200},
  "aspect_ratios": {"1.78": 800, "1.6": 200},
  "channels": {"RGB (3 channels)": 1000},
  "computed_threshold": 3.5,
  "black_frames": ["frame_042.png", "frame_099.png"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `total_images` | int | Total HR images analyzed |
| `resolutions` | object | `{"WxH": count}` map |
| `aspect_ratios` | object | `{"ratio": count}` map |
| `channels` | object | `{"description": count}` map |
| `computed_threshold` | float | Adaptive brightness threshold used |
| `black_frames` | string[] | Filenames that fall below the threshold |

If the dataset has no `HR/` directory:
```json
{"error": "HR directory not found. Run validation/build first."}
```

---

#### `model.info`

Returns information about a model configuration.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | yes | Model name |

**Response data:**
```json
{"model": "rrdb_esrgan"}
```

(Current implementation echoes the model name — the payload may expand in future versions.)

---

#### `job.cancel`

Cancels a running job by sending SIGTERM to its subprocess.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `job_id` | string | yes | Job ID (returned by async commands) |

**Response data:**
```json
{"status": "cancelling", "job_id": "train_1747000000_a1b2c3d4"}
```

If the job is not found:
```json
{"status": "not_found", "job_id": "nonexistent"}
```

---

#### `job.list`

Lists all completed/cancelled/failed jobs from the workspace job manifests directory.

**Params:** none

**Response data:**
```json
{
  "jobs": [
    {
      "job_id": "train_1747000000_a1b2c3d4",
      "job_type": "train",
      "status": "completed",
      "pid": 12345,
      "started_at": "2025-05-16T12:00:00Z",
      "finished_at": "2025-05-16T13:30:00Z",
      "exit_code": 0,
      "project": "my_project",
      "log_path": null,
      "error_message": null
    }
  ]
}
```

---

#### `job.status`

Gets the manifest for a specific job.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `job_id` | string | yes | Job ID |

**Response data** (job found — same manifest shape as `job.list`):
```json
{
  "job_id": "train_1747000000_a1b2c3d4",
  "job_type": "train",
  "status": "completed",
  ...
}
```

If not found:
```json
{"job_id": "nonexistent", "status": "not_found"}
```

---

### Asynchronous Commands — response type `accepted`

These commands spawn a subprocess and return immediately with a `job_id`.
The actual work runs in the background. Progress and results arrive as
unsolicited events (see Unsolicited Events section).

---

#### `train.start`

Start training a super-resolution model.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `model_name` | string | no | Model name (e.g. `"rrdb_esrgan"`, `"swinir"`) |
| `dataset` | string | no | Dataset path or name |
| `config` | string | no | Path to training config YAML |
| `resume` | string | no | Path to checkpoint to resume from |
| `device` | string | no | `"cuda"`, `"cpu"`, or `"auto"` |
| `batch_size` | int | no | Batch size |
| `learning_rate` | float | no | Learning rate |
| `max_epochs` | int | no | Maximum epochs |
| `project` | string | no | Project name (requires workspace) |
| `machine` | bool | no | Enable machine-readable metrics output |
| `experiment_id` | string | no | Experiment identifier |

**Response data:**
```json
{"status": "accepted", "job_id": "train_1747000000_a1b2c3d4"}
```

---

#### `infer.start`

Run super-resolution inference on an image or video.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | yes | Model checkpoint path |
| `input_path` | string | yes | Input image or video path |
| `output` | string | yes | Output path |
| `tile` | int | no | Tile size for tiled inference (default: 512) |
| `overlap` | int | no | Tile overlap in pixels (default: 64) |
| `device` | string | no | `"cuda"`, `"cpu"`, or `"auto"` |

**Response data:**
```json
{"status": "accepted", "job_id": "infer_1747000100_b5e6f7a8"}
```

---

#### `dataset.build`

Build a dataset from a video file or validate a preprocessed directory.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `input` | string | yes | Input video file or preprocessed dataset directory |
| `out` | string | no | Output dataset directory (required if input is a video) |
| `config` | string | no | Dataset config YAML path |

**Response data:**
```json
{"status": "accepted", "job_id": "dataset_build_1747000200_c9d0e1f2"}
```

---

## Unsolicited Events

All events are broadcast to every connected GUI client. Every event includes a
`job_id` field.

### Job Lifecycle Events

#### `log`

Relayed from the subprocess stdout.

```json
{"type": "log", "level": "info", "message": "Training Epoch 1/100", "job_id": "train_1747000000_a1b2c3d4"}
```

| Field | Possible values |
|-------|-----------------|
| `level` | `"info"`, `"warning"` |

#### `done`

Emitted when a subprocess exits.

```json
{"type": "done", "exit_code": 0, "elapsed_seconds": null, "job_id": "train_1747000000_a1b2c3d4"}
```

| `exit_code` | Constant | Meaning |
|-------------|----------|---------|
| `0` | `EXIT_SUCCESS` | Completed successfully |
| `1` | `EXIT_ERROR` | Generic error |
| `130` | `EXIT_CANCELLED` | Cancelled via SIGTERM / keyboard interrupt |

### Progress Events (SocketReporter)

These come from any command that uses `resolve_reporter()`. The `total` field
may be `null` when the total count is unknown.

```json
{"type": "progress_start", "total": 100, "desc": "Epoch 1/10", "job_id": "train_..."}
{"type": "progress_update", "n": 1, "job_id": "train_..."}
{"type": "progress_end", "job_id": "train_..."}
{"type": "postfix", "desc": "Processing...", "loss": 0.05, "job_id": "train_..."}
```

| Type | Extra fields |
|------|--------------|
| `progress_start` | `total` (int or null), `desc` (string) |
| `progress_update` | `n` (int, default 1) |
| `progress_end` | none |
| `postfix` | `desc` (string), plus any key=value pairs |

### Training Events (SocketCallback / Trainer)

These only appear during `train.start` jobs, emitted by the `SocketCallback`
hooked into the `Trainer`.

#### `phase`

```json
{"type": "phase", "phase": "training", "max_epochs": 100, "job_id": "train_..."}
{"type": "phase", "phase": "saving", "epoch": 25, "job_id": "train_..."}
{"type": "phase", "phase": "complete", "job_id": "train_..."}
{"type": "phase", "phase": "cancelled", "epoch": 25, "job_id": "train_..."}
```

| `phase` | When emitted |
|---------|--------------|
| `"training"` | Start of training loop, once per job |
| `"saving"` | Before saving a checkpoint |
| `"complete"` | Training finished successfully |
| `"cancelled"` | Training cancelled mid-way |

#### `step`

Emitted every `metrics_frequency` batches during training.

```json
{
  "type": "step",
  "epoch": 1,
  "batch": 10,
  "total_batches": 100,
  "pixel": 0.05,
  "total": 0.05,
  "lr": 0.0001,
  "job_id": "train_..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `epoch` | int | Current epoch (1-indexed) |
| `batch` | int | Current batch within epoch (1-indexed) |
| `total_batches` | int | Total batches per epoch |
| `pixel` | float | Pixel loss value |
| `perceptual` | float | Perceptual loss (only if enabled in config) |
| `total` | float | Combined loss |
| `lr` | float | Current learning rate |

#### `validate`

Emitted after each validation pass (controlled by `save_per_epoch`).

```json
{"type": "validate", "epoch": 1, "psnr": 30.2, "ssim": 0.89, "job_id": "train_..."}
```

| Field | Type | Description |
|-------|------|-------------|
| `epoch` | int | Epoch number |
| `psnr` | float | Peak Signal-to-Noise Ratio |
| `ssim` | float | Structural Similarity Index |

---

## Job Lifecycle

```
accept ─► pending ─► running ─► completed
                          │            │
                          │            ├─ exit_code 0
                          │            │
                          ├────────────┤
                          │            ├─ exit_code 130 → cancelled
                          │            │
                          └────────────┤
                                       └─ exit_code 1+ → failed
```

1. **accept** — Server receives async command, calls `JobManager.start_job()`.
2. **pending** — Subprocess is spawned. Server waits for it to connect the
   control socket. A 10-second timer fires if the subprocess never connects,
   broadcasting a warning log.
3. **running** — Subprocess connects control socket, sends `hello` with
   matching `job_id` + `token`. Events and logs start flowing.
4. **completed / cancelled / failed** — Subprocess exits. `JobManager` maps
   the exit code to a status, writes a manifest file to
   `<workspace>/jobs/<job_id>.json`, and broadcasts a `done` event.

### Manifest File

Persisted as JSON at `<workspace>/jobs/<job_id>.json`:

```json
{
  "job_id": "train_1747000000_a1b2c3d4",
  "job_type": "train",
  "status": "completed",
  "pid": 12345,
  "started_at": "2025-05-16T12:00:00Z",
  "finished_at": "2025-05-16T13:30:00Z",
  "exit_code": 0,
  "project": "my_project",
  "log_path": null,
  "error_message": null
}
```

Manifests persist across server restarts. Use `job.list` and `job.status` to
query them.

---

## Subprocess Integration (for reference)

When the server spawns a subprocess (`srengine train run`, etc.), it sets the
environment variable `SRENGINE_GUI_SOCKET` to a JSON string:

```json
{
  "job_id": "train_1747000000_a1b2c3d4",
  "token": "<32-byte-hex>",
  "control_host": "127.0.0.1",
  "control_port": 45091
}
```

The subprocess calls `connect_control_socket()` which:
1. Opens a TCP connection to `control_host:control_port`
2. Sends `{"type": "hello", "job_id": "...", "token": "..."}`
3. Waits for `{"status": "ok"}` ack (or raises `ConnectionRefusedError` if
   rejected)
4. Returns a `(job_id, send_fn, close_fn)` tuple

After handshake, the subprocess uses `SocketReporter` (for progress bars) and
`SocketCallback` (for training lifecycle) to send events over the control
socket. The server's `ControlHandler` receives these and broadcasts them to
all GUI clients.

The helpers in `sr_engine/cli/helpers.py` provide three integration points:

| Function | Returns when GUI socket is set | Returns otherwise |
|----------|-------------------------------|-------------------|
| `resolve_reporter()` | `SocketReporter` → sends progress events | `TqdmReporter` (terminal bar) |
| `resolve_callbacks()` | `[SocketCallback]` | `[]` (empty) |
| `resolve_cancel_check()` | Installs SIGTERM handler, returns `was_cancelled` | `lambda: False` |

---

## Godot C# Client Implementation

Here's a complete C# client class for Godot 4.x. It runs the network I/O on a
background thread and delivers messages to the main thread via `ConcurrentQueue`,
consumed in `_Process`.

### SrEngineClient.cs

```csharp
using Godot;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

public partial class SrEngineClient : Node
{
    private TcpClient _tcp;
    private NetworkStream _stream;
    private Thread _receiveThread;
    private CancellationTokenSource _cts;
    private int _nextRequestId = 1;

    // Incoming messages queued for main-thread dispatch via _Process
    private readonly ConcurrentQueue<Dictionary<string, JsonElement>> _incoming
        = new ConcurrentQueue<Dictionary<string, JsonElement>>();

    // Pending requests awaiting a response by request ID
    private readonly ConcurrentDictionary<string, TaskCompletionSource<Dictionary<string, JsonElement>>>
        _pending = new ConcurrentDictionary<string, TaskCompletionSource<Dictionary<string, JsonElement>>>();

    private readonly object _sendLock = new();

    // ── Signals (Godot events for the scene tree) ──────────────────────

    [Signal] public delegate void ConnectedEventHandler();
    [Signal] public delegate void DisconnectedEventHandler();
    [Signal] public delegate void ErrorReceivedEventHandler(string requestId, string message, string errorType);

    // Unsolicited events
    [Signal] public delegate void JobLogEventHandler(string jobId, string level, string message);
    [Signal] public delegate void JobDoneEventHandler(string jobId, int exitCode);
    [Signal] public delegate void ProgressStartEventHandler(string jobId, int? total, string desc);
    [Signal] public delegate void ProgressUpdateEventHandler(string jobId, int n);
    [Signal] public delegate void ProgressEndEventHandler(string jobId);
    [Signal] public delegate void PostfixEventHandler(string jobId, string desc);
    [Signal] public delegate void TrainingPhaseEventHandler(string jobId, string phase);
    [Signal] public delegate void TrainingStepEventHandler(string jobId, int epoch, int batch, int totalBatches, float loss);
    [Signal] public delegate void TrainingValidateEventHandler(string jobId, int epoch, float psnr, float ssim);

    // ── Connection ─────────────────────────────────────────────────────

    public async Task ConnectAsync(string host, int port)
    {
        _cts = new CancellationTokenSource();
        _tcp = new TcpClient();
        await _tcp.ConnectAsync(host, port);
        _stream = _tcp.GetStream();

        _receiveThread = new Thread(ReceiveLoop) { IsBackground = true };
        _receiveThread.Start();

        // Send hello handshake
        var result = await SendCommandAsync("hello", new Dictionary<string, JsonElement>());
        int schemaVersion = result["schema_version"].GetInt32();
        GD.Print($"Connected. Schema version: {schemaVersion}");
        CallDeferred(MethodName.EmitSignal, SignalName.Connected);
    }

    public void Disconnect()
    {
        _cts?.Cancel();
        _stream?.Close();
        _tcp?.Close();
        _receiveThread?.Join(1000);
        CallDeferred(MethodName.EmitSignal, SignalName.Disconnected);
    }

    // ── Send / Receive ─────────────────────────────────────────────────

    public async Task<Dictionary<string, JsonElement>> SendCommandAsync(
        string command, Dictionary<string, JsonElement> paramsDict = null)
    {
        var requestId = $"req_{_nextRequestId++}";
        var request = new Dictionary<string, object>
        {
            ["id"] = requestId,
            ["command"] = command,
        };
        if (paramsDict != null && paramsDict.Count > 0)
            request["params"] = paramsDict;

        var json = JsonSerializer.Serialize(request);
        var tcs = new TaskCompletionSource<Dictionary<string, JsonElement>>();
        _pending[requestId] = tcs;

        lock (_sendLock)
        {
            var data = Encoding.UTF8.GetBytes(json + "\n");
            _stream.Write(data, 0, data.Length);
        }

        return await tcs.Task;
    }

    // ── High-level job methods ─────────────────────────────────────────

    public async Task<string> StartTrainAsync(Dictionary<string, JsonElement> paramsDict)
    {
        var result = await SendCommandAsync("train.start", paramsDict);
        return result["job_id"].GetString();
    }

    public async Task<string> StartInferAsync(Dictionary<string, JsonElement> paramsDict)
    {
        var result = await SendCommandAsync("infer.start", paramsDict);
        return result["job_id"].GetString();
    }

    public async Task<string> StartDatasetBuildAsync(Dictionary<string, JsonElement> paramsDict)
    {
        var result = await SendCommandAsync("dataset.build", paramsDict);
        return result["job_id"].GetString();
    }

    public async Task<Dictionary<string, JsonElement>> GetJobStatusAsync(string jobId)
    {
        var p = new Dictionary<string, JsonElement>
        {
            ["job_id"] = JsonSerializer.SerializeToElement(jobId)
        };
        return await SendCommandAsync("job.status", p);
    }

    public async Task<List<Dictionary<string, JsonElement>>> ListJobsAsync()
    {
        var result = await SendCommandAsync("job.list");
        var jobs = new List<Dictionary<string, JsonElement>>();
        foreach (var item in result["jobs"].EnumerateArray())
            jobs.Add(JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(item.GetRawText()));
        return jobs;
    }

    // ── Main-thread event dispatch ─────────────────────────────────────

    public override void _Process(double delta)
    {
        while (_incoming.TryDequeue(out var msg))
        {
            DispatchMessage(msg);
        }
    }

    private void DispatchMessage(Dictionary<string, JsonElement> msg)
    {
        msg.TryGetValue("type", out var typeEl);
        string type = typeEl.GetString() ?? "";

        switch (type)
        {
            case "result":
            case "accepted":
                DispatchResponse(msg);
                return;

            case "error":
                DispatchError(msg);
                return;

            // ── Unsolicited events ──
            case "log":
                EmitSignal(SignalName.JobLog,
                    msg["job_id"].GetString(),
                    msg["level"].GetString(),
                    msg["message"].GetString());
                return;

            case "done":
                EmitSignal(SignalName.JobDone,
                    msg["job_id"].GetString(),
                    msg["exit_code"].GetInt32());
                return;

            case "progress_start":
                EmitSignal(SignalName.ProgressStart,
                    msg["job_id"].GetString(),
                    msg.ContainsKey("total") && msg["total"].ValueKind != JsonValueKind.Null
                        ? msg["total"].GetInt32() : (int?)null,
                    msg["desc"].GetString());
                return;

            case "progress_update":
                EmitSignal(SignalName.ProgressUpdate,
                    msg["job_id"].GetString(),
                    msg["n"].GetInt32());
                return;

            case "progress_end":
                EmitSignal(SignalName.ProgressEnd,
                    msg["job_id"].GetString());
                return;

            case "postfix":
                EmitSignal(SignalName.Postfix,
                    msg["job_id"].GetString(),
                    msg["desc"].GetString());
                return;

            case "phase":
                EmitSignal(SignalName.TrainingPhase,
                    msg["job_id"].GetString(),
                    msg["phase"].GetString());
                return;

            case "step":
                EmitSignal(SignalName.TrainingStep,
                    msg["job_id"].GetString(),
                    msg["epoch"].GetInt32(),
                    msg["batch"].GetInt32(),
                    msg["total_batches"].GetInt32(),
                    msg["total"].GetSingle());
                return;

            case "validate":
                EmitSignal(SignalName.TrainingValidate,
                    msg["job_id"].GetString(),
                    msg["epoch"].GetInt32(),
                    msg["psnr"].GetSingle(),
                    msg["ssim"].GetSingle());
                return;
        }
    }

    private void DispatchResponse(Dictionary<string, JsonElement> msg)
    {
        string requestId = msg["id"].GetString();
        if (_pending.TryRemove(requestId, out var tcs))
            tcs.TrySetResult(msg["data"]);
    }

    private void DispatchError(Dictionary<string, JsonElement> msg)
    {
        string requestId = msg["id"].GetString();
        string errorMsg = msg["message"].GetString();
        string errorType = msg["error_type"].GetString();

        if (_pending.TryRemove(requestId, out var tcs))
            tcs.TrySetException(new Exception($"{errorType}: {errorMsg}"));

        EmitSignal(SignalName.ErrorReceived, requestId, errorMsg, errorType);
    }

    // ── Background receive loop ────────────────────────────────────────

    private void ReceiveLoop()
    {
        var buffer = new byte[65536];
        var leftover = new StringBuilder();

        try
        {
            while (!_cts.Token.IsCancellationRequested)
            {
                int bytesRead = _stream.Read(buffer, 0, buffer.Length);
                if (bytesRead == 0) break;

                leftover.Append(Encoding.UTF8.GetString(buffer, 0, bytesRead));
                string data = leftover.ToString();

                int newlineIdx;
                while ((newlineIdx = data.IndexOf('\n')) >= 0)
                {
                    string line = data.Substring(0, newlineIdx).Trim();
                    if (line.Length > 0)
                    {
                        var msg = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(line);
                        _incoming.Enqueue(msg);
                    }
                    data = data.Substring(newlineIdx + 1);
                }

                leftover.Clear();
                leftover.Append(data);
            }
        }
        catch (Exception ex) when (ex is IOException || ex is ObjectDisposedException)
        {
            // Connection closed
        }

        CallDeferred(MethodName.EmitSignal, SignalName.Disconnected);
    }

    public override void _ExitTree()
    {
        Disconnect();
    }
}
```

### Usage in a Godot scene

```csharp
public partial class MyGui : Node
{
    private SrEngineClient _client;

    public override void _Ready()
    {
        _client = new SrEngineClient();
        AddChild(_client);

        _client.Connected += () => GD.Print("Connected to sr-engine!");
        _client.JobLog += (jobId, level, msg) =>
            GD.PrintRich($"[{level}] [{jobId}] {msg}");
        _client.TrainingStep += (jobId, epoch, batch, total, loss) =>
            GD.Print($"Epoch {epoch}: batch {batch}/{total}, loss={loss:F4}");
        _client.JobDone += (jobId, exitCode) =>
            GD.Print($"Job {jobId} finished with code {exitCode}");

        _ = ConnectAndStartJob();
    }

    private async Task ConnectAndStartJob()
    {
        try
        {
            await _client.ConnectAsync("127.0.0.1", 8765);

            var trainParams = new Dictionary<string, JsonElement>
            {
                ["model_name"] = JsonSerializer.SerializeToElement("rrdb_esrgan"),
                ["dataset"] = JsonSerializer.SerializeToElement("/data/my_dataset"),
                ["max_epochs"] = JsonSerializer.SerializeToElement(50),
                ["project"] = JsonSerializer.SerializeToElement("my_project"),
            };

            string jobId = await _client.StartTrainAsync(trainParams);
            GD.Print($"Training started: {jobId}");
        }
        catch (Exception ex)
        {
            GD.PrintErr($"Failed: {ex.Message}");
        }
    }

    public override void _ExitTree()
    {
        _client?.Disconnect();
    }
}
```

---

## Error Handling

### Error Response

Every command can fail with an `error` type response:

```json
{"id": "req_1", "type": "error", "message": "Something went wrong", "error_type": "ValueError"}
```

The `error_type` field contains the Python exception class name.

### Common Failure Scenarios

| Scenario | What happens |
|----------|-------------|
| Malformed JSON | Server responds with `{"type": "error", "message": "Malformed JSON"}` (request ID defaults to `"0"`) |
| Missing `command` field | `{"type": "error", "message": "Missing 'command' field"}` |
| Unknown command | `{"type": "error", "message": "Unknown command: foo"}` |
| Handler exception | Caught, sent as error with the exception type name |
| Subprocess hello timeout | Server broadcasts `{"type": "log", "level": "warning", "message": "Job ...: subprocess did not connect control socket within 10s"}` |
| Dead GUI client | Server silently removes client from broadcast list on next `broadcast()` call |
| Connection drop | Receive loop exits cleanly; no reconnection is attempted (implement in your client) |

### Reconnection

The C# client above does not auto-reconnect. Recommended strategy in your GUI:

1. Listen to the `Disconnected` signal.
2. After a brief delay (1-5s exponential backoff), attempt `ConnectAsync` again.
3. Re-send `hello` and re-query `job.list` to recover state.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Server won't start | No workspace found | Run `srengine workspace init` or set `SRENGINE_WORKSPACE` |
| `ConnectionRefused` | Server not running | Verify with `srengine serve start`; check `--port` matches |
| `hello` response never arrives | Wrong host/port or firewall | Test with `nc 127.0.0.1 8765` and send the hello JSON manually |
| "Subprocess did not connect control socket" | Subprocess crashed before handshake | Check workspace jobs for the job manifest; check exit code |
| Job shows `running` forever | Subprocess lost or hung | Send `job.cancel` and inspect system processes |
| No events received after job starts | Subprocess may not use `resolve_reporter()`/`resolve_callbacks()` | Verify the subprocess receives `SRENGINE_GUI_SOCKET` env var |
| Events arrive on one client but not another | Client disconnected and reconnected mid-stream | Events are broadcast to all currently connected clients; missed events are not replayed |
