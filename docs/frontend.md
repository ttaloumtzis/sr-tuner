# Frontend / Desktop GUI

## Overview

sr-engine includes a native desktop GUI built with **React 18 + TypeScript** (frontend) and **Tauri 2** (Rust desktop shell). It communicates with the Python backend via the FastAPI REST API on `localhost:8765`.

The GUI provides a tab-based interface for all major operations: project management, dataset building, model configuration, training with live metrics, inference, and checkpoint management.

---

## Technology Stack

| Layer | Technology | Version |
|-------|------------|---------|
| UI Framework | React | 18 |
| Language | TypeScript | 5 |
| Build Tool | Vite | 5 |
| State Management | Zustand | 5 |
| Desktop Shell | Tauri | 2 (Rust) |
| Testing | Vitest + Testing Library | — |
| Styling | CSS variables (design tokens) + inline styles | — |

---

## Directory Layout

```
frontend/
├── index.html
├── package.json
├── vite.config.ts                  # Vite config with API proxy to :8765
├── vitest.config.ts
├── tsconfig.json
├── tsconfig.node.json
└── src/
    ├── main.tsx                    # React entry point
    ├── App.tsx                     # Root component with tab routing
    ├── index.css
    ├── lib/
    │   ├── api.ts                  # API client (fetch-based)
    │   ├── api-types.ts            # TypeScript interfaces for API
    │   ├── tokens.css              # CSS design tokens
    │   ├── eta.ts                  # ETA calculation utilities
    │   ├── metrics.ts              # Metrics data processing
    │   ├── namingPattern.ts        # Naming convention helpers
    │   ├── scanDatasets.ts         # Dataset scanning utilities
    │   ├── srproj.ts               # SRProjFile schema types
    │   ├── SRProjManager.ts        # Project file management
    │   ├── useSaveTrigger.ts       # Save trigger hook
    │   ├── useUiStatePersist.ts    # UI state persistence hook
    │   └── vramEstimate.ts         # VRAM estimation utilities
    ├── hooks/
    │   ├── useSSEConnection.ts     # Server health check + retry
    │   ├── useTrainingSSE.ts       # Training SSE event stream
    │   ├── useDatasetSSE.ts        # Dataset SSE event stream
    │   └── __tests__/
    ├── store/
    │   ├── projectStore.ts         # Project state
    │   ├── uiStore.ts              # UI state, tabs, toasts
    │   ├── datasetStore.ts         # Dataset operations state
    │   ├── modelStore.ts           # Model architecture + instance state
    │   ├── trainingStore.ts        # Training run state
    │   ├── inferenceStore.ts       # Inference job state
    │   ├── runConfigStore.ts       # Training hyperparameters
    │   ├── checkpointStore.ts      # Checkpoint management
    │   └── __tests__/
    ├── screens/
    │   ├── ProjectScreen.tsx       # Landing page
    │   ├── dataset/
    │   │   ├── ScreenDatasetSetup.tsx
    │   │   ├── ScreenDatasetCreate.tsx
    │   │   ├── ScreenBrowseDatasets.tsx
    │   │   ├── ScreenMergeDatasets.tsx
    │   │   └── DegradationPanel.tsx
    │   ├── model/
    │   │   └── ScreenModelConfig.tsx
    │   ├── training/
    │   │   └── ScreenTrainingSetup.tsx
    │   ├── metrics/
    │   │   └── ScreenMetrics.tsx
    │   ├── checkpoints/
    │   │   └── ScreenCheckpoints.tsx
    │   └── inference/
    │       └── ScreenInference.tsx
    ├── components/
    │   ├── shell/
    │   │   ├── TabBar.tsx
    │   │   ├── StatusBar.tsx
    │   │   ├── ToastProvider.tsx
    │   │   ├── ErrorRouter.tsx
    │   │   ├── ErrorDialog.tsx
    │   │   ├── ConnectionErrorDialog.tsx
    │   │   ├── TitleBar.tsx
    │   │   ├── LandingTitleBar.tsx
    │   │   └── SettingsModal.tsx
    │   ├── ui/
    │   │   ├── index.ts
    │   │   ├── Btn.tsx
    │   │   ├── Dropdown.tsx
    │   │   ├── Field.tsx
    │   │   ├── InfoRow.tsx
    │   │   ├── Panel.tsx
    │   │   ├── PathInput.tsx
    │   │   ├── PBar.tsx
    │   │   ├── Tag.tsx
    │   │   └── Toggle.tsx
    │   ├── dataset/
    │   │   └── JobOverlay.tsx
    │   └── metrics/
    │       └── RunComparisonTable.tsx
    └── __mocks__/
```

---

## Screens

### 1. ProjectScreen (Landing)

**File:** `screens/ProjectScreen.tsx` (630 lines)

The landing page shown when no project is open. Provides:

- **New Project** — Create a new workspace directory with name, path, and backend selection
- **Recent Projects** — List of recently opened projects with quick-open
- **Open Project** — File browser to select an existing workspace directory

On project open/create, the tab bar appears with all 6 tabs.

**Key states:**
- `projectStore.project` — current project path
- `projectStore.isOpen` — whether a project is loaded

---

### 2. Dataset Screen

**Screen:** `ScreenDatasetSetup.tsx` — Tab router with 3 sub-tabs

#### 2a. ScreenDatasetCreate

Build a dataset from a video file.

**Features:**
- Video file picker (PathInput)
- Output directory selector
- Degradation configuration panel (DegradationPanel)
- Degradation presets or manual toggle for each stage
- Config overrides for blur, noise, JPEG, JPEG2000, color jitter
- Resize method selection
- Build button → triggers `POST /api/datasets/build`
- Job overlay with SSE progress (JobOverlay)

#### 2b. ScreenBrowseDatasets

List and inspect existing datasets.

**Features:**
- Fetches `GET /api/datasets` with optional scale filter
- Shows dataset cards with name, scale, pair count
- Validate button → `POST /api/datasets/validate`
- Health check button → `POST /api/datasets/health`
- Prune black frames from health results

#### 2c. ScreenMergeDatasets

Merge multiple datasets by scale.

**Features:**
- Source datasets directory selection
- Scale factor filter
- Output name and path
- Merge button → `POST /api/datasets/merge`

---

### 3. Model Screen

**Screen:** `ScreenModelConfig.tsx` (889 lines)

Two sub-modes: Create and View.

#### Create Mode

**Features:**
- Architecture selector dropdown (Real-ESRGAN / SwinIR)
- Architecture-specific parameter sliders:
  - Real-ESRGAN: `num_feat`, `num_block`, `num_grow_ch`, `scale`
  - SwinIR: `embed_dim`, `depths`, `num_heads`, `window_size`, `scale`
- Real-time YAML config preview
- Instance name input
- Create button → `POST /api/models/instances`

#### View Mode

**Features:**
- Instance list with architecture, scale, latest version
- Instance detail panel with full config
- Version history table
- Export button (ONNX, TorchScript, SafeTensors)
- Delete instance

---

### 4. Training Screen

**Screen:** `ScreenTrainingSetup.tsx` (435 lines)

**Features:**
- Model instance selector (resolves to architecture)
- Dataset selector
- Hyperparameter inputs:
  - Batch size, learning rate, max epochs
  - Patch size, seed
  - Mixed precision toggle (fp16)
  - Validation split, save frequency
  - Perceptual loss weight, warmup steps
  - Adam betas, weight decay
- VRAM estimate display
- Start training button → `POST /api/train/start`
- Resume from version selector

---

### 5. Metrics Screen

**Screen:** `ScreenMetrics.tsx` (614 lines)

**Features:**
- Live training metrics dashboard
- SSE connection via `useTrainingSSE`
- Real-time loss chart (loss_total, loss_pixel, loss_perceptual vs epoch/batch)
- GPU utilization gauge
- VRAM usage bar
- ETA display
- Validation images (thumbnails from SSE validate events)
- Training history (past runs from `checkpointStore`)

---

### 6. Checkpoints Screen

**Screen:** `ScreenCheckpoints.tsx` (574 lines)

**Features:**
- List all checkpoints across runs
- Sort by epoch, date, or metric
- Filter by run or instance
- Export individual checkpoints
- Delete old checkpoints
- Version promotion

---

### 7. Inference Screen

**Screen:** `ScreenInference.tsx` (875 lines)

**Features:**
- Input image drag-and-drop zone
- Optional ground truth image (for quality comparison)
- Model selector (instance or direct checkpoint)
- Tiling configuration (tile size, overlap)
- Run inference button → `POST /api/infer/start`
- Before/after comparison with draggable splitter
- Quality metrics (PSNR, SSIM between output and GT)
- Image info display (dimensions, file size, format)

---

## State Management (Zustand)

9 stores manage all application state:

### projectStore

```typescript
interface ProjectState {
  project: string | null
  isOpen: boolean
  openProject: (path: string) => Promise<void>
  closeProject: () => void
}
```

### uiStore

```typescript
interface UiState {
  activeTab: string
  serverConnected: boolean
  toasts: Toast[]
  expandedPanels: Record<string, boolean>
  setActiveTab: (tab: string) => void
  setServerConnected: (connected: boolean) => void
  addToast: (toast: Toast) => void
  removeToast: (id: string) => void
}
```

### datasetStore

Manages: active sub-tab, build parameters, validation results, health check results, merge parameters, active job IDs.

### modelStore

Manages: architecture list, instances list, selected instance, create/view mode, create form state.

### trainingStore

Manages: active run status, SSE event stream, job history, hyperparameter presets.

### inferenceStore

Manages: input/output paths, model selection, job state, inference results, comparison state.

### runConfigStore

Manages: training hyperparameters as a single config object (batch_size, learning_rate, max_epochs, etc.).

### checkpointStore

Manages: checkpoint lists grouped by run, sort/filter state, export queue.

---

## API Client

**File:** `lib/api.ts` (77 lines)

A minimal fetch-based API client with typed methods:

```typescript
// Workspace
getWorkspace()        → GET /api/workspace
initWorkspace(path)   → POST /api/workspace/init
checkWorkspace()      → GET /api/workspace/check

// Models
listModels()          → GET /api/models
listInstances()       → GET /api/models/instances
getInstance(name)     → GET /api/models/instances/{name}
createInstance(data)  → POST /api/models/instances
exportModel(name, fmt)→ POST /api/models/instances/{name}/export
getVersions(name)     → GET /api/models/instances/{name}/versions
deleteInstance(name)  → DELETE /api/models/instances/{name}

// Training
startTraining(params) → POST /api/train/start
validateDataset(params)→ POST /api/train/validate-dataset

// Inference
startInference(params)→ POST /api/infer/start

// Datasets
listDatasets(scale?)  → GET /api/datasets
buildDataset(params)  → POST /api/datasets/build
validateDataset(params)→ POST /api/datasets/validate
validateDatasetAsync(params)→ POST /api/datasets/validate-async
healthCheck(params)   → POST /api/datasets/health
mergeDatasets(params) → POST /api/datasets/merge
pruneDataset(params)  → POST /api/datasets/prune

// Jobs
listJobs()            → GET /api/jobs
getJob(id)            → GET /api/jobs/{id}
cancelJob(id)         → POST /api/jobs/{id}/cancel

// Environment
getEnv()              → GET /api/env
```

**Type definitions** (`lib/api-types.ts`, 200 lines) mirror the Pydantic schemas from the API server.

---

## SSE Hooks

### useSSEConnection

**File:** `hooks/useSSEConnection.ts` (51 lines)

Manages the server health check connection with exponential backoff retry:

```typescript
function useSSEConnection()
```

- Polls `GET /api/health` every 5 seconds
- On failure: retries with exponential backoff (1s, 2s, 4s, 8s, max 30s)
- Updates `uiStore.serverConnected`
- Shows `ConnectionErrorDialog` when disconnected

### useTrainingSSE

**File:** `hooks/useTrainingSSE.ts`

Subscribes to `GET /api/events?job_id=<id>` for training progress:

```typescript
function useTrainingSSE(jobId: string | null)
```

- Handles all event types: `phase`, `step`, `validate`, `done`, `error`, `hardware`
- Updates `trainingStore` with live metrics
- Renders validation frame thumbnails from `validate` events

### useDatasetSSE

**File:** `hooks/useDatasetSSE.ts`

Subscribes to `GET /api/events?job_id=<id>` for dataset operations:

```typescript
function useDatasetSSE(jobId: string | null)
```

- Handles: `progress_start`, `progress_update`, `progress_end`, `done`, `error`
- Updates `datasetStore` with progress and completion state

---

## UI Component Library

### Shell Components

| Component | File | Purpose |
|-----------|------|---------|
| `TabBar` | `components/shell/TabBar.tsx` | Tab navigation between screens |
| `StatusBar` | `components/shell/StatusBar.tsx` | Bottom status bar with server status |
| `ToastProvider` | `components/shell/ToastProvider.tsx` | Toast notification system |
| `ErrorRouter` | `components/shell/ErrorRouter.tsx` | Route-level error boundary |
| `ErrorDialog` | `components/shell/ErrorDialog.tsx` | Modal error dialog |
| `ConnectionErrorDialog` | `components/shell/ConnectionErrorDialog.tsx` | Server connection lost dialog |
| `TitleBar` | `components/shell/TitleBar.tsx` | Window title bar (inside app) |
| `LandingTitleBar` | `components/shell/LandingTitleBar.tsx` | Landing page title bar |
| `SettingsModal` | `components/shell/SettingsModal.tsx` | Application settings modal |

### UI Primitives

| Component | File | Purpose |
|-----------|------|---------|
| `Btn` | `components/ui/Btn.tsx` | Button with variants (primary, secondary, danger) |
| `Dropdown` | `components/ui/Dropdown.tsx` | Select dropdown |
| `Field` | `components/ui/Field.tsx` | Form field with label and validation |
| `InfoRow` | `components/ui/InfoRow.tsx` | Label-value information row |
| `Panel` | `components/ui/Panel.tsx` | Collapsible content panel |
| `PathInput` | `components/ui/PathInput.tsx` | File path input with browse button |
| `PBar` | `components/ui/PBar.tsx` | Progress bar |
| `Tag` | `components/ui/Tag.tsx` | Tag/chip for status display |
| `Toggle` | `components/ui/Toggle.tsx` | Toggle switch |

### Domain Components

| Component | File | Purpose |
|-----------|------|---------|
| `JobOverlay` | `components/dataset/JobOverlay.tsx` | Overlay showing dataset job progress |
| `RunComparisonTable` | `components/metrics/RunComparisonTable.tsx` | Side-by-side training run comparison |

---

## Development Setup

### Prerequisites

- Node.js 18+
- Python server running on `localhost:8765`

### Install Dependencies

```bash
cd frontend
npm install
```

### Development Server

```bash
npm run dev
```

Starts Vite dev server on `http://localhost:1420` with API proxy to `http://localhost:8765`. The proxy is configured in `vite.config.ts`:

```typescript
server: {
  port: 1420,
  proxy: {
    "/api": "http://localhost:8765",
  },
}
```

### Testing

```bash
# Run tests
npx vitest run

# Watch mode
npx vitest
```

### Production Build

```bash
npm run build
```

Output goes to `frontend/dist/` for Tauri bundling.

---

## Tauri Integration

The frontend is bundled as a Tauri 2 desktop application. See [Desktop Guide](desktop.md) for details on:

- Python server lifecycle management
- Tauri commands (filesystem, process)
- Building and distribution

---

## Design Tokens

**File:** `lib/tokens.css`

CSS custom properties define the visual design system:

```css
:root {
  --color-bg: #1a1a2e;
  --color-surface: #16213e;
  --color-primary: #0f3460;
  --color-accent: #e94560;
  --color-text: #eaeaea;
  --color-text-muted: #8899aa;
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 32px;
}
```

---

## See Also

- [API Reference](api-reference.md) — REST API consumed by the frontend
- [Desktop Guide](desktop.md) — Tauri shell and build process
- [Architecture Overview](architecture.md) — System-level architecture