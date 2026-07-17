import { invoke } from "@tauri-apps/api/core";
import { message, confirm } from "@tauri-apps/plugin-dialog";
import { type SRProjFile, type SRProjRun, SRPROJ_SCHEMA_VERSION } from "./srproj";

let _filePath: string | null = null;
let _project: SRProjFile | null = null;

export const SRProjManager = {
  get filePath(): string | null {
    return _filePath;
  },

  get current(): SRProjFile | null {
    return _project;
  },

  async load(path: string): Promise<SRProjFile> {
    let raw: string;
    try {
      raw = await invoke<string>("read_text_file", { path });
    } catch (err) {
      throw new Error(`Cannot read file: ${err}`);
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      await message(
        "The selected .srproj file is corrupted or not valid JSON.\n\nPlease choose a different project.",
        { title: "Invalid Project File", kind: "error" }
      );
      throw new Error("Invalid JSON in .srproj file");
    }

    const proj = parsed as SRProjFile;

    // Task 3.4: version check — warn on mismatch, attempt load, show error on parse failure
    if (proj.version !== SRPROJ_SCHEMA_VERSION) {
      const proceed = await confirm(
        `This project was created with schema version "${proj.version}".\n` +
          `Current version is "${SRPROJ_SCHEMA_VERSION}".\n\n` +
          "The project will still be opened, but some fields may not load correctly.\n\n" +
          "Continue?",
        { title: "Schema Version Mismatch", kind: "warning" }
      );
      if (!proceed) throw new Error("User cancelled version-mismatch load");
    }

    _filePath = path;
    _project = proj;
    return proj;
  },

  async save(): Promise<void> {
    if (!_filePath || !_project) return;
    _project = { ..._project, last_modified_at: new Date().toISOString() };
    const json = JSON.stringify(_project, null, 2);
    await invoke("write_text_file", { path: _filePath, content: json });
  },

  setProject(path: string, proj: SRProjFile): void {
    _filePath = path;
    _project = proj;
  },

  getRun(runId: string): SRProjRun | undefined {
    return _project?.runs.find((r) => r.run_id === runId);
  },

  addRun(run: SRProjRun): void {
    if (!_project) return;
    _project = { ..._project, runs: [..._project.runs, run] };
  },

  updateRun(runId: string, patch: Partial<SRProjRun>): void {
    if (!_project) return;
    _project = {
      ..._project,
      runs: _project.runs.map((r) =>
        r.run_id === runId ? { ...r, ...patch } : r
      ),
    };
  },

  setActiveRun(runId: string | null): void {
    if (!_project) return;
    _project = {
      ..._project,
      ui_state: { ..._project.ui_state, last_active_run_id: runId },
    };
  },

  setActiveTab(tab: SRProjFile["ui_state"]["last_active_tab"]): void {
    if (!_project) return;
    _project = {
      ..._project,
      ui_state: { ..._project.ui_state, last_active_tab: tab },
    };
  },

  close(): void {
    _filePath = null;
    _project = null;
  },
};
