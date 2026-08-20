import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Btn } from "../components/ui/Btn";
import { PathInput } from "../components/ui/PathInput";
import { useProjectStore } from "../store/projectStore";
import { SRPROJ_SCHEMA_VERSION, type SRProjFile } from "../lib/srproj";
import { join } from "../lib/path";

// ── Directory tree preview ─────────────────────────────────────────────────

function DirPreview({ parentDir, name }: { parentDir: string; name: string }) {
  const stem = name.trim();
  if (!parentDir || !stem) return null;
  const root = join(parentDir, stem);
  return (
    <div
      style={{
        marginTop: 10,
        background: "var(--bg3)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-sm)",
        padding: "8px 12px",
        fontFamily: "var(--font-mono)",
        fontSize: 11,
        color: "var(--muted)",
        lineHeight: 1.7,
      }}
    >
      <div style={{ color: "var(--text)" }}>{root}/</div>
      <div>&nbsp;&nbsp;├─ {stem}.srproj</div>
      <div>&nbsp;&nbsp;├─ datasets/</div>
      <div>&nbsp;&nbsp;├─ models/</div>
      <div>&nbsp;&nbsp;├─ experiments/</div>
      <div>&nbsp;&nbsp;├─ configs/</div>
      <div>&nbsp;&nbsp;├─ logs/</div>
      <div>&nbsp;&nbsp;└─ checkpoints/</div>
    </div>
  );
}

// ── New project form ───────────────────────────────────────────────────────

export function NewProjectForm({ onDone }: { onDone: () => void }) {
  const [name, setName] = useState("");
  const [parentDir, setParentDir] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const { openProject } = useProjectStore();

  const handleCreate = async () => {
    const stem = name.trim();
    if (!stem) { setError("Project name is required."); return; }
    if (!parentDir) { setError("Select a parent directory."); return; }

    setCreating(true);
    setError(null);

    const projectRoot = join(parentDir, stem);
    const projFile = join(projectRoot, stem + ".srproj");

    try {
      // Do not overwrite an existing project
      const exists = await invoke<boolean>("path_exists", { path: projFile });
      if (exists) {
        setError(`A project named "${stem}" already exists in that directory.`);
        setCreating(false);
        return;
      }

      await invoke("create_dir_all", { path: join(projectRoot, "datasets") });
      await invoke("create_dir_all", { path: join(projectRoot, "models") });
      await invoke("create_dir_all", { path: join(projectRoot, "experiments") });
      await invoke("create_dir_all", { path: join(projectRoot, "configs") });
      await invoke("create_dir_all", { path: join(projectRoot, "logs") });
      await invoke("create_dir_all", { path: join(projectRoot, "checkpoints") });

      // Write initial .srproj
      const now = new Date().toISOString();
      const proj: SRProjFile = {
        version: SRPROJ_SCHEMA_VERSION,
        name: stem,
        created_at: now,
        last_modified_at: now,
        default_dataset: {
          training_path: "",
          validation_path: "",
          validation_strategy: "auto_split",
          validation_split_ratio: 0.1,
          dataset_type: "image_folder",
        },
        default_model: {
          architecture: "rrdb_esrgan",
          upscale_factor: 4,
        },
        models: [],
        runs: [],
        ui_state: {
          last_active_run_id: null,
          last_active_tab: null,
          expanded_panels: {},
        },
        metadata: {
          app_version: "0.1.0",
          notes: null,
          tags: [],
        },
      };

      await invoke("write_text_file", {
        path: projFile,
        contents: JSON.stringify(proj, null, 2),
      });

      await openProject(projFile);
      onDone();
    } catch (err) {
      setError(String(err));
    } finally {
      setCreating(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div>
        <label
          style={{
            display: "block",
            fontSize: 10,
            fontWeight: 600,
            color: "var(--muted)",
            textTransform: "uppercase",
            letterSpacing: "0.06em",
            marginBottom: 5,
          }}
        >
          Project Name
        </label>
        <input
          value={name}
          onChange={(e) => { setName(e.target.value); setError(null); }}
          placeholder="my-sr-project"
          style={{
            width: "100%",
            background: "var(--bg3)",
            border: `1px solid ${error && !name.trim() ? "var(--red)" : "var(--border)"}`,
            borderRadius: "var(--radius-sm)",
            padding: "6px 10px",
            fontSize: 12,
            color: "var(--text)",
            fontFamily: "var(--font-mono)",
            outline: "none",
          }}
        />
      </div>

      <div>
        <label
          style={{
            display: "block",
            fontSize: 10,
            fontWeight: 600,
            color: "var(--muted)",
            textTransform: "uppercase",
            letterSpacing: "0.06em",
            marginBottom: 5,
          }}
        >
          Parent Directory
        </label>
        <PathInput
          value={parentDir}
          onChange={(p) => { setParentDir(p); setError(null); }}
          browseTitle="Select parent directory"
          placeholder="Choose where to create the project"
        />
      </div>

      <DirPreview parentDir={parentDir} name={name} />

      {error && (
        <div
          style={{
            fontSize: 11,
            color: "var(--red)",
            background: "#3d1a1a",
            border: "1px solid #e05c5c44",
            borderRadius: "var(--radius-sm)",
            padding: "6px 10px",
          }}
        >
          {error}
        </div>
      )}

      <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 4 }}>
        <Btn onClick={onDone}>Cancel</Btn>
        <Btn variant="solid" onClick={handleCreate} disabled={creating}>
          {creating ? "Creating…" : "Create Project"}
        </Btn>
      </div>
    </div>
  );
}