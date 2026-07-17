import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Btn } from "../components/ui/Btn";
import { PathInput } from "../components/ui/PathInput";
import { useProjectStore } from "../store/projectStore";
import { useToast } from "../components/shell/ToastProvider";
import { SRPROJ_SCHEMA_VERSION, type SRProjFile } from "../lib/srproj";

// ── Recent projects persistence ────────────────────────────────────────────

const RECENT_KEY = "sr-tuner:recent-projects";
const MAX_RECENT = 8;

export interface RecentEntry {
  name: string;
  filePath: string;
  lastOpened: string;
}

function loadRecent(): RecentEntry[] {
  try {
    const raw = localStorage.getItem(RECENT_KEY);
    return raw ? (JSON.parse(raw) as RecentEntry[]) : [];
  } catch {
    return [];
  }
}

function saveRecent(entries: RecentEntry[]): void {
  localStorage.setItem(RECENT_KEY, JSON.stringify(entries.slice(0, MAX_RECENT)));
}

export function addToRecent(entry: RecentEntry): RecentEntry[] {
  const existing = loadRecent().filter((e) => e.filePath !== entry.filePath);
  const next = [entry, ...existing].slice(0, MAX_RECENT);
  saveRecent(next);
  return next;
}

function removeRecent(filePath: string): RecentEntry[] {
  const next = loadRecent().filter((e) => e.filePath !== filePath);
  saveRecent(next);
  return next;
}

// ── Directory tree preview ─────────────────────────────────────────────────

function DirPreview({ parentDir, name }: { parentDir: string; name: string }) {
  const stem = name.trim();
  if (!parentDir || !stem) return null;
  const root = parentDir.replace(/\/$/, "") + "/" + stem;
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
      <div>&nbsp;&nbsp;├─ experiments/</div>
      <div>&nbsp;&nbsp;├─ checkpoints/</div>
      <div>&nbsp;&nbsp;└─ logs/</div>
    </div>
  );
}

// ── New project form (§8.5, §8.6, §8.11) ──────────────────────────────────

function NewProjectForm({ onDone }: { onDone: () => void }) {
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

    const projectRoot = parentDir.replace(/\/$/, "") + "/" + stem;
    const projFile = projectRoot + "/" + stem + ".srproj";

    try {
      // §8.11: Do not overwrite an existing project
      const exists = await invoke<boolean>("path_exists", { path: projFile });
      if (exists) {
        setError(`A project named "${stem}" already exists in that directory.`);
        setCreating(false);
        return;
      }

      // §8.6: Create subdirectories
      await invoke("create_dir_all", { path: projectRoot + "/experiments" });
      await invoke("create_dir_all", { path: projectRoot + "/checkpoints" });
      await invoke("create_dir_all", { path: projectRoot + "/logs" });

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
          architecture: "Real-ESRGAN",
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
        content: JSON.stringify(proj, null, 2),
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

// ── Recent projects list (§8.4, §8.12) ────────────────────────────────────

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return iso;
  }
}

function RecentRow({
  entry,
  isLast,
  onOpen,
  onRemove,
}: {
  entry: RecentEntry;
  isLast: boolean;
  onOpen: (e: RecentEntry) => void;
  onRemove: (filePath: string) => void;
}) {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "9px 14px",
        borderBottom: isLast ? "none" : "1px solid var(--border)",
        background: hovered ? "var(--bg2)" : "transparent",
        cursor: "pointer",
        transition: "var(--transition-fast)",
      }}
      onClick={() => onOpen(entry)}
    >
      <div
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: "var(--green)",
          flexShrink: 0,
        }}
      />
      <span style={{ fontSize: 12, color: "var(--text)", fontWeight: 500, flexShrink: 0 }}>
        {entry.name}
      </span>
      <span
        style={{
          fontSize: 11,
          color: "var(--muted)",
          fontFamily: "var(--font-mono)",
          flex: 1,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          minWidth: 0,
        }}
      >
        {entry.filePath}
      </span>
      <span style={{ fontSize: 11, color: "var(--dim)", flexShrink: 0 }}>
        {formatDate(entry.lastOpened)}
      </span>
      {hovered && (
        <button
          onClick={(ev) => { ev.stopPropagation(); onRemove(entry.filePath); }}
          title="Remove from list"
          style={{
            background: "none",
            border: "none",
            color: "var(--muted)",
            cursor: "pointer",
            fontSize: 14,
            lineHeight: 1,
            padding: "0 2px",
            flexShrink: 0,
          }}
        >
          ×
        </button>
      )}
    </div>
  );
}

function RecentList({
  entries,
  onOpen,
  onRemove,
}: {
  entries: RecentEntry[];
  onOpen: (e: RecentEntry) => void;
  onRemove: (filePath: string) => void;
}) {
  if (entries.length === 0) {
    return (
      <div
        style={{
          background: "var(--bg1)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-md)",
          padding: "20px 16px",
          textAlign: "center",
          color: "var(--dim)",
          fontSize: 12,
        }}
      >
        No recent projects
      </div>
    );
  }

  return (
    <div
      style={{
        background: "var(--bg1)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)",
        overflow: "hidden",
      }}
    >
      {entries.map((entry, i) => (
        <RecentRow
          key={entry.filePath}
          entry={entry}
          isLast={i === entries.length - 1}
          onOpen={onOpen}
          onRemove={onRemove}
        />
      ))}
    </div>
  );
}

// ── Action card ────────────────────────────────────────────────────────────

function ActionCard({
  icon,
  title,
  subtitle,
  onClick,
  active,
}: {
  icon: string;
  title: string;
  subtitle: string;
  onClick: () => void;
  active?: boolean;
}) {
  const [hovered, setHovered] = useState(false);
  const lit = hovered || active;

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: lit ? "var(--bg2)" : "var(--bg1)",
        border: `1px solid ${lit ? "var(--green)66" : "var(--border)"}`,
        borderRadius: "var(--radius-md)",
        padding: "18px 20px",
        cursor: "pointer",
        transition: "var(--transition-normal)",
        display: "flex",
        flexDirection: "column",
        gap: 6,
      }}
    >
      <span style={{ fontSize: 20 }}>{icon}</span>
      <span
        style={{
          fontSize: 13,
          fontWeight: 600,
          color: lit ? "var(--green)" : "var(--text)",
          transition: "color 0.12s",
        }}
      >
        {title}
      </span>
      <span style={{ fontSize: 11, color: "var(--muted)" }}>{subtitle}</span>
    </div>
  );
}

// ── Open project card (§8.7) ───────────────────────────────────────────────

function OpenProjectCard({
  onRecentUpdate,
}: {
  onRecentUpdate: (entries: RecentEntry[]) => void;
}) {
  const [pickerPath, setPickerPath] = useState("");
  const [opening, setOpening] = useState(false);
  const { openProject } = useProjectStore();
  const { show: showToast } = useToast();

  const handlePathSelected = async (path: string) => {
    setPickerPath(path);
    if (!path) return;
    setOpening(true);
    try {
      await openProject(path);
      onRecentUpdate(loadRecent());
    } catch (err) {
      showToast("error", `Failed to open: ${String(err)}`);
    } finally {
      setOpening(false);
      setPickerPath("");
    }
  };

  return (
    <div
      style={{
        background: "var(--bg1)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)",
        padding: "18px 20px",
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      <span style={{ fontSize: 20 }}>📂</span>
      <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>
        {opening ? "Opening…" : "Open Project"}
      </span>
      <span style={{ fontSize: 11, color: "var(--muted)" }}>Browse for a .srproj file</span>
      <PathInput
        value={pickerPath}
        onChange={handlePathSelected}
        browseTitle="Select .srproj file"
        placeholder="Select .srproj file…"
        compact
      />
    </div>
  );
}

// ── Main ProjectScreen (§8.1) ──────────────────────────────────────────────

type View = "landing" | "new";

export function ProjectScreen() {
  const [view, setView] = useState<View>("landing");
  const [recent, setRecent] = useState<RecentEntry[]>([]);
  const { openProject } = useProjectStore();
  const { show: showToast } = useToast();

  useEffect(() => {
    setRecent(loadRecent());
  }, []);

  const handleOpenRecent = async (entry: RecentEntry) => {
    // §8.12: stale entry check
    const exists = await invoke<boolean>("path_exists", { path: entry.filePath });
    if (!exists) {
      showToast("error", `Project not found: ${entry.filePath}`);
      setRecent(removeRecent(entry.filePath));
      return;
    }
    try {
      await openProject(entry.filePath);
    } catch (err) {
      showToast("error", `Failed to open project: ${String(err)}`);
    }
  };

  const handleRemoveRecent = (filePath: string) => {
    setRecent(removeRecent(filePath));
  };

  const handleProjectCreated = () => {
    setRecent(loadRecent());
    setView("landing");
  };

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: "var(--bg0)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
        overflow: "auto",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: 720,
          display: "flex",
          flexDirection: "column",
          gap: 28,
        }}
      >
        {/* §8.2: Logo / title block */}
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 26,
              fontWeight: 700,
              color: "var(--green)",
              letterSpacing: "0.04em",
            }}
          >
            SR TUNER
          </span>
          <span style={{ fontSize: 12, color: "var(--muted)" }}>
            Super-resolution model training &amp; fine-tuning
          </span>
        </div>

        {/* §8.3: Action cards */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <ActionCard
            icon="＋"
            title="New Project"
            subtitle="Create a new SR training project"
            onClick={() => setView(view === "new" ? "landing" : "new")}
            active={view === "new"}
          />
          <OpenProjectCard onRecentUpdate={setRecent} />
        </div>

        {/* §8.5: New project form (expanded inline) */}
        {view === "new" && (
          <div
            style={{
              background: "var(--bg1)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              padding: 16,
            }}
          >
            <div
              style={{
                fontSize: 11,
                fontWeight: 600,
                color: "var(--muted)",
                textTransform: "uppercase",
                letterSpacing: "0.06em",
                marginBottom: 12,
              }}
            >
              New Project
            </div>
            <NewProjectForm onDone={handleProjectCreated} />
          </div>
        )}

        {/* §8.4: Recent projects */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <span
            style={{
              fontSize: 10,
              fontWeight: 600,
              color: "var(--muted)",
              textTransform: "uppercase",
              letterSpacing: "0.06em",
            }}
          >
            Recent Projects
          </span>
          <RecentList
            entries={recent}
            onOpen={handleOpenRecent}
            onRemove={handleRemoveRecent}
          />
        </div>
      </div>
    </div>
  );
}
