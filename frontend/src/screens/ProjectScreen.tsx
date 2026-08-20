import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { PathInput } from "../components/ui/PathInput";
import { useProjectStore } from "../store/projectStore";
import { useToast } from "../components/shell/ToastProvider";
import { ActionCard } from "./ActionCard";
import { NewProjectForm } from "./NewProjectForm";
import { RecentList } from "./RecentList";
import { loadRecent, removeRecent, type RecentEntry } from "./recentProjects";

export { addToRecent, type RecentEntry } from "./recentProjects";

// ── Open project card ───────────────────────────────────────────────────────

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
        fileFilters={[{ name: "SR Project", extensions: ["srproj"] }]}
      />
    </div>
  );
}

// ── Main ProjectScreen ──────────────────────────────────────────────────────

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
    try {
      const exists = await invoke<boolean>("path_exists", { path: entry.filePath });
      if (!exists) {
        showToast("error", `Project not found: ${entry.filePath}`);
        setRecent(removeRecent(entry.filePath));
        return;
      }
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
          maxWidth: "min(960px, 90vw)",
          display: "flex",
          flexDirection: "column",
          gap: 28,
        }}
      >
        {/* Logo / title block */}
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

        {/* Action cards */}
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

        {/* New project form (expanded inline) */}
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

        {/* Recent projects */}
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