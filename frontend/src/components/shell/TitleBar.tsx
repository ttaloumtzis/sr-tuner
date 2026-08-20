import { useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useSaveTrigger } from "../../lib/useSaveTrigger";
import { useProjectStore } from "../../store/projectStore";
import { parentFromProjFile } from "../../lib/path";
import { SettingsModal } from "./SettingsModal";

function TrafficLights() {
  const win = getCurrentWindow();
  return (
    <div className="traffic-light">
      <button
        onClick={() => win.close()}
        style={{ background: "#e05c5c" }}
        title="Close"
        aria-label="Close window"
      />
      <button
        onClick={() => win.minimize()}
        style={{ background: "#f5a623" }}
        title="Minimize"
        aria-label="Minimize window"
      />
      <button
        onClick={() => win.toggleMaximize()}
        style={{ background: "#4dba7f" }}
        title="Maximize"
        aria-label="Maximize window"
      />
    </div>
  );
}

export function TitleBar() {
  const project = useProjectStore((s) => s.project);
  const { saving, triggerSave } = useSaveTrigger();
  const [settingsOpen, setSettingsOpen] = useState(false);

  const name = project?.name ?? "";
  const filePath = project?.filePath ?? "";
  const parentDir = filePath ? parentFromProjFile(filePath) : "";

  return (
    <div className="bar titlebar">
      <TrafficLights />

      <div
        data-tauri-drag-region
        className="titlebar-center"
      >
        <span className="titlebar-title">
          {name}
        </span>
        {parentDir && (
          <span className="titlebar-subtitle">
            {parentDir}
          </span>
        )}
      </div>

      <button
        onClick={() => setSettingsOpen(true)}
        className="titlebar-icon-btn"
        title="Settings"
        aria-label="Open settings"
      >
        ⚙
      </button>
      <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <button
        onClick={triggerSave}
        className={`titlebar-save titlebar-save-${saving ? "on" : "off"}`}
        title="Save project (Ctrl+S)"
      >
        {saving ? "Saved" : "Save"}
      </button>
    </div>
  );
}
