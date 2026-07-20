import React, { useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useSaveTrigger } from "../../lib/useSaveTrigger";
import { useProjectStore } from "../../store/projectStore";
import { parentFromProjFile } from "../../lib/path";
import { SettingsModal } from "./SettingsModal";

function TrafficLights() {
  const win = getCurrentWindow();
  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center", flexShrink: 0 }}>
      <button
        onClick={() => win.close()}
        style={dot("#e05c5c")}
        title="Close"
        aria-label="Close window"
      />
      <button
        onClick={() => win.minimize()}
        style={dot("#f5a623")}
        title="Minimize"
        aria-label="Minimize window"
      />
      <button
        onClick={() => win.toggleMaximize()}
        style={dot("#4dba7f")}
        title="Maximize"
        aria-label="Maximize window"
      />
    </div>
  );
}

function dot(color: string): React.CSSProperties {
  return {
    width: 12,
    height: 12,
    borderRadius: "50%",
    background: color,
    border: "none",
    cursor: "pointer",
    padding: 0,
    flexShrink: 0,
  };
}

export function TitleBar() {
  const project = useProjectStore((s) => s.project);
  const { saving, triggerSave } = useSaveTrigger();
  const [settingsOpen, setSettingsOpen] = useState(false);

  const name = project?.name ?? "";
  const filePath = project?.filePath ?? "";
  const parentDir = filePath ? parentFromProjFile(filePath) : "";

  return (
    <div
      style={{
        height: "var(--titlebar-h)",
        display: "flex",
        alignItems: "center",
        background: "var(--bg1)",
        borderBottom: "1px solid var(--border)",
        padding: "0 12px",
        userSelect: "none",
        flexShrink: 0,
        position: "relative",
        zIndex: 10001,
      }}
    >
      <TrafficLights />

      <div
        data-tauri-drag-region
        style={{
          flex: 1,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          gap: 6,
          minWidth: 0,
          height: "100%",
        }}
      >
        <span
          style={{
            color: "var(--text)",
            fontSize: 12,
            fontFamily: "var(--font-mono)",
            fontWeight: 600,
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            maxWidth: 240,
          }}
        >
          {name}
        </span>
        {parentDir && (
          <span
            style={{
              color: "var(--muted)",
              fontSize: 11,
              fontFamily: "var(--font-mono)",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
              maxWidth: 320,
            }}
          >
            {parentDir}
          </span>
        )}
      </div>

      <button
        onClick={() => setSettingsOpen(true)}
        style={{
          background: "none",
          border: "none",
          color: "var(--muted)",
          fontSize: 14,
          cursor: "pointer",
          padding: "2px 6px",
          marginRight: 4,
          flexShrink: 0,
          lineHeight: 1,
        }}
        title="Settings"
        aria-label="Open settings"
      >
        ⚙
      </button>
      <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <button
        onClick={triggerSave}
        style={{
          background: saving ? "var(--green-dim)" : "var(--bg3)",
          border: `1px solid ${saving ? "var(--green)" : "var(--border)"}`,
          color: saving ? "var(--green)" : "var(--muted)",
          fontSize: 11,
          fontFamily: "var(--font-sans)",
          padding: "3px 10px",
          borderRadius: "var(--radius-sm)",
          cursor: "pointer",
          transition: "var(--transition-fast)",
          flexShrink: 0,
          fontWeight: saving ? 600 : 400,
        }}
        title="Save project (Ctrl+S)"
      >
        {saving ? "Saved" : "Save"}
      </button>
    </div>
  );
}
