// §19.14 [Gap J] — Settings panel / modal housing the Export Logs action.

import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import { Btn } from "../ui/Btn";

interface Props {
  open: boolean;
  onClose: () => void;
}

function todayIso(): string {
  const now = new Date();
  return now.toISOString().slice(0, 10); // YYYY-MM-DD
}

export function SettingsModal({ open, onClose }: Props) {
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportDone, setExportDone] = useState(false);

  if (!open) return null;

  async function handleExportLogs() {
    setExportError(null);
    setExportDone(false);

    const defaultFilename = `sr-tuner-logs-${todayIso()}.zip`;
    const destPath = await save({
      defaultPath: defaultFilename,
      filters: [{ name: "ZIP archive", extensions: ["zip"] }],
    });

    if (!destPath) return; // user cancelled

    setExporting(true);
    try {
      await invoke("export_logs", { destPath });
      setExportDone(true);
    } catch (err) {
      setExportError(String(err));
    } finally {
      setExporting(false);
    }
  }

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.6)",
        zIndex: 5000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: "var(--bg1)",
          border: "1px solid var(--border)",
          borderRadius: 6,
          padding: "24px 28px",
          width: 340,
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: "var(--text)",
              fontFamily: "var(--font-mono)",
            }}
          >
            Settings
          </span>
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "none",
              color: "var(--muted)",
              fontSize: 16,
              cursor: "pointer",
              padding: "0 2px",
              lineHeight: 1,
            }}
            title="Close"
          >
            ✕
          </button>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <span style={{ fontSize: 11, color: "var(--muted)", fontWeight: 600 }}>
            Diagnostics
          </span>
          <Btn onClick={handleExportLogs} disabled={exporting}>
            {exporting ? "Exporting…" : "Export Logs"}
          </Btn>
          {exportDone && (
            <span style={{ fontSize: 10, color: "var(--green)", fontFamily: "var(--font-mono)" }}>
              Logs exported successfully
            </span>
          )}
          {exportError && (
            <span style={{ fontSize: 10, color: "var(--red)", fontFamily: "var(--font-mono)" }}>
              {exportError}
            </span>
          )}
          <span style={{ fontSize: 10, color: "var(--dim)" }}>
            Saves the last 5 log files from app data to a ZIP archive.
          </span>
        </div>
      </div>
    </div>
  );
}
