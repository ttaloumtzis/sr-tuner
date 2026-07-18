import { useState, useEffect, useCallback } from "react";
import { readDir } from "@tauri-apps/plugin-fs";
import { homeDir, join } from "@tauri-apps/api/path";
import { Btn } from "./Btn";

interface QuickLink {
  label: string;
  path: string;
}

interface FilePickerProps {
  isOpen: boolean;
  title?: string;
  defaultPath?: string;
  onSelect: (path: string) => void;
  onClose: () => void;
}

async function resolveHome(): Promise<string> {
  try {
    return await homeDir();
  } catch {
    return "/home";
  }
}

async function listDirectory(path: string): Promise<string[]> {
  try {
    const entries = await readDir(path);
    const items = entries
      .map((e) => (e.isDirectory ? e.name + "/" : e.name))
      .sort();
    return items;
  } catch {
    return [];
  }
}

export function FilePicker({
  isOpen,
  title,
  defaultPath,
  onSelect,
  onClose,
}: FilePickerProps) {
  const [path, setPath] = useState<string>(defaultPath ?? "");
  const [entries, setEntries] = useState<string[]>([]);
  const [quickLinks, setQuickLinks] = useState<QuickLink[]>([]);

  useEffect(() => {
    resolveHome().then(async (h) => {
      const initial = defaultPath && defaultPath.length > 0 ? defaultPath : h;
      setPath(initial);

      const projects = await join(h, "Projects").catch(() => h + "/Projects");
      const datasets = await join(h, "datasets").catch(() => h + "/datasets");
      const videos   = await join(h, "Videos").catch(() => h + "/Videos");
      const downloads= await join(h, "Downloads").catch(() => h + "/Downloads");

      setQuickLinks([
        { label: "Home",       path: h },
        { label: "Projects",   path: projects },
        { label: "Datasets",   path: datasets },
        { label: "Videos",     path: videos },
        { label: "Downloads",  path: downloads },
        { label: "/ (root)",   path: "/" },
      ]);
    });
  }, [defaultPath]);

  const loadDir = useCallback(async (dir: string) => {
    setPath(dir);
    const list = await listDirectory(dir);
    setEntries(list);
  }, []);

  useEffect(() => {
    if (isOpen && path) loadDir(path);
  }, [isOpen, path, loadDir]);

  if (!isOpen) return null;

  const handleItemDoubleClick = async (item: string) => {
    if (!item.endsWith("/")) return;
    const next = await join(path, item.slice(0, -1)).catch(() => path + "/" + item.slice(0, -1));
    loadDir(next);
  };

  const handleItemClick = (item: string) => {
    if (!item.endsWith("/")) {
      join(path, item)
        .then((p) => setPath(p))
        .catch(() => setPath(path + "/" + item));
    }
  };

  const navigateUp = async () => {
    const parts = path.replace(/\/$/, "").split("/");
    if (parts.length <= 1) return;
    parts.pop();
    const parent = parts.join("/") || "/";
    loadDir(parent);
  };

  return (
    <div
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.65)",
        zIndex: 9999,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        style={{
          background: "var(--bg1)",
          border: "1px solid var(--border2)",
          borderRadius: "var(--radius-lg)",
          width: 540,
          maxHeight: "80vh",
          display: "flex",
          flexDirection: "column",
          boxShadow: "0 24px 64px rgba(0,0,0,0.6)",
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: "10px 16px",
            borderBottom: "1px solid var(--border)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            background: "var(--bg2)",
            flexShrink: 0,
          }}
        >
          <span style={{ fontSize: 12, color: "var(--text)", fontWeight: 500 }}>
            {title ?? "Select Path"}
          </span>
          <button
            onClick={onClose}
            style={{
              background: "transparent",
              border: "none",
              color: "var(--muted)",
              cursor: "pointer",
              fontSize: 16,
              lineHeight: 1,
              padding: "0 4px",
            }}
          >
            ×
          </button>
        </div>

        {/* Body: sidebar + browser */}
        <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
          {/* Quick Access sidebar */}
          <div
            style={{
              width: 140,
              borderRight: "1px solid var(--border)",
              padding: "8px 0",
              background: "var(--bg2)",
              flexShrink: 0,
              overflowY: "auto",
            }}
          >
            <div
              style={{
                padding: "4px 12px",
                fontSize: 9,
                color: "var(--dim)",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                marginBottom: 4,
              }}
            >
              Quick Access
            </div>
            {quickLinks.map((q) => {
              const active = path === q.path || path.startsWith(q.path + "/");
              return (
                <div
                  key={q.path}
                  onClick={() => loadDir(q.path)}
                  style={{
                    padding: "6px 14px",
                    fontSize: 11,
                    cursor: "pointer",
                    color: active ? "var(--green)" : "var(--muted)",
                    background: active ? "var(--green-dim)" : "transparent",
                    transition: "background 0.1s",
                  }}
                  onMouseEnter={(e) => {
                    if (!active)
                      (e.currentTarget as HTMLDivElement).style.background = "var(--bg3)";
                  }}
                  onMouseLeave={(e) => {
                    if (!active)
                      (e.currentTarget as HTMLDivElement).style.background = "transparent";
                  }}
                >
                  {q.label}
                </div>
              );
            })}
          </div>

          {/* Directory browser */}
          <div style={{ flex: 1, padding: 8, overflowY: "auto", display: "flex", flexDirection: "column", gap: 4 }}>
            {/* Current path + up button */}
            <div
              style={{
                fontSize: 10,
                color: "var(--dim)",
                fontFamily: "var(--font-mono)",
                padding: "4px 6px",
                background: "var(--bg2)",
                borderRadius: "var(--radius-sm)",
                display: "flex",
                alignItems: "center",
                gap: 6,
              }}
            >
              <span
                onClick={navigateUp}
                title="Go up"
                style={{ cursor: "pointer", color: "var(--muted)", flexShrink: 0 }}
              >
                ↑
              </span>
              <span
                style={{
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  flex: 1,
                }}
              >
                {path}
              </span>
            </div>

            {/* Entries */}
            {entries.length === 0 ? (
              <div style={{ fontSize: 11, color: "var(--dim)", padding: "6px 8px" }}>
                (empty)
              </div>
            ) : (
              entries.map((item) => {
                const isDir = item.endsWith("/");
                return (
                  <div
                    key={item}
                    onClick={() => handleItemClick(item)}
                    onDoubleClick={() => handleItemDoubleClick(item)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      padding: "5px 8px",
                      borderRadius: "var(--radius-sm)",
                      cursor: isDir ? "pointer" : "default",
                      fontSize: 11,
                      color: "var(--text)",
                      transition: "background 0.1s",
                    }}
                    onMouseEnter={(e) => {
                      (e.currentTarget as HTMLDivElement).style.background = "var(--bg3)";
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLDivElement).style.background = "transparent";
                    }}
                  >
                    <span
                      style={{
                        color: isDir ? "var(--blue)" : "var(--muted)",
                        fontFamily: "var(--font-mono)",
                        fontSize: 12,
                        flexShrink: 0,
                      }}
                    >
                      {isDir ? "▸" : " "}
                    </span>
                    <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {item}
                    </span>
                    {isDir && (
                      <span style={{ fontSize: 9, color: "var(--dim)", flexShrink: 0 }}>
                        dbl-click
                      </span>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Footer: path bar + actions */}
        <div
          style={{
            padding: "10px 14px",
            borderTop: "1px solid var(--border)",
            background: "var(--bg2)",
            display: "flex",
            gap: 8,
            alignItems: "center",
            flexShrink: 0,
          }}
        >
          <div
            style={{
              flex: 1,
              background: "var(--bg3)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-sm)",
              padding: "5px 9px",
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              color: "var(--text)",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {path}
          </div>
          <Btn
            variant="solid"
            color="var(--green)"
            onClick={() => {
              onSelect(path);
              onClose();
            }}
          >
            Select
          </Btn>
          <Btn onClick={onClose}>Cancel</Btn>
        </div>
      </div>
    </div>
  );
}
