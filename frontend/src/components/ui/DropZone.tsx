import { useEffect, useRef, useState } from "react";
import { Btn } from "./Btn";
import { basename } from "../../lib/path";

interface DropZoneProps {
  label: string;
  sublabel?: string;
  path?: string | null;
  name?: string;
  accent?: string;
  dragBackground?: string;
  highlightOnDrag?: boolean;
  compact?: boolean;
  large?: boolean;
  fileFilters?: { name: string; extensions: string[] }[];
  browseLabel?: string;
  browseTitle?: string;
  selectedAsRow?: boolean;
  osDrag?: boolean;
  onSelect: (path: string) => void;
  onClear?: () => void;
}

const IMAGE_FILTERS = [
  { name: "Images", extensions: ["png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif"] },
];

/** File drop zone with a browse fallback. `compact`/`large` pick the two
 *  historical variants (inference-style small vs dataset video zone).
 *  `selectedAsRow` renders the selected file as a removable row below the zone
 *  instead of replacing the label inline. */
export function DropZone({
  label,
  sublabel,
  path,
  name,
  accent = "var(--border)",
  dragBackground,
  highlightOnDrag = true,
  compact,
  large,
  fileFilters,
  browseLabel = "Browse…",
  browseTitle,
  selectedAsRow,
  osDrag,
  onSelect,
  onClear,
}: DropZoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const onSelectRef = useRef(onSelect);
  useEffect(() => { onSelectRef.current = onSelect; });

  useEffect(() => {
    if (!osDrag) return;
    let unlisten: (() => void) | null = null;
    (async () => {
      try {
        const { getCurrentWebview } = await import("@tauri-apps/api/webview");
        unlisten = await getCurrentWebview().onDragDropEvent((event) => {
          if (event.payload.type === "over" || event.payload.type === "enter") {
            setDragOver(true);
          } else if (event.payload.type === "leave") {
            setDragOver(false);
          } else if (event.payload.type === "drop") {
            setDragOver(false);
            const paths = event.payload.paths as string[];
            if (paths.length > 0) onSelectRef.current(paths[0]);
          }
        });
      } catch {
        // running in browser dev mode — no Tauri drag-drop events
      }
    })();
    return () => { unlisten?.(); };
  }, [osDrag]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      const filePath = (file as File & { path?: string }).path || file.name;
      if (filePath) onSelect(filePath);
    }
  };

  const handleBrowse = async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        directory: false,
        multiple: false,
        title: browseTitle,
        defaultPath: path ?? undefined,
        filters: fileFilters ?? IMAGE_FILTERS,
      });
      if (selected) onSelect(selected);
    } catch {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = (fileFilters ?? IMAGE_FILTERS)
        .map((f) => f.extensions.map((e) => "." + e).join(","))
        .join(",");
      input.multiple = false;
      input.onchange = () => {
        const file = input.files?.[0];
        const filePath = (file as File & { path?: string }).path || file?.name;
        if (filePath) onSelect(filePath);
      };
      input.click();
    }
  };

  const hasFile = Boolean(path);
  const displayName = name ?? (path ? basename(path) : "");
  const dragBorder = highlightOnDrag
    ? accent === "var(--border)"
      ? "var(--green)"
      : accent
    : accent;

  return (
    <>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={["dropzone", compact && "dropzone-compact", large && "dropzone-large"].filter(Boolean).join(" ")}
        style={{
          border: `${large ? 2 : 1.5}px dashed ${dragOver ? dragBorder : accent}`,
          background: dragOver
            ? dragBackground ?? (large ? "var(--greenDim)" : "var(--bg2)")
            : large
            ? "var(--bg2)"
            : "transparent",
        }}
      >
        {selectedAsRow ? (
          <>
            <span className="dropzone-label">{label}</span>
            {sublabel && <span className="dropzone-sublabel">{sublabel}</span>}
            <Btn small variant="ghost" onClick={handleBrowse} style={{ marginTop: 4 }}>
              {browseLabel}
            </Btn>
          </>
        ) : (
          <>
            {hasFile ? (
              <div className="dropzone-name" title={path ?? undefined}>
                {displayName}
              </div>
            ) : (
              <span className="dropzone-label">{label}</span>
            )}
            <div className="dropzone-actions">
              <Btn small onClick={handleBrowse}>{browseLabel}</Btn>
              {onClear && hasFile && (
                <Btn small onClick={onClear}>Clear</Btn>
              )}
            </div>
          </>
        )}
      </div>

      {selectedAsRow && hasFile && (
        <div className="dropzone-row">
          <div className="dropzone-row-grid">
            <span className="dropzone-row-name" title={path ?? undefined}>
              {displayName}
            </span>
            <button
              onClick={onClear}
              title="Remove"
              className="dropzone-row-clear"
              onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.opacity = "1"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.opacity = "0.4"; }}
            >
              ✕
            </button>
          </div>
        </div>
      )}
    </>
  );
}