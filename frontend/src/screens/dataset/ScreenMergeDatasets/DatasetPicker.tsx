import { useState, type CSSProperties } from "react";
import { CheckCircle, AlertCircle, Search } from "lucide-react";
import { Panel } from "../../../components/ui/Panel";
import type { ScannedDataset } from "../../../lib/scanDatasets";

interface DatasetPickerProps {
  scanned: ScannedDataset[];
  selectedPaths: Set<string>;
  scanning: boolean;
  scaleOptions: number[];
  scaleFilter: number | null;
  onScaleFilterChange: (scale: number | null) => void;
  onToggle: (path: string) => void;
  onSelectVisible: (paths: string[]) => void;
  onClearVisible: (paths: string[]) => void;
}

const pillBtn: CSSProperties = {
  background: "var(--bg3)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius-sm)",
  color: "var(--muted)",
  fontSize: 10,
  padding: "3px 10px",
  cursor: "pointer",
  transition: "var(--transition-fast)",
};

export function DatasetPicker({
  scanned,
  selectedPaths,
  scanning,
  scaleOptions,
  scaleFilter,
  onScaleFilterChange,
  onToggle,
  onSelectVisible,
  onClearVisible,
}: DatasetPickerProps) {
  const [query, setQuery] = useState("");

  const visible = scanned.filter((d) => {
    const matchesQuery = d.name.toLowerCase().includes(query.toLowerCase());
    const matchesScale = scaleFilter === null || d.scale === scaleFilter;
    return matchesQuery && matchesScale;
  });

  const groups = new Map<number, ScannedDataset[]>();
  for (const d of visible) {
    const arr = groups.get(d.scale) ?? [];
    arr.push(d);
    groups.set(d.scale, arr);
  }
  const scaleKeys = [...groups.keys()].sort((a, b) => a - b);
  const selectedVisible = visible.filter((d) => selectedPaths.has(d.path)).length;

  return (
    <Panel title="Source Datasets" subtitle={`${scanned.length} detected`}>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div
            style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              gap: 6,
              background: "var(--bg3)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-sm)",
              padding: "4px 8px",
              minWidth: 0,
            }}
          >
            <Search size={12} color="var(--muted)" style={{ flexShrink: 0 }} />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search datasets…"
              aria-label="Search datasets"
              style={{
                flex: 1,
                minWidth: 0,
                background: "transparent",
                border: "none",
                outline: "none",
                color: "var(--text)",
                fontSize: 11,
                fontFamily: "var(--font-sans)",
              }}
            />
          </div>
          <span
            title={`${selectedVisible} of ${visible.length} shown are selected`}
            style={{
              fontSize: 10,
              color: "var(--muted)",
              fontFamily: "var(--font-mono)",
              whiteSpace: "nowrap",
              flexShrink: 0,
            }}
          >
            {selectedVisible}/{visible.length}
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 4, flexWrap: "wrap" }}>
          {[null, ...scaleOptions].map((scale) => {
            const active = scaleFilter === scale;
            return (
              <button
                key={scale ?? "all"}
                onClick={() => onScaleFilterChange(scale)}
                style={{
                  background: active ? "var(--green)" : "var(--bg3)",
                  border: `1px solid ${active ? "var(--green)" : "var(--border)"}`,
                  color: active ? "#0d0f11" : "var(--muted)",
                  fontSize: 10,
                  fontWeight: active ? 600 : 400,
                  padding: "2px 10px",
                  borderRadius: "var(--radius-sm)",
                  cursor: "pointer",
                  transition: "var(--transition-fast)",
                }}
              >
                {scale === null ? "All" : `×${scale}`}
              </button>
            );
          })}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <button
            onClick={() => onSelectVisible(visible.map((d) => d.path))}
            disabled={visible.length === 0}
            style={pillBtn}
          >
            Select all
          </button>
          <button
            onClick={() => onClearVisible(visible.map((d) => d.path))}
            disabled={visible.length === 0}
            style={pillBtn}
          >
            Clear
          </button>
          {scanning && (
            <span
              style={{
                marginLeft: "auto",
                fontSize: 10,
                color: "var(--dim)",
                fontStyle: "italic",
              }}
            >
              Scanning…
            </span>
          )}
        </div>
      </div>

      {scanned.length === 0 && !scanning && (
        <div
          style={{
            padding: "16px 4px",
            fontSize: 11,
            color: "var(--dim)",
            textAlign: "center",
          }}
        >
          No datasets found in the project. Use the Create Dataset tab to add one.
        </div>
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 4 }}>
        {scaleKeys.map((scale) => (
          <div
            key={scale}
            role="group"
            aria-label={`Scale ×${scale} datasets`}
            style={{ display: "flex", flexDirection: "column", gap: 2 }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                fontSize: 9,
                color: "var(--dim)",
                textTransform: "uppercase",
                letterSpacing: 1,
                fontWeight: 600,
                padding: "2px 4px",
              }}
            >
              <span>×{scale}</span>
              <span style={{ flex: 1, height: 1, background: "var(--border)" }} />
              <span>
                {groups.get(scale)!.length} dataset
                {groups.get(scale)!.length !== 1 ? "s" : ""}
              </span>
            </div>
            {groups.get(scale)!.map((ds) => {
              const checked = selectedPaths.has(ds.path);
              return (
                <label
                  key={ds.path}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "5px 6px",
                    borderRadius: "var(--radius-sm)",
                    cursor: "pointer",
                    fontSize: 11,
                    color: "var(--text)",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => onToggle(ds.path)}
                    style={{ accentColor: "var(--green)" }}
                  />
                  <span
                    title={ds.name}
                    style={{
                      flex: 1,
                      minWidth: 0,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {ds.name}
                  </span>
                  <span
                    style={{
                      fontSize: 10,
                      color: "var(--dim)",
                      fontFamily: "var(--font-mono)",
                      flexShrink: 0,
                    }}
                  >
                    {ds.pairCount.toLocaleString()} pairs
                  </span>
                  {ds.hasManifest ? (
                    <CheckCircle
                      size={12}
                      color="var(--green)"
                      style={{ flexShrink: 0 }}
                      aria-label="Manifest OK"
                    />
                  ) : (
                    <AlertCircle
                      size={12}
                      color="var(--amber)"
                      style={{ flexShrink: 0 }}
                      aria-label="Missing manifest"
                    />
                  )}
                </label>
              );
            })}
          </div>
        ))}
      </div>
    </Panel>
  );
}
