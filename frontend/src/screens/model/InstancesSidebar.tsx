import { useState, useMemo } from "react";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { Tag } from "../../components/ui/Tag";
import type { ModelInstance } from "../../lib/api-types";

export function InstancesSidebar({
  instances, loading, selectedName, onSelect, onCreateClick,
}: {
  instances: ModelInstance[];
  loading: boolean;
  selectedName: string | null;
  onSelect: (name: string) => void;
  onCreateClick: () => void;
}) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(
    () => instances.filter((m) => m.name.toLowerCase().includes(query.toLowerCase())),
    [instances, query],
  );

  if (loading) {
    return (
      <Panel title="Model Instances" shrink>
        <div style={{ padding: 16, textAlign: "center", fontSize: 12, color: "var(--muted)" }}>Loading...</div>
      </Panel>
    );
  }

  if (instances.length === 0) {
    return (
      <Panel title="Model Instances" shrink>
        <div style={{ padding: 16, textAlign: "center" }}>
          <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>No model instances yet</div>
          <Btn variant="solid" small onClick={onCreateClick}>Create Model</Btn>
        </div>
      </Panel>
    );
  }

  return (
    <div className="mv-sidebar">
      <Panel title="Model Instances" shrink noPadding>
        <div style={{ padding: "8px 10px 4px", display: "flex", flexDirection: "column", gap: 6 }}>
          <div className="mv-sidebar-header">
            <span>Instances</span>
            <span className="mv-sidebar-count">{filtered.length} / {instances.length}</span>
          </div>
          <input
            className="mv-sidebar-search"
            type="text"
            placeholder="Search…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <div style={{ flex: 1, overflow: "auto", padding: "4px 0" }}>
          {filtered.map((m) => (
            <button
              key={m.name}
              className={`mv-instance-row${selectedName === m.name ? " selected" : ""}`}
              onClick={() => onSelect(m.name)}
            >
              <span className="mv-instance-name">{m.name}</span>
              {m.architecture && (
                <Tag color="cyan">{m.architecture}</Tag>
              )}
              {m.scale != null && (
                <span style={{ fontSize: 9, color: "var(--dim)", fontFamily: "var(--font-mono)", flexShrink: 0 }}>{m.scale}x</span>
              )}
            </button>
          ))}
        </div>
      </Panel>
    </div>
  );
}