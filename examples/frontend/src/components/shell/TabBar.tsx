import { useState } from "react";
import { useUiStore, type TabId } from "../../store/uiStore";
import { useTrainingStore } from "../../store/trainingStore";

interface Tab {
  id: TabId;
  label: string;
  n: number;
}

const TABS: Tab[] = [
  { id: "dataset",     label: "Dataset Setup",   n: 1 },
  { id: "model",       label: "Model Config",     n: 2 },
  { id: "training",    label: "Training Setup",   n: 3 },
  { id: "metrics",     label: "Live Metrics",     n: 4 },
  { id: "checkpoints", label: "Checkpoints",      n: 5 },
  { id: "history",     label: "Run History",      n: 6 },
  { id: "inference",   label: "Inference",        n: 7 },
];

function PulsingDot() {
  return (
    <span
      style={{
        display: "inline-block",
        width: 6,
        height: 6,
        borderRadius: "50%",
        background: "var(--green)",
        marginLeft: 5,
        animation: "tabbar-pulse 1.4s ease-in-out infinite",
        flexShrink: 0,
      }}
    />
  );
}

interface TabButtonProps {
  tab: Tab;
  active: boolean;
  showDot: boolean;
  onClick: () => void;
}

function TabButton({ tab, active, showDot, onClick }: TabButtonProps) {
  const [hovered, setHovered] = useState(false);

  return (
    <button
      role="tab"
      aria-selected={active}
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 4,
        padding: "0 14px",
        background: "none",
        border: "none",
        borderBottom: active
          ? "2px solid var(--green)"
          : "2px solid transparent",
        color: active ? "var(--text)" : hovered ? "var(--text)" : "var(--muted)",
        fontSize: 12,
        fontFamily: "var(--font-sans)",
        cursor: "pointer",
        whiteSpace: "nowrap",
        transition: "color 0.12s ease, border-color 0.12s ease",
        flexShrink: 0,
        height: "100%",
      }}
    >
      <span style={{ color: active ? "var(--green)" : "var(--dim)", fontSize: 10 }}>
        {tab.n}
      </span>
      {tab.label}
      {showDot && <PulsingDot />}
    </button>
  );
}

export function TabBar() {
  const activeTab = useUiStore((s) => s.activeTab);
  const setActiveTab = useUiStore((s) => s.setActiveTab);
  const trainingStatus = useTrainingStore((s) => s.status);
  const isTraining = trainingStatus === "running" || trainingStatus === "paused";

  return (
    <>
      <style>{`
        @keyframes tabbar-pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%       { opacity: 0.4; transform: scale(0.7); }
        }
      `}</style>
      <div
        role="tablist"
        style={{
          height: "var(--tabbar-h)",
          display: "flex",
          alignItems: "stretch",
          background: "var(--bg1)",
          borderBottom: "1px solid var(--border)",
          padding: "0 8px",
          gap: 2,
          flexShrink: 0,
          overflowX: "auto",
        }}
      >
        {TABS.map((tab) => (
          <TabButton
            key={tab.id}
            tab={tab}
            active={activeTab === tab.id}
            showDot={tab.id === "metrics" && isTraining && activeTab !== tab.id}
            onClick={() => setActiveTab(tab.id)}
          />
        ))}
      </div>
    </>
  );
}
