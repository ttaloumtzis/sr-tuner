import { getCurrentWindow } from "@tauri-apps/api/window";

function TrafficLights() {
  const win = getCurrentWindow();
  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center", flexShrink: 0 }}>
      <button
        onClick={() => win.close()}
        style={{
          width: 12, height: 12, borderRadius: "50%",
          background: "#e05c5c", border: "none", cursor: "pointer",
          padding: 0, flexShrink: 0,
        }}
        title="Close"
        aria-label="Close window"
      />
      <button
        onClick={() => win.minimize()}
        style={{
          width: 12, height: 12, borderRadius: "50%",
          background: "#f5a623", border: "none", cursor: "pointer",
          padding: 0, flexShrink: 0,
        }}
        title="Minimize"
        aria-label="Minimize window"
      />
      <button
        onClick={() => win.toggleMaximize()}
        style={{
          width: 12, height: 12, borderRadius: "50%",
          background: "#4dba7f", border: "none", cursor: "pointer",
          padding: 0, flexShrink: 0,
        }}
        title="Maximize"
        aria-label="Maximize window"
      />
    </div>
  );
}

export function LandingTitleBar() {
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
          minWidth: 0,
          height: "100%",
        }}
      >
        <span
          style={{
            color: "var(--green)",
            fontSize: 12,
            fontFamily: "var(--font-mono)",
            fontWeight: 700,
            letterSpacing: "0.04em",
          }}
        >
          SR TUNER
        </span>
      </div>
      <div style={{ width: 12 * 3 + 6 * 2, flexShrink: 0 }} />
    </div>
  );
}