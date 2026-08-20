import { getCurrentWindow } from "@tauri-apps/api/window";

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

export function LandingTitleBar() {
  return (
    <div className="bar titlebar">
      <TrafficLights />
      <div
        data-tauri-drag-region
        className="titlebar-center"
      >
        <span className="titlebar-brand">
          SR TUNER
        </span>
      </div>
      <div className="titlebar-spacer" />
    </div>
  );
}