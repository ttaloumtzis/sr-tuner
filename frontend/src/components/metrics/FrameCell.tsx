import { useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";

export function FrameCell({ label, path, onExpand }: { label: string; path: string | null; onExpand: () => void }) {
  const [hover, setHover] = useState(false);
  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={path ? onExpand : undefined}
      onKeyDown={path ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onExpand(); } } : undefined}
      role="button"
      tabIndex={path ? 0 : -1}
      aria-label={path ? `Open ${label} frame preview` : undefined}
      style={{
        background: "var(--bg2)", border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)", overflow: "hidden",
        display: "flex", alignItems: "center", justifyContent: "center",
        position: "relative", minHeight: 60, cursor: path ? "zoom-in" : "default",
        transition: "border-color 0.15s ease", outline: "none",
      }}
    >
      {path ? (
        <img
          src={convertFileSrc(path)}
          alt={label}
          loading="lazy"
          style={{ width: "100%", height: "100%", objectFit: "contain" }}
        />
      ) : (
        <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>—</span>
      )}
      <span style={{
        position: "absolute", top: 5, left: 6, fontSize: 9, fontWeight: 600,
        color: "var(--text)", fontFamily: "var(--font-mono)", letterSpacing: "0.04em",
        background: "rgba(13,15,17,0.75)", padding: "2px 6px", borderRadius: 20,
        backdropFilter: "blur(2px)",
      }}>
        {label}
      </span>
      {path && (
        <span style={{
          position: "absolute", top: 5, right: 6, width: 18, height: 18, borderRadius: "50%",
          display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11,
          color: "var(--text)", background: "rgba(13,15,17,0.75)", opacity: hover ? 1 : 0,
          transition: "opacity 0.12s ease", pointerEvents: "none",
        }}>
          ⤢
        </span>
      )}
    </div>
  );
}