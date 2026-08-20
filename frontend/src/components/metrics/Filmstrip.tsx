import { useEffect, useRef } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import type { ValidationHistoryEntry } from "../../store/trainingStore";
import { fmt } from "../../lib/format";
import { pathFor, entryPsnr } from "./frameMeta";

export function Filmstrip({
  history, selectedEpoch, isLive, onSelect, onResumeLive,
}: {
  history: ValidationHistoryEntry[];
  selectedEpoch: number | null;
  isLive: boolean;
  onSelect: (epoch: number) => void;
  onResumeLive: () => void;
}) {
  const activeRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    activeRef.current?.scrollIntoView({ behavior: "smooth", inline: "nearest", block: "nearest" });
  }, [selectedEpoch, history.length]);

  if (history.length === 0) return null;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "0 14px 8px" }}>
      <div style={{
        display: "flex", gap: 5, overflowX: "auto", overflowY: "hidden",
        paddingBottom: 2, flex: 1, scrollbarWidth: "thin",
      }}>
        {history.map((entry) => {
          const active = entry.epoch === selectedEpoch;
          const thumb = pathFor(entry, "sr") ?? pathFor(entry, "lr");
          return (
            <button
              key={entry.epoch}
              ref={active ? activeRef : undefined}
              onClick={() => onSelect(entry.epoch)}
              title={`Epoch ${entry.epoch}${entryPsnr(entry) != null ? ` · PSNR ${fmt(entryPsnr(entry))} dB` : ""}`}
              style={{
                flexShrink: 0, width: 42, height: 42, borderRadius: "var(--radius-sm)",
                border: active ? "1.5px solid var(--blue)" : "1.5px solid var(--border)",
                background: "var(--bg2)", padding: 0, cursor: "pointer", overflow: "hidden",
                position: "relative", transition: "border-color 0.12s ease",
              }}
            >
              {thumb ? (
                <img src={convertFileSrc(thumb)} alt={`epoch ${entry.epoch}`}
                  style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} loading="lazy" />
              ) : (
                <span style={{ fontSize: 8, color: "var(--dim)" }}>—</span>
              )}
              <span style={{
                position: "absolute", bottom: 0, left: 0, right: 0, textAlign: "center",
                fontSize: 7.5, fontFamily: "var(--font-mono)", color: "var(--text)",
                background: "rgba(13,15,17,0.72)", lineHeight: "11px",
              }}>
                {entry.epoch}
              </span>
            </button>
          );
        })}
      </div>
      {!isLive && (
        <button
          onClick={onResumeLive}
          style={{
            flexShrink: 0, fontSize: 9.5, fontFamily: "var(--font-mono)", cursor: "pointer",
            color: "var(--green)", background: "var(--green-dim)", border: "1px solid rgba(77,186,127,0.3)",
            borderRadius: 20, padding: "3px 9px", display: "flex", alignItems: "center", gap: 4,
          }}
        >
          ● jump to latest
        </button>
      )}
    </div>
  );
}