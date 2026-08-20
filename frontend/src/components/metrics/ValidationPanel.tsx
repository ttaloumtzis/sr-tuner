import { useMemo, useRef, useState, type ReactNode } from "react";
import { useTrainingStore } from "../../store/trainingStore";
import type { ValidationFrames } from "../../store/trainingStore";
import { fmt } from "../../lib/format";
import { FRAME_ORDER, FRAME_META, pathFor } from "./frameMeta";
import { Filmstrip } from "./Filmstrip";
import { Lightbox } from "./Lightbox";
import { FrameCell } from "./FrameCell";

function PanelHeader({ label, right }: { label: string; right?: ReactNode }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "9px 14px 7px", flexShrink: 0, gap: 8,
    }}>
      <span style={{
        fontSize: 10, letterSpacing: "0.06em", color: "var(--muted)",
        fontFamily: "var(--font-mono)", fontWeight: 600, textTransform: "uppercase",
      }}>
        {label}
      </span>
      {right}
    </div>
  );
}

export function ValidationPanel() {
  const latestFrames = useTrainingStore((s) => s.validationFrames);
  const history = useTrainingStore((s) => s.validationHistory);
  const validationRunning = useTrainingStore((s) => s.validationRunning);
  const [pinnedEpoch, setPinnedEpoch] = useState<number | null>(null);
  const [lightbox, setLightbox] = useState<{ epoch: number; kind: (typeof FRAME_ORDER)[number] } | null>(null);
  const lightboxTriggerRef = useRef<HTMLElement | null>(null);

  const latestEpoch = history.length > 0 ? history[history.length - 1].epoch : null;
  const isLive = pinnedEpoch == null;
  const selectedEpoch = pinnedEpoch ?? latestEpoch;

  const openLightbox = (kind: (typeof FRAME_ORDER)[number]) => {
    if (selectedEpoch == null) return;
    lightboxTriggerRef.current = document.activeElement as HTMLElement | null;
    setLightbox({ epoch: selectedEpoch, kind });
  };

  const closeLightbox = () => {
    setLightbox(null);
    lightboxTriggerRef.current?.focus?.();
    lightboxTriggerRef.current = null;
  };

  const selectedEntry = useMemo(
    () => history.find((e) => e.epoch === selectedEpoch) ?? null,
    [history, selectedEpoch],
  );

  const activeFrames: ValidationFrames | null = selectedEntry ?? (isLive ? latestFrames : null);

  const cells = FRAME_ORDER.map((kind) => ({
    kind, label: FRAME_META[kind].label, path: pathFor(activeFrames, kind),
  }));

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <PanelHeader
        label="Validation Frames"
        right={
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {selectedEntry?.psnr != null && (
              <span style={{ fontSize: 9.5, color: "var(--green)", fontFamily: "var(--font-mono)" }}>
                {fmt(selectedEntry.psnr)} dB
              </span>
            )}
            {selectedEpoch != null && (
              <span style={{
                fontSize: 9.5, fontFamily: "var(--font-mono)", color: isLive ? "var(--green)" : "var(--muted)",
                background: isLive ? "var(--green-dim)" : "var(--bg2)", padding: "2px 8px", borderRadius: 20,
                border: isLive ? "1px solid rgba(77,186,127,0.3)" : "1px solid var(--border)",
              }}>
                {isLive && validationRunning
                  ? `● validating · e${selectedEpoch}`
                  : isLive
                    ? `● live · e${selectedEpoch}`
                    : `epoch ${selectedEpoch}`}
              </span>
            )}
          </div>
        }
      />

      <Filmstrip
        history={history}
        selectedEpoch={selectedEpoch}
        isLive={isLive}
        onSelect={(epoch) => setPinnedEpoch(epoch === latestEpoch ? null : epoch)}
        onResumeLive={() => setPinnedEpoch(null)}
      />

      <div style={{
        display: "grid", gridTemplateColumns: "1fr 1fr", gridTemplateRows: "1fr 1fr",
        gap: 6, flex: 1, minHeight: 0, padding: "0 14px 14px",
      }}>
        {cells.map(({ kind, label, path }) => (
          <FrameCell key={kind} label={label} path={path} onExpand={() => openLightbox(kind)} />
        ))}
      </div>

      {lightbox && history.length > 0 && (
        <Lightbox
          history={history}
          epoch={lightbox.epoch}
          kind={lightbox.kind}
          onClose={closeLightbox}
          onNavigate={(epoch, kind) => {
            setLightbox({ epoch, kind });
            if (epoch !== latestEpoch) setPinnedEpoch(epoch);
            else setPinnedEpoch(null);
          }}
        />
      )}
    </div>
  );
}