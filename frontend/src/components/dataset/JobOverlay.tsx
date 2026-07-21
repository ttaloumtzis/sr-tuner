import { useEffect, useRef, useState } from "react";
import { useDatasetStore } from "../../store/datasetStore";
import { PBar } from "../ui/PBar";

export function JobOverlay() {
  const jobStatus = useDatasetStore((s) => s.jobStatus);
  const jobError = useDatasetStore((s) => s.jobError);
  const steps = useDatasetStore((s) => s.progressSteps);
  const clearJob = useDatasetStore((s) => s.clearJob);
  const [visible, setVisible] = useState(false);
  const [autoDismiss, setAutoDismiss] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (jobStatus === "running") {
      setVisible(true);
      setAutoDismiss(false);
      if (timerRef.current) clearTimeout(timerRef.current);
    } else if (jobStatus === "done") {
      if (!autoDismiss) {
        setAutoDismiss(true);
        timerRef.current = setTimeout(() => {
          clearJob();
          setVisible(false);
          setAutoDismiss(false);
        }, 1500);
      }
    } else if (jobStatus === "error") {
      setVisible(true);
      if (timerRef.current) clearTimeout(timerRef.current);
    } else {
      setVisible(false);
      setAutoDismiss(false);
    }
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [jobStatus]);

  const handleDismiss = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    clearJob();
    setVisible(false);
    setAutoDismiss(false);
  };

  const handleBackdropClick = () => {
    if (jobStatus === "done" || jobStatus === "error") {
      handleDismiss();
    }
  };

  if (!visible && jobStatus === "idle") return null;

  const colorMap: Record<string, string> = {
    active: "var(--amber)",
    done: "var(--green)",
    pending: "var(--dim)",
  };

  const iconMap: Record<string, string> = {
    active: "\u25B6",
    done: "\u2713",
    pending: "\u25CB",
  };

  return (
    <div style={{
      position: "fixed",
      inset: 0,
      zIndex: 9999,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      background: "rgba(0,0,0,0.55)",
      backdropFilter: "blur(2px)",
    }} onClick={handleBackdropClick}>
      <div onClick={(e) => e.stopPropagation()} style={{
        background: "var(--bg2)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)",
        width: 400,
        maxWidth: "90vw",
        boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}>
        {steps.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column" }}>
            {steps.map((step) => {
              const color = colorMap[step.status] || "var(--dim)";
              const icon = iconMap[step.status] || "\u25CB";
              const pct = step.total != null && step.total > 0 ? Math.round((step.current / step.total) * 100) : null;

              return (
                <div key={step.id} style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 3,
                  padding: "10px 14px",
                  borderBottom: step.id < steps.length - 1 ? "1px solid var(--border)" : undefined,
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontSize: 12, color }}>{icon}</span>
                    <span style={{ fontSize: 12, color: "var(--text)", fontWeight: step.status === "active" ? 600 : 400, flex: 1 }}>
                      {step.desc}
                    </span>
                    {step.status === "active" && (
                      <span style={{ fontSize: 11, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>
                        {step.total != null ? `${step.current}/${step.total} · ${pct}%` : `${step.current}`}
                      </span>
                    )}
                    {step.status === "done" && (
                      <span style={{ fontSize: 11, color: "var(--green)", fontFamily: "var(--font-mono)" }}>
                        {step.total != null ? `${step.current}/${step.total}` : `${step.current}`}
                      </span>
                    )}
                  </div>
                  {step.status === "active" && (
                    <PBar value={step.current} max={step.total ?? (step.current || 1)} color="var(--amber)" height={5} />
                  )}
                  {step.status === "done" && (
                    <PBar value={step.total ?? step.current} max={step.total ?? (step.current || 1)} color="var(--green)" height={5} />
                  )}
                </div>
              );
            })}
          </div>
        )}

        {steps.length === 0 && jobStatus === "running" && (
          <div style={{ padding: "16px 14px", fontSize: 12, color: "var(--muted)", textAlign: "center" }}>
            Starting...
          </div>
        )}

        <div style={{ padding: "8px 14px", display: "flex", alignItems: "center", justifyContent: "space-between", borderTop: "1px solid var(--border)" }}>
          {jobStatus === "done" && (
            <span style={{ fontSize: 12, color: "var(--green)", fontWeight: 600 }}>Complete</span>
          )}
          {jobStatus === "error" && (
            <span style={{ fontSize: 12, color: "var(--red)", fontWeight: 600, flex: 1 }}>
              {jobError || "Unknown error"}
            </span>
          )}
          {jobStatus === "running" && <span />}
          <button onClick={handleDismiss} style={{
            background: "var(--bg3)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)",
            color: "var(--muted)",
            fontSize: 11,
            padding: "4px 12px",
            cursor: "pointer",
            fontWeight: 500,
          }}>
            {jobStatus === "running" ? "Hide" : "Dismiss"}
          </button>
        </div>
      </div>
    </div>
  );
}
