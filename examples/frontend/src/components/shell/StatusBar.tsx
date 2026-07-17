import { useProjectStore } from "../../store/projectStore";
import { useTrainingStore } from "../../store/trainingStore";
import { useRunConfigStore } from "../../store/runConfigStore";
import { useUiStore } from "../../store/uiStore";

function etaMmSs(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function StatusBar() {
  const project = useProjectStore((s) => s.project);
  const closeProject = useProjectStore((s) => s.closeProject);
  const trainingStatus = useTrainingStore((s) => s.status);
  const epoch = useTrainingStore((s) => s.epoch);
  const etaSec = useTrainingStore((s) => s.etaSec);
  const resumeFrom = useRunConfigStore((s) => s.resumeFrom);
  const totalEpochs = useRunConfigStore((s) => s.schedule.totalEpochs);
  const deviceName = useUiStore((s) => s.deviceName);

  const isRunning = trainingStatus === "running";
  const arch = project?.default_model?.architecture ?? null;

  return (
    <div
      style={{
        height: "var(--statusbar-h)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        background: "var(--bg1)",
        borderTop: "1px solid var(--border)",
        padding: "0 12px",
        flexShrink: 0,
        fontSize: 11,
        fontFamily: "var(--font-mono)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
        {project && (
          <span style={{ color: "var(--green)", fontWeight: 600 }}>
            {project.name}
          </span>
        )}
        {arch && <span style={{ color: "var(--muted)" }}>{arch}</span>}
        {resumeFrom && (
          <span style={{ color: "var(--amber)", fontSize: 10 }}>
            resume from ep{resumeFrom.resume_epoch} queued
          </span>
        )}
        {isRunning && (
          <span style={{ color: "var(--amber)" }}>
            Ep {epoch} / {totalEpochs}
            {etaSec !== null && ` · ETA ${etaMmSs(etaSec)}`}
          </span>
        )}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 12, flexShrink: 0 }}>
        {deviceName && (
          <span style={{ color: "var(--muted)" }}>{deviceName}</span>
        )}
        <button
          onClick={closeProject}
          style={{
            background: "none",
            border: "none",
            color: "var(--muted)",
            fontSize: 11,
            fontFamily: "var(--font-mono)",
            cursor: "pointer",
            padding: 0,
          }}
          onMouseEnter={(e) =>
            ((e.target as HTMLButtonElement).style.color = "var(--text)")
          }
          onMouseLeave={(e) =>
            ((e.target as HTMLButtonElement).style.color = "var(--muted)")
          }
          title="Return to project list"
        >
          ← projects
        </button>
      </div>
    </div>
  );
}
