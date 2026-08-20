import { invoke } from "@tauri-apps/api/core";
import { useInferenceStore } from "../../store/inferenceStore";
import { Btn } from "../../components/ui/Btn";
import { InfoRow } from "../../components/ui/InfoRow";

export function ResultFooter() {
  const result = useInferenceStore((s) => s.result);
  const status = useInferenceStore((s) => s.status);
  const errorMsg = useInferenceStore((s) => s.errorMsg);
  const gtPath = useInferenceStore((s) => s.gtPath);

  if (status === "error") {
    return (
      <div className="si-error-bar">
        <span className="si-error-icon">⚠</span>
        <span className="si-error-msg">
          {errorMsg || "Inference failed"}
        </span>
        <Btn small color="var(--red)" onClick={() => useInferenceStore.getState().resetRun()}>Dismiss</Btn>
      </div>
    );
  }

  if (status !== "done" || !result?.success) return null;

  const metrics = result.metrics;
  const outRes = result.output_resolution;
  const inRes = result.input_resolution;

  let scaleLabel = "—";
  if (inRes && outRes && inRes.width > 0) {
    const sx = outRes.width / inRes.width;
    const sy = outRes.height / inRes.height;
    if (sx === sy) scaleLabel = `${sx}×`;
  }
  const timeMs = result.inference_time_ms;
  const timeLabel = timeMs != null
    ? timeMs < 1000 ? `${timeMs.toFixed(0)} ms` : `${(timeMs / 1000).toFixed(2)} s`
    : "—";

  return (
    <div className="si-footer">
      {/* Metrics */}
      <div className="si-col-200">
        <div className="si-section-label">Quality Metrics</div>
        <InfoRow label="PSNR" value={metrics?.psnr ?? null} color="var(--green)" dec={2} border emphasis labelSize={10} labelMono />
        <InfoRow label="SSIM" value={metrics?.ssim ?? null} color="var(--blue)" dec={4} border emphasis labelSize={10} labelMono />
        <InfoRow label="LPIPS" value={metrics?.lpips ?? null} color="var(--muted)" dec={4} border emphasis labelSize={10} labelMono />
        <InfoRow label="MS-SSIM" value={metrics?.ms_ssim ?? null} color="var(--muted)" dec={4} border emphasis labelSize={10} labelMono />
        {!gtPath && !metrics && (
          <div className="si-hint-sm">Add a GT image to compute metrics</div>
        )}
      </div>

      {/* Info */}
      <div className="si-col-180">
        <div className="si-section-label">Image Info</div>
        <InfoRow label="Input" value={inRes ? `${inRes.width}×${inRes.height}` : "—"} border labelSize={10} labelMono />
        <InfoRow label="Output" value={outRes ? `${outRes.width}×${outRes.height}` : "—"} border labelSize={10} labelMono />
        <InfoRow label="Scale" value={scaleLabel} border labelSize={10} labelMono />
        <InfoRow label="Time" value={timeLabel} border labelSize={10} labelMono />
      </div>

      {/* Output path */}
      {result.output && (
        <div className="si-output-col">
          <div className="si-section-label">Output</div>
          <div className="si-output-path">{result.output}</div>
          <div>
            <Btn small onClick={() => { try { invoke("open_in_file_manager", { path: result.output }); } catch { /* browser mode */ } }}>
              Open in file manager
            </Btn>
          </div>
        </div>
      )}
    </div>
  );
}