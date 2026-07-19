import { Panel } from "../../components/ui/Panel";
import { Dropdown } from "../../components/ui/Dropdown";
import { useDatasetStore } from "../../store/datasetStore";

function RangeRow({ label, min, max, valueMin, valueMax, onMin, onMax, step = 1, unit = "" }: {
  label: string; min: number; max: number; valueMin: number; valueMax: number;
  onMin: (v: number) => void; onMax: (v: number) => void; step?: number; unit?: string;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
      <span style={{ fontSize: 10, color: "var(--muted)", minWidth: 75 }}>{label}</span>
      <input
        type="number" value={valueMin} min={min} max={max} step={step}
        onChange={(e) => onMin(Number(e.target.value))}
        style={{ width: "clamp(55px, 5vw, 80px)", background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", color: "var(--text)", fontSize: 10, padding: "2px 6px", fontFamily: "var(--font-mono)", outline: "none" }}
      />
      <span style={{ fontSize: 10, color: "var(--dim)" }}>→</span>
      <input
        type="number" value={valueMax} min={min} max={max} step={step}
        onChange={(e) => onMax(Number(e.target.value))}
        style={{ width: "clamp(55px, 5vw, 80px)", background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", color: "var(--text)", fontSize: 10, padding: "2px 6px", fontFamily: "var(--font-mono)", outline: "none" }}
      />
      <span style={{ fontSize: 10, color: "var(--dim)" }}>{unit}</span>
    </div>
  );
}

function DegSection({ title, enabled, onToggle, children }: { title: string; enabled: boolean; onToggle: (v: boolean) => void; children: React.ReactNode }) {
  return (
    <div style={{ border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", overflow: "hidden" }}>
      <div
        onClick={() => onToggle(!enabled)}
        style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 10px", background: enabled ? "var(--bg2)" : "var(--bg1)", cursor: "pointer", userSelect: "none", borderBottom: enabled ? "1px solid var(--border)" : "none" }}
      >
        <div style={{ width: 14, height: 14, borderRadius: "50%", background: enabled ? "var(--green)" : "var(--bg3)", border: `1px solid ${enabled ? "var(--green)" : "var(--border)"}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
          <span style={{ fontSize: 8, color: enabled ? "#0d0f11" : "transparent", fontWeight: 700 }}>✓</span>
        </div>
        <span style={{ fontSize: 11, fontWeight: 600, color: enabled ? "var(--text)" : "var(--muted)" }}>{title}</span>
      </div>
      {enabled && <div style={{ padding: "8px 10px", display: "flex", flexDirection: "column", gap: 6 }}>{children}</div>}
    </div>
  );
}

export function DegradationPanel() {
  const s = useDatasetStore();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <Panel title="Extraction Settings">
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600 }}>FPS</span>
              <input type="number" value={s.frameRate} min={1} max={120} onChange={(e) => s.setFrameRate(Number(e.target.value))}
                style={{ width: "clamp(50px, 4vw, 75px)", background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", color: "var(--text)", fontSize: 10, padding: "2px 6px", fontFamily: "var(--font-mono)", outline: "none" }} />
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600 }}>Format</span>
              <Dropdown value={s.frameFormat} options={["png", "jpg", "webp"]} onChange={s.setFrameFormat} mono />
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600 }}>Start (s)</span>
              <input type="number" value={s.startTime} min={0} step={0.1} onChange={(e) => s.setStartTime(Number(e.target.value))}
                style={{ width: "clamp(50px, 4vw, 75px)", background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", color: "var(--text)", fontSize: 10, padding: "2px 6px", fontFamily: "var(--font-mono)", outline: "none" }} />
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600 }}>Duration (s)</span>
              <input type="number" value={s.duration ?? ""} min={0} step={0.1} placeholder="∞"
                onChange={(e) => s.setDuration(e.target.value ? Number(e.target.value) : null)}
                style={{ width: "clamp(50px, 4vw, 75px)", background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", color: "var(--text)", fontSize: 10, padding: "2px 6px", fontFamily: "var(--font-mono)", outline: "none" }} />
            </div>
          </div>
        </div>
      </Panel>

      <Panel title="Degradations">
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <DegSection title="Blur" enabled={s.degBlur} onToggle={s.setDegBlur}>
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 10, color: "var(--muted)", minWidth: 75 }}>Gaussian kernel</span>
                <input type="number" value={s.blurKernelSize} min={3} max={61} step={2} onChange={(e) => s.setBlurKernelSize(Number(e.target.value))}
                  style={{ width: "clamp(50px, 4vw, 75px)", background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", color: "var(--text)", fontSize: 10, padding: "2px 6px", fontFamily: "var(--font-mono)", outline: "none" }} />
              </div>
              <RangeRow label="Sigma range" min={0.1} max={10} step={0.1} valueMin={s.blurSigmaMin} valueMax={s.blurSigmaMax} onMin={(v) => s.setBlurSigmaRange(v, s.blurSigmaMax)} onMax={(v) => s.setBlurSigmaRange(s.blurSigmaMin, v)} />
              <ProbControl label="Apply prob" value={s.blurGaussianProb} onChange={s.setBlurGaussianProb} />
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 4 }}>
                <label style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, color: "var(--muted)", cursor: "pointer" }}>
                  <input type="checkbox" checked={s.motionBlurEnabled} onChange={(e) => s.setMotionBlurEnabled(e.target.checked)} style={{ accentColor: "var(--green)" }} />
                  Motion blur
                </label>
                {s.motionBlurEnabled && (
                  <>
                    <span style={{ fontSize: 10, color: "var(--muted)" }}>max kernel</span>
                    <input type="number" value={s.motionBlurMaxKernel} min={3} max={99} step={2} onChange={(e) => s.setMotionBlurMaxKernel(Number(e.target.value))}
                      style={{ width: "clamp(45px, 4vw, 70px)", background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", color: "var(--text)", fontSize: 10, padding: "2px 6px", fontFamily: "var(--font-mono)", outline: "none" }} />
                    <ProbControl label="prob" value={s.blurMotionProb} onChange={s.setBlurMotionProb} />
                  </>
                )}
              </div>
            </div>
          </DegSection>

          <DegSection title="Noise" enabled={s.degNoise} onToggle={s.setDegNoise}>
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              <RangeRow label="Gaussian σ" min={0} max={100} step={1} valueMin={s.noiseSigmaMin} valueMax={s.noiseSigmaMax} onMin={(v) => s.setNoiseSigmaRange(v, s.noiseSigmaMax)} onMax={(v) => s.setNoiseSigmaRange(s.noiseSigmaMin, v)} />
              <ProbControl label="Gaussian prob" value={s.noiseGaussianProb} onChange={s.setNoiseGaussianProb} />
              <RangeRow label="Poisson scale" min={0.01} max={10} step={0.01} valueMin={s.poissonScaleMin} valueMax={s.poissonScaleMax} onMin={(v) => s.setPoissonScaleRange(v, s.poissonScaleMax)} onMax={(v) => s.setPoissonScaleRange(s.poissonScaleMin, v)} />
              <ProbControl label="Poisson prob" value={s.noisePoissonProb} onChange={s.setNoisePoissonProb} />
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 10, color: "var(--muted)", minWidth: 75 }}>Salt & pepper</span>
                <input type="number" value={s.saltPepperAmount} min={0} max={0.1} step={0.001} onChange={(e) => s.setSaltPepperAmount(Number(e.target.value))}
                  style={{ width: "clamp(50px, 4vw, 75px)", background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", color: "var(--text)", fontSize: 10, padding: "2px 6px", fontFamily: "var(--font-mono)", outline: "none" }} />
                <ProbControl label="prob" value={s.noiseSaltPepperProb} onChange={s.setNoiseSaltPepperProb} />
              </div>
            </div>
          </DegSection>

          <DegSection title="JPEG" enabled={s.degJpeg} onToggle={s.setDegJpeg}>
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              <RangeRow label="Quality" min={1} max={100} step={1} valueMin={s.jpegQualityMin} valueMax={s.jpegQualityMax} onMin={(v) => s.setJpegQualityRange(v, s.jpegQualityMax)} onMax={(v) => s.setJpegQualityRange(s.jpegQualityMin, v)} />
              <ProbControl label="Apply prob" value={s.jpegProb} onChange={s.setJpegProb} />
            </div>
          </DegSection>

          <DegSection title="JPEG2000" enabled={s.degJpeg2000} onToggle={s.setDegJpeg2000}>
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              <RangeRow label="Quality" min={1} max={100} step={1} valueMin={s.jpeg2000QualityMin} valueMax={s.jpeg2000QualityMax} onMin={(v) => s.setJpeg2000QualityRange(v, s.jpeg2000QualityMax)} onMax={(v) => s.setJpeg2000QualityRange(s.jpeg2000QualityMin, v)} />
              <ProbControl label="Apply prob" value={s.jpeg2000Prob} onChange={s.setJpeg2000Prob} />
            </div>
          </DegSection>

          <DegSection title="Color Jitter" enabled={s.degColorJitter} onToggle={s.setDegColorJitter}>
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 10, color: "var(--muted)", minWidth: 75 }}>Hue</span>
                <span style={{ fontSize: 10, color: "var(--dim)" }}>±</span>
                <input type="number" value={s.jitterHueRange} min={0} max={0.5} step={0.01} onChange={(e) => s.setJitterHueRange(Number(e.target.value))}
                  style={{ width: "clamp(50px, 4vw, 75px)", background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", color: "var(--text)", fontSize: 10, padding: "2px 6px", fontFamily: "var(--font-mono)", outline: "none" }} />
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 10, color: "var(--muted)", minWidth: 75 }}>Saturation</span>
                <span style={{ fontSize: 10, color: "var(--dim)" }}>±</span>
                <input type="number" value={s.jitterSaturationRange} min={0} max={1} step={0.01} onChange={(e) => s.setJitterSaturationRange(Number(e.target.value))}
                  style={{ width: "clamp(50px, 4vw, 75px)", background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", color: "var(--text)", fontSize: 10, padding: "2px 6px", fontFamily: "var(--font-mono)", outline: "none" }} />
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 10, color: "var(--muted)", minWidth: 75 }}>Value</span>
                <span style={{ fontSize: 10, color: "var(--dim)" }}>±</span>
                <input type="number" value={s.jitterValueRange} min={0} max={1} step={0.01} onChange={(e) => s.setJitterValueRange(Number(e.target.value))}
                  style={{ width: "clamp(50px, 4vw, 75px)", background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", color: "var(--text)", fontSize: 10, padding: "2px 6px", fontFamily: "var(--font-mono)", outline: "none" }} />
              </div>
              <ProbControl label="Apply prob" value={s.jitterProb} onChange={s.setJitterProb} />
            </div>
          </DegSection>
        </div>
      </Panel>
    </div>
  );
}

function ProbControl({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span style={{ fontSize: 10, color: "var(--muted)", minWidth: 75 }}>{label}</span>
      <input type="range" min={0} max={1} step={0.01} value={value} onChange={(e) => onChange(Number(e.target.value))}
        style={{ flex: 1, width: "100%", accentColor: "var(--green)", height: 4 }} />
      <span style={{ fontSize: 10, color: "var(--text)", fontFamily: "var(--font-mono)", minWidth: 30, textAlign: "right" }}>{value.toFixed(2)}</span>
    </div>
  );
}