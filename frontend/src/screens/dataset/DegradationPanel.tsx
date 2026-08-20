import type { CSSProperties } from "react";
import { Panel } from "../../components/ui/Panel";
import { NumberInput } from "../../components/ui/NumberInput";
import { useDatasetStore } from "../../store/datasetStore";

const rangeInputStyle: CSSProperties = { width: "clamp(55px, 5vw, 80px)", fontSize: 10, padding: "2px 6px" };

function RangeRow({ label, min, max, valueMin, valueMax, onMin, onMax, step = 1, unit = "" }: {
  label: string; min: number; max: number; valueMin: number; valueMax: number;
  onMin: (v: number) => void; onMax: (v: number) => void; step?: number; unit?: string;
}) {
  return (
    <div className="dsc-range-row">
      <span className="dsc-field-label">{label}</span>
      <NumberInput value={valueMin} min={min} max={max} step={step} onChange={onMin} style={rangeInputStyle} />
      <span className="dsc-dim">→</span>
      <NumberInput value={valueMax} min={min} max={max} step={step} onChange={onMax} style={rangeInputStyle} />
      <span className="dsc-dim">{unit}</span>
    </div>
  );
}

function DegSection({ title, enabled, onToggle, children }: { title: string; enabled: boolean; onToggle: (v: boolean) => void; children: React.ReactNode }) {
  return (
    <div className="dsc-deg-section">
      <div
        onClick={() => onToggle(!enabled)}
        className={`dsc-deg-head${enabled ? " dsc-deg-head-on" : " dsc-deg-head-off"}`}
      >
        <div className={`dsc-deg-radio${enabled ? " dsc-deg-radio-on" : " dsc-deg-radio-off"}`}>
          <span className={`dsc-deg-check${enabled ? " dsc-deg-check-on" : " dsc-deg-check-off"}`}>✓</span>
        </div>
        <span className={`dsc-deg-title${enabled ? " dsc-deg-title-on" : " dsc-deg-title-off"}`}>{title}</span>
      </div>
      {enabled && <div className="dsc-deg-body">{children}</div>}
    </div>
  );
}

export function DegradationPanel() {
  const s = useDatasetStore();

  return (
    <div className="dsc-col">
      <Panel title="Extraction Settings">
        <div className="dsc-stack">
          <div className="dsc-field-row-wrap">
            <div className="dsc-field-row">
              <span className="dsc-field-label-bold">FPS</span>
              <input type="number" value={s.frameRate} min={1} max={120} onChange={(e) => s.setFrameRate(Number(e.target.value))}
                className="dsc-num" />
            </div>
            <div className="dsc-field-row">
              <span className="dsc-field-label-bold">Start (s)</span>
              <input type="number" value={s.startTime} min={0} step={0.1} onChange={(e) => s.setStartTime(Number(e.target.value))}
                className="dsc-num" />
            </div>
            <div className="dsc-field-row">
              <span className="dsc-field-label-bold">Duration (s)</span>
              <input type="number" value={s.duration ?? ""} min={0} step={0.1} placeholder="∞"
                onChange={(e) => s.setDuration(e.target.value ? Number(e.target.value) : null)}
                className="dsc-num" />
            </div>
          </div>
        </div>
      </Panel>

      <Panel title="Degradations">
        <div className="dsc-stack">
          <DegSection title="Blur" enabled={s.degBlur} onToggle={s.setDegBlur}>
            <div className="dsc-stack-5">
              <div className="dsc-field-row">
                <span className="dsc-field-label">Gaussian kernel</span>
                <input type="number" value={s.blurKernelSize} min={3} max={61} step={2} onChange={(e) => s.setBlurKernelSize(Number(e.target.value))}
                  className="dsc-num" />
              </div>
              <RangeRow label="Sigma range" min={0.1} max={10} step={0.1} valueMin={s.blurSigmaMin} valueMax={s.blurSigmaMax} onMin={(v) => s.setBlurSigmaRange(v, s.blurSigmaMax)} onMax={(v) => s.setBlurSigmaRange(s.blurSigmaMin, v)} />
              <ProbControl label="Apply prob" value={s.blurGaussianProb} onChange={s.setBlurGaussianProb} />
              <div className="dsc-field-row-mt">
                <label className="dsc-check-label">
                  <input type="checkbox" checked={s.motionBlurEnabled} onChange={(e) => s.setMotionBlurEnabled(e.target.checked)} className="dsc-check" />
                  Motion blur
                </label>
                {s.motionBlurEnabled && (
                  <>
                    <span className="dsc-dim">max kernel</span>
                    <input type="number" value={s.motionBlurMaxKernel} min={3} max={99} step={2} onChange={(e) => s.setMotionBlurMaxKernel(Number(e.target.value))}
                      className="dsc-num-sm" />
                    <ProbControl label="prob" value={s.blurMotionProb} onChange={s.setBlurMotionProb} />
                  </>
                )}
              </div>
            </div>
          </DegSection>

          <DegSection title="Noise" enabled={s.degNoise} onToggle={s.setDegNoise}>
            <div className="dsc-stack-5">
              <RangeRow label="Gaussian σ" min={0} max={100} step={1} valueMin={s.noiseSigmaMin} valueMax={s.noiseSigmaMax} onMin={(v) => s.setNoiseSigmaRange(v, s.noiseSigmaMax)} onMax={(v) => s.setNoiseSigmaRange(s.noiseSigmaMin, v)} />
              <ProbControl label="Gaussian prob" value={s.noiseGaussianProb} onChange={s.setNoiseGaussianProb} />
              <RangeRow label="Poisson scale" min={0.01} max={10} step={0.01} valueMin={s.poissonScaleMin} valueMax={s.poissonScaleMax} onMin={(v) => s.setPoissonScaleRange(v, s.poissonScaleMax)} onMax={(v) => s.setPoissonScaleRange(s.poissonScaleMin, v)} />
              <ProbControl label="Poisson prob" value={s.noisePoissonProb} onChange={s.setNoisePoissonProb} />
              <div className="dsc-field-row">
                <span className="dsc-field-label">Salt & pepper</span>
                <input type="number" value={s.saltPepperAmount} min={0} max={0.1} step={0.001} onChange={(e) => s.setSaltPepperAmount(Number(e.target.value))}
                  className="dsc-num" />
                <ProbControl label="prob" value={s.noiseSaltPepperProb} onChange={s.setNoiseSaltPepperProb} />
              </div>
            </div>
          </DegSection>

          <DegSection title="JPEG" enabled={s.degJpeg} onToggle={s.setDegJpeg}>
            <div className="dsc-stack-5">
              <RangeRow label="Quality" min={1} max={100} step={1} valueMin={s.jpegQualityMin} valueMax={s.jpegQualityMax} onMin={(v) => s.setJpegQualityRange(v, s.jpegQualityMax)} onMax={(v) => s.setJpegQualityRange(s.jpegQualityMin, v)} />
              <ProbControl label="Apply prob" value={s.jpegProb} onChange={s.setJpegProb} />
            </div>
          </DegSection>

          <DegSection title="JPEG2000" enabled={s.degJpeg2000} onToggle={s.setDegJpeg2000}>
            <div className="dsc-stack-5">
              <RangeRow label="Quality" min={1} max={100} step={1} valueMin={s.jpeg2000QualityMin} valueMax={s.jpeg2000QualityMax} onMin={(v) => s.setJpeg2000QualityRange(v, s.jpeg2000QualityMax)} onMax={(v) => s.setJpeg2000QualityRange(s.jpeg2000QualityMin, v)} />
              <ProbControl label="Apply prob" value={s.jpeg2000Prob} onChange={s.setJpeg2000Prob} />
            </div>
          </DegSection>

          <DegSection title="Color Jitter" enabled={s.degColorJitter} onToggle={s.setDegColorJitter}>
            <div className="dsc-stack-5">
              <div className="dsc-field-row">
                <span className="dsc-field-label">Hue</span>
                <span className="dsc-dim">±</span>
                <input type="number" value={s.jitterHueRange} min={0} max={0.5} step={0.01} onChange={(e) => s.setJitterHueRange(Number(e.target.value))}
                  className="dsc-num" />
              </div>
              <div className="dsc-field-row">
                <span className="dsc-field-label">Saturation</span>
                <span className="dsc-dim">±</span>
                <input type="number" value={s.jitterSaturationRange} min={0} max={1} step={0.01} onChange={(e) => s.setJitterSaturationRange(Number(e.target.value))}
                  className="dsc-num" />
              </div>
              <div className="dsc-field-row">
                <span className="dsc-field-label">Value</span>
                <span className="dsc-dim">±</span>
                <input type="number" value={s.jitterValueRange} min={0} max={1} step={0.01} onChange={(e) => s.setJitterValueRange(Number(e.target.value))}
                  className="dsc-num" />
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
    <div className="dsc-prob-row">
      <span className="dsc-field-label">{label}</span>
      <input type="range" min={0} max={1} step={0.01} value={value} onChange={(e) => onChange(Number(e.target.value))}
        className="dsc-prob-range" />
      <span className="dsc-prob-value">{value.toFixed(2)}</span>
    </div>
  );
}