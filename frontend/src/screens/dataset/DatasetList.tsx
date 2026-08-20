import { Search, CheckCircle } from "lucide-react";
import type { DatasetInfo } from "../../lib/api-types";
import type { JobStatus, ProgressStep } from "../../store/datasetStore";

interface DatasetListProps {
  datasets: DatasetInfo[];
  filteredDatasets: DatasetInfo[];
  selectedName: string | null;
  searchQuery: string;
  onSearchChange: (q: string) => void;
  selectedScaleFilter: string;
  onScaleFilterChange: (s: string) => void;
  jobStatus: JobStatus;
  jobType: "build" | "health" | "merge" | "prune" | "validate" | null;
  jobDatasetPath: string | null;
  progressSteps: ProgressStep[];
  onSelect: (name: string) => void;
}

export function DatasetList({
  datasets,
  filteredDatasets,
  selectedName,
  searchQuery,
  onSearchChange,
  selectedScaleFilter,
  onScaleFilterChange,
  jobStatus,
  jobType,
  jobDatasetPath,
  progressSteps,
  onSelect,
}: DatasetListProps) {
  return (
    <aside className="sr-sidebar">
      <div className="sidebar-header">
        <h3>Datasets</h3>
        <span className="dataset-count-badge">{filteredDatasets.length}</span>
      </div>

      <div className="sidebar-controls">
        <div className="search-input-wrapper">
          <Search size={14} className="search-icon" />
          <input
            type="text"
            placeholder="Search datasets…"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
          />
        </div>

        <div className="scale-filter-pills">
          {["all", ...Array.from(new Set(datasets.map((d) => `x${d.scale}`))).sort()].map((scale) => (
            <button
              key={scale}
              className={`scale-pill ${selectedScaleFilter === scale ? "active" : ""}`}
              onClick={() => onScaleFilterChange(scale)}
            >
              {scale.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      <div className="dataset-list">
        {filteredDatasets.map((ds) => {
          const isSelected = ds.name === selectedName;
          const validatingThis = jobStatus === "running" && jobType === "validate" && jobDatasetPath === ds.path;
          const activeStep = [...progressSteps].reverse().find((st) => st.status === "active");
          const valPct = validatingThis && activeStep && activeStep.total != null && activeStep.total > 0
            ? Math.round((activeStep.current / activeStep.total) * 100)
            : null;
          return (
            <div
              key={ds.name}
              className={`dataset-card ${isSelected ? "selected" : ""}`}
              onClick={() => onSelect(ds.name)}
            >
              <div className="card-top-row">
                <span className="dataset-name" title={ds.name}>
                  {ds.name}
                </span>
                {validatingThis ? (
                  <span className="validation-progress-badge" title={activeStep?.desc}>
                    {valPct != null ? `${valPct}%` : "…"}
                  </span>
                ) : (
                  <CheckCircle size={14} className="manifest-check" />
                )}
              </div>
              <div className="card-bottom-row">
                <span className="scale-tag">x{ds.scale}</span>
                <span className="pairs-count">{ds.num_pairs.toLocaleString()} pairs</span>
              </div>
            </div>
          );
        })}
      </div>
    </aside>
  );
}