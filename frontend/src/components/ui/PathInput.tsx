import { Btn } from "./Btn";
import { open } from "@tauri-apps/plugin-dialog";

interface PathInputProps {
  value?: string;
  onChange?: (path: string) => void;
  browseTitle?: string;
  mono?: boolean;
  compact?: boolean;
  placeholder?: string;
  fileFilters?: { name: string; extensions: string[] }[];
}

export function PathInput({
  value = "",
  onChange,
  browseTitle,
  mono,
  compact,
  placeholder = "No path selected",
  fileFilters,
}: PathInputProps) {
  const handleBrowse = async () => {
    const selected = await open({
      directory: !fileFilters,
      multiple: false,
      title: browseTitle,
      defaultPath: value || undefined,
      filters: fileFilters,
    });
    if (selected) {
      onChange?.(selected);
    }
  };

  return (
    <div style={{ display: "flex", gap: 6, minWidth: 0 }}>
      <div
        style={{
          background: "var(--bg3)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-sm)",
          padding: compact ? "4px 8px" : "6px 10px",
          fontSize: compact ? 10 : 12,
          color: value ? "var(--text)" : "var(--dim)",
          flex: 1,
          minWidth: 0,
          fontFamily: mono ? "var(--font-mono)" : "var(--font-sans)",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          outline: "none",
        }}
      >
        {value || placeholder}
      </div>
      <Btn
        small={compact}
        onClick={handleBrowse}
        style={{ flexShrink: 0 }}
      >
        Browse…
      </Btn>
    </div>
  );
}
