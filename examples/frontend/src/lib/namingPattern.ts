const NAMING_PATTERN_RE = /^%0?\d*d$/;

export function validateNamingPattern(pattern: string): string | null {
  if (!pattern || pattern === "") return null;
  if (!NAMING_PATTERN_RE.test(pattern)) {
    return "Invalid pattern — use printf integer format (e.g. %06d, %04d)";
  }
  return null;
}

export function previewFilename(pattern: string): string {
  const p = pattern || "%06d";
  const match = p.match(/^%0?(\d*)d$/);
  if (!match) return "invalid";
  const width = parseInt(match[1] || "0", 10);
  return String(1).padStart(width, "0") + ".png";
}
