const IS_WINDOWS = navigator.userAgent.includes("Windows");
const SEP = IS_WINDOWS ? "\\" : "/";
const SEP_RE = /[/\\]/;

export function basename(path: string): string {
  return path.split(SEP_RE).pop() ?? path;
}

function detectSep(path: string): string {
  const firstSep = path.match(/[/\\]/);
  return firstSep ? firstSep[0] : SEP;
}

export function dirname(path: string): string {
  const parts = path.split(SEP_RE);
  if (parts.length <= 1) return ".";
  if (parts.every(p => p === "")) return ".";
  parts.pop();
  return parts.join(detectSep(path));
}

export function join(...parts: string[]): string {
  const nonEmpty = parts.filter(p => p.length > 0);
  if (nonEmpty.length === 0) return "";
  return nonEmpty
    .map(p => p.replace(/[/\\]+$/, ""))
    .filter(Boolean)
    .join(SEP);
}

export function normalize(path: string): string {
  return path.replace(/[/\\]+/g, SEP);
}

export function parentFromProjFile(path: string): string {
  const result = path.replace(/[/\\][^/\\]+\.srproj$/, "");
  return result !== path ? result : dirname(path);
}
