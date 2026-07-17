export function calcEta(
  totalIter: number,
  currentIter: number,
  speed: number,
): number | null {
  if (speed <= 0) return null;
  const remaining = totalIter - currentIter;
  return remaining > 0 ? remaining / speed : null;
}
