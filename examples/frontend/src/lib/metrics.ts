import type { MetricsEvent } from "./ipc-types";

/**
 * Return a downsampled view of events at a uniform stride so that at most
 * maxPoints are rendered.  The last event is always included so the chart
 * tip is always current.  When events.length <= maxPoints the original array
 * is returned unchanged.
 *
 * Used only in the chart render path — the store keeps the full array.
 */
export function downsample(
  events: MetricsEvent[],
  maxPoints = 200
): MetricsEvent[] {
  if (events.length <= maxPoints) return events;

  const stride = Math.ceil(events.length / maxPoints);
  const result: MetricsEvent[] = [];

  for (let i = 0; i < events.length; i += stride) {
    result.push(events[i]);
  }

  // Guarantee the last event is always present.
  const last = events[events.length - 1];
  if (result[result.length - 1] !== last) {
    result.push(last);
  }

  return result;
}
