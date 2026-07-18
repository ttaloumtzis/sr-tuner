import type { SSEEvent } from "./api-types";
import { useTrainingStore } from "../store/trainingStore";
import { useUiStore } from "../store/uiStore";

type EventHandler = (event: SSEEvent) => void;

const listeners = new Set<EventHandler>();

export function onSSEEvent(handler: EventHandler): () => void {
  listeners.add(handler);
  return () => listeners.delete(handler);
}

function dispatch(event: SSEEvent): void {
  for (const h of listeners) h(event);

  switch (event.type) {
    case "phase":
      if (event.phase === "cancelled") {
        useTrainingStore.setState({ status: "idle" });
      }
      break;

    case "step": {
      const prev = useTrainingStore.getState();
      const nextLoss = [...prev.lossHistory, event.loss as number].slice(-500);
      const nextPsnr = event.psnr != null
        ? [...prev.psnrHistory, event.psnr as number].slice(-500)
        : prev.psnrHistory;
      useTrainingStore.setState({
        iter: event.batch as number,
        epoch: event.epoch,
        gLoss: (event.loss as number) ?? 0,
        psnr: (event.psnr as number) ?? null,
        lossHistory: nextLoss,
        psnrHistory: nextPsnr,
      });
      break;
    }

    case "validate":
      useTrainingStore.setState({
        psnr: event.psnr as number ?? null,
        ssim: event.ssim as number ?? null,
      });
      break;

    case "done":
      useTrainingStore.setState({ status: "done" });
      break;

    case "error":
      useUiStore.getState().setLastApiError(event);
      break;
  }
}

export function connectSSE(baseUrl: string, jobId: string): EventSource {
  const es = new EventSource(`${baseUrl}/api/events?job_id=${jobId}`);
  es.onmessage = (e) => {
    try {
      const event = JSON.parse(e.data) as SSEEvent;
      dispatch(event);
    } catch {
      // ignore malformed events
    }
  };
  es.onerror = () => {
    useUiStore.getState().setServerConnected(false);
  };
  es.onopen = () => {
    useUiStore.getState().setServerConnected(true);
  };
  return es;
}