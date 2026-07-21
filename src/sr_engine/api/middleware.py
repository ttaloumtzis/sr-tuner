"""ASGI middleware for request/response logging with correlation IDs."""

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request/response with timing and correlation ID.

    * Generates or propagates ``X-Request-ID`` and binds it to structlog
      context vars so all downstream logs within the same request carry it.
    * SSE endpoints (``/api/events``) are handled separately — they log
      connect/disconnect instead of duration, avoiding false ``WARN`` alerts
      from long-lived connections.
    * Requests slower than 500 ms are logged at ``WARN`` level.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(
            "X-Request-ID",
            uuid.uuid4().hex[:12],
        )
        structlog.contextvars.bind_contextvars(request_id=request_id)

        client_ip = request.client.host if request.client else "unknown"
        structlog.contextvars.bind_contextvars(client_ip=client_ip)

        method = request.method
        path = request.url.path
        query = str(request.url.query)

        log = structlog.get_logger("sr_engine.api.middleware")

        # ── SSE — long-lived connection, log connect/disconnect ──────────
        if path == "/api/events":
            log.info(
                "sse_connected",
                method=method,
                path=path,
                query=query,
            )
            try:
                response = await call_next(request)
            finally:
                log.info(
                    "sse_disconnected",
                    method=method,
                    path=path,
                    query=query,
                )
            return response

        # ── Normal request — measure duration ────────────────────────────
        structlog.contextvars.bind_contextvars(
            method=method,
            path=path,
            query=query,
        )

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled_exception")
            raise

        duration_ms = (time.perf_counter() - start) * 1000.0
        status = response.status_code

        log_method = log.warning if duration_ms > 500 else log.info
        log_method(
            "request_finished",
            status=status,
            duration_ms=round(duration_ms, 1),
        )

        return response