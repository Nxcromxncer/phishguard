"""Small in-memory sliding-window rate limiter.

Added beyond the required file list because API-abuse protection is an explicit
security requirement and FastAPI has no built-in limiter.

Limitations (documented, not hidden):
* Per process, in memory: with several workers each has its own counters.
* Keyed by client address. Behind a reverse proxy every request shares the
  proxy's address unless you configure trusted forwarded headers.
* Intended for a localhost developer tool, not as DDoS protection.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Callable


class SlidingWindowRateLimiter:
    """Allow at most ``max_requests`` per ``window_seconds`` for each key."""

    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 60.0,
        max_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        self._max = max_requests
        self._window = window_seconds
        self._max_clients = max_clients
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Record a request. Returns ``(allowed, retry_after_seconds)``."""
        now = self._clock()
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                if len(self._hits) >= self._max_clients:
                    self._evict(now)
                hits = deque()
                self._hits[key] = hits

            cutoff = now - self._window
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self._max:
                retry_after = math.ceil(hits[0] + self._window - now)
                return False, max(1, retry_after)

            hits.append(now)
            return True, 0

    def _evict(self, now: float) -> None:
        """Drop expired clients; if still full, drop the oldest-tracked one."""
        cutoff = now - self._window
        for key in [k for k, q in self._hits.items() if not q or q[-1] <= cutoff]:
            del self._hits[key]
        if len(self._hits) >= self._max_clients:
            self._hits.pop(next(iter(self._hits)))
