"""Client-side rate limiting over several sliding windows at once."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence


class SlidingWindowLimiter:
    """Blocks until one more call fits under every (max_calls, window_seconds) limit.

    Upstox enforces 50 req/s, 500 req/min and 2000 req per 30 min per user for standard
    APIs. The defaults used by the Upstox provider stay comfortably below all three.
    """

    def __init__(
        self,
        limits: Sequence[tuple[int, float]],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not limits:
            raise ValueError("at least one limit is required")
        self._limits = sorted(((int(n), float(w)) for n, w in limits), key=lambda x: x[1])
        if any(n < 1 or w <= 0 for n, w in self._limits):
            raise ValueError("limits must have max_calls >= 1 and window > 0")
        self._max_window = self._limits[-1][1]
        self._calls: deque[float] = deque()
        self._clock = clock
        self._sleep = sleep

    def acquire(self) -> float:
        """Reserve one call slot; returns the total seconds spent waiting."""
        waited = 0.0
        while True:
            now = self._clock()
            while self._calls and now - self._calls[0] >= self._max_window:
                self._calls.popleft()
            wait = 0.0
            for max_calls, window in self._limits:
                recent = [t for t in self._calls if now - t < window]
                if len(recent) >= max_calls:
                    # The call that must age out before another one fits.
                    blocker = recent[-max_calls]
                    wait = max(wait, window - (now - blocker))
            if wait <= 0:
                self._calls.append(now)
                return waited
            self._sleep(wait)
            waited += wait
