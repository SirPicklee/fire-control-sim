"""Sensor timing: sample clocks, processing latency and a measurement buffer.

Every measurement carries its time of validity (ICD). It reaches the
fusion processor at arrival = validity + latency. The buffer releases
measurements that have arrived by `now`, sorted by validity, and flags
out-of-sequence measurements (OOSM): those older than one already released.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorClock:
    rate: float                        # Hz
    phase: float = 0.0                 # s, first sample time
    jitter: float = 0.0                # s, random timing jitter (std)

    def times(self, t_end: float, rng: np.random.Generator | None = None) -> np.ndarray:
        n = int(np.floor((t_end - self.phase) * self.rate + 1e-9)) + 1
        t = self.phase + np.arange(max(n, 0)) / self.rate
        if self.jitter > 0.0:
            if rng is None:
                raise ValueError("jitter needs an rng")
            t = np.sort(t + rng.normal(0.0, self.jitter, t.size))
        return t[(t >= 0.0) & (t <= t_end)]


@dataclass(frozen=True)
class Latency:
    mean: float = 0.0                  # s
    std: float = 0.0                   # s

    def arrival(self, t_valid: float, rng: np.random.Generator | None = None) -> float:
        if self.std == 0.0:
            return t_valid + self.mean
        if rng is None:
            raise ValueError("random latency needs an rng")
        return t_valid + max(0.0, rng.normal(self.mean, self.std))


class MeasurementBuffer:
    """Holds measurements until they arrive; releases them in validity order."""

    def __init__(self):
        self._heap = []
        self._count = itertools.count()
        self.last_valid = -np.inf
        self.n_oosm = 0

    def push(self, arrival: float, meas) -> None:
        heapq.heappush(self._heap, (arrival, next(self._count), meas))

    def pop_ready(self, now: float):
        """List of (measurement, is_oosm) that arrived by `now`, sorted by m.t."""
        ready = []
        while self._heap and self._heap[0][0] <= now:
            ready.append(heapq.heappop(self._heap)[2])
        ready.sort(key=lambda m: m.t)
        out = []
        for m in ready:
            oosm = m.t < self.last_valid
            if oosm:
                self.n_oosm += 1
            else:
                self.last_valid = m.t
            out.append((m, oosm))
        return out

    def __len__(self) -> int:
        return len(self._heap)