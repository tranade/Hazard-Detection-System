"""EMA smoothing for corner (u,v,score) tracks and simple FPS bench."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from hazard_cv.geometry.hazards import CornerHazard


@dataclass
class _Track:
    u: float
    v: float
    score: float
    level: str
    hits: int = 1


class TemporalCornerFilter:
    """Match corners frame-to-frame by nearest-neighbor in image space; EMA smooth."""

    def __init__(self, alpha: float = 0.35, match_dist: float = 25.0) -> None:
        self.alpha = alpha
        self.match_dist = match_dist
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

    def update(self, corners: List[CornerHazard]) -> List[Tuple[int, CornerHazard]]:
        """Return list of (track_id, smoothed_corner)."""
        if not corners:
            return []
        used: set[int] = set()
        out: List[Tuple[int, CornerHazard]] = []
        for c in corners:
            best_id = None
            best_d = self.match_dist
            for tid, tr in self._tracks.items():
                if tid in used:
                    continue
                d = float(np.hypot(tr.u - c.u, tr.v - c.v))
                if d < best_d:
                    best_d = d
                    best_id = tid
            if best_id is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _Track(u=c.u, v=c.v, score=c.score, level=c.level)
                used.add(tid)
                out.append(
                    (
                        tid,
                        CornerHazard(
                            u=c.u, v=c.v, angle_deg=c.angle_deg, score=c.score, level=c.level
                        ),
                    )
                )
                continue
            tr = self._tracks[best_id]
            a = self.alpha
            tr.u = (1 - a) * tr.u + a * c.u
            tr.v = (1 - a) * tr.v + a * c.v
            tr.score = (1 - a) * tr.score + a * c.score
            tr.level = c.level if c.score > tr.score else tr.level
            tr.hits += 1
            used.add(best_id)
            out.append(
                (
                    best_id,
                    CornerHazard(
                        u=tr.u,
                        v=tr.v,
                        angle_deg=c.angle_deg,
                        score=tr.score,
                        level=tr.level,
                    ),
                )
            )
        return out


def bench_fps(frames: Iterable[np.ndarray], fn: Callable[[np.ndarray], object]) -> Tuple[float, float]:
    """Run fn on each frame; return (mean_fps, mean_ms_per_frame)."""
    times: List[float] = []
    n = 0
    for fr in frames:
        t0 = time.perf_counter()
        fn(fr)
        times.append(time.perf_counter() - t0)
        n += 1
    if not times:
        return 0.0, 0.0
    mean_s = float(np.mean(times))
    return (1.0 / mean_s) if mean_s > 0 else 0.0, mean_s * 1000.0
