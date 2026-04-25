"""Prioritized, throttled TTS for hazard phrases."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import pyttsx3

from hazard_cv.geometry.hazards import CornerHazard


class HazardNarrator:
    def __init__(
        self,
        min_interval_s: float = 4.0,
        min_depth_change_m: float = 0.25,
        rate: int = 175,
    ) -> None:
        self.min_interval_s = min_interval_s
        self.min_depth_change_m = min_depth_change_m
        self._last_speak = 0.0
        self._last_phrase = ""
        self._last_depth: Optional[float] = None
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)

    def announce_hazards(
        self,
        items: List[tuple[CornerHazard, Optional[float]]],
        image_width: float = 640.0,
        speak: bool = True,
        clock: Optional[float] = None,
    ) -> Tuple[float, str]:
        """Return (latency_s, phrase). If speak is False, only compute phrase (for tests)."""
        clock = clock or time.perf_counter()
        if not items:
            return 0.0, ""
        ranked = []
        for c, z in items:
            pr = {"high": 3.0, "medium": 2.0, "low": 1.0}[c.level]
            dist_term = 1.0 / (0.5 + (z or 2.0))
            ranked.append((pr + dist_term, c, z))
        ranked.sort(key=lambda x: -x[0])
        _, c, z = ranked[0]
        side = "center"
        if c.u < image_width * 0.35:
            side = "left"
        elif c.u > image_width * 0.65:
            side = "right"
        ztxt = f", about {z:.1f} meters ahead" if z is not None else ""
        phrase = f"{c.level} hazard sharp corner near {side}{ztxt}."
        now = time.perf_counter()
        if self.min_interval_s > 0 and (now - self._last_speak) < self.min_interval_s:
            if phrase == self._last_phrase:
                if z is None or self._last_depth is None:
                    return now - clock, phrase
                if abs(z - self._last_depth) < self.min_depth_change_m:
                    return now - clock, phrase
        if speak:
            self._engine.say(phrase)
            self._engine.runAndWait()
        latency = time.perf_counter() - clock
        self._last_speak = now
        self._last_phrase = phrase
        self._last_depth = z
        return latency, phrase
