"""Optional YOLO / Outdoor Hazard integration — stub unless ultralytics is installed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SemanticDetection:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class OutdoorSemanticDetector:
    """Placeholder for YOLO on COCO-style or custom outdoor hazard weights.

    Install optional dependency: ``pip install ultralytics`` then replace
    ``predict`` with a real ``YOLO(weights).predict(...)`` call mapping class ids
    to hazard names (e.g. person, car, bench) for fusion with geometric corners.
    """

    def __init__(self, weights: Optional[str] = None) -> None:
        self.weights = weights
        self._model = None
        try:
            from ultralytics import YOLO  # type: ignore

            if weights:
                self._model = YOLO(weights)
        except Exception:
            self._model = None

    def available(self) -> bool:
        return self._model is not None

    def predict(self, bgr: np.ndarray) -> List[SemanticDetection]:
        if self._model is None:
            return []
        results = self._model.predict(bgr, verbose=False)
        out: List[SemanticDetection] = []
        for r in results:
            if r.boxes is None:
                continue
            names = r.names
            for b in r.boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                label = names.get(cls_id, str(cls_id))
                out.append(
                    SemanticDetection(
                        label=label,
                        confidence=conf,
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                    )
                )
        return out
