"""Small JSON label files for corner hazard evaluation (precision/recall/F1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from hazard_cv.geometry.hazards import CornerHazard, match_label_level


@dataclass
class CornerLabel:
    image_stem: str
    u: float
    v: float
    radius: float
    level: str


def load_corner_labels(path: Path) -> List[CornerLabel]:
    data = json.loads(Path(path).read_text())
    items = data.get("labels", data)
    out: List[CornerLabel] = []
    for row in items:
        out.append(
            CornerLabel(
                image_stem=str(row["image"]),
                u=float(row["u"]),
                v=float(row["v"]),
                radius=float(row.get("radius", 18.0)),
                level=str(row["level"]),
            )
        )
    return out


def associate_predictions(
    preds: List[CornerHazard], labels: List[CornerLabel], image_stem: str
) -> Tuple[int, int, int]:
    """Greedy match preds to labels within radius; returns (tp, fp, fn) for level agreement."""
    labs = [L for L in labels if L.image_stem == image_stem]
    unmatched = set(range(len(labs)))
    tp = fp = 0
    for p in preds:
        best = None
        best_d = 1e9
        for i in list(unmatched):
            L = labs[i]
            d = float(np.hypot(p.u - L.u, p.v - L.v))
            if d <= L.radius and d < best_d:
                best_d = d
                best = i
        if best is None:
            fp += 1
            continue
        unmatched.remove(best)
        L = labs[best]
        if match_label_level(p.level, L.level):
            tp += 1
        else:
            fp += 1
    fn = len(unmatched)
    return tp, fp, fn


def prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


def evaluate_label_file(
    preds_by_image: Dict[str, List[CornerHazard]], labels_path: Path
) -> Dict[str, Any]:
    labels = load_corner_labels(labels_path)
    stems = {L.image_stem for L in labels}
    total_tp = total_fp = total_fn = 0
    for stem in stems:
        preds = preds_by_image.get(stem, [])
        tp, fp, fn = associate_predictions(preds, labels, stem)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    p, r, f1 = prf1(total_tp, total_fp, total_fn)
    return {"tp": total_tp, "fp": total_fp, "fn": total_fn, "precision": p, "recall": r, "f1": f1}
