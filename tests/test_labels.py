import json
from pathlib import Path

from hazard_cv.geometry.hazards import CornerHazard
from hazard_cv.geometry.labels import associate_predictions, evaluate_label_file, load_corner_labels


def test_load_corner_labels(tmp_path: Path) -> None:
    p = tmp_path / "l.json"
    p.write_text(
        json.dumps(
            {
                "labels": [
                    {"image": "a", "u": 10, "v": 10, "radius": 5, "level": "high"},
                ]
            }
        )
    )
    labs = load_corner_labels(p)
    assert len(labs) == 1 and labs[0].level == "high"


def test_associate_predictions() -> None:
    from hazard_cv.geometry.labels import CornerLabel

    labels = [
        CornerLabel("im0", 50, 50, 10, "high"),
        CornerLabel("im0", 150, 150, 10, "low"),
    ]
    preds = [
        CornerHazard(52, 51, 40, 1.0, "high"),
        CornerHazard(300, 300, 90, 1.0, "low"),
    ]
    tp, fp, fn = associate_predictions(preds, labels, "im0")
    assert tp >= 1
    assert fp >= 1
    assert fn >= 0


def test_evaluate_label_file(tmp_path: Path) -> None:
    labels = tmp_path / "corner_labels.json"
    labels.write_text(
        json.dumps(
            {
                "labels": [
                    {"image": "x", "u": 10, "v": 10, "radius": 20, "level": "medium"},
                ]
            }
        )
    )
    preds = {"x": [CornerHazard(12, 11, 70, 1.0, "medium")]}
    rep = evaluate_label_file(preds, labels)
    assert rep["tp"] == 1 and rep["fn"] == 0
