import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO depth params from calibration .npz to JSON."
    )
    parser.add_argument(
        "--calibration",
        default="camera_calibration.npz",
        help="Input calibration .npz from chessboard_calibrate.py",
    )
    parser.add_argument(
        "--output",
        default="depth_params.json",
        help="Output JSON file with focal_px, principal_x, baseline_m",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cal_path = Path(args.calibration).expanduser().resolve()
    if not cal_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {cal_path}")

    with np.load(str(cal_path)) as data:
        required = ("K_left", "T")
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(
                f"Calibration file missing keys: {missing}. Expected K_left and T."
            )

        k_left = data["K_left"].astype(np.float64)
        t = data["T"].astype(np.float64).reshape(-1)

    payload = {
        "source_calibration": str(cal_path),
        "focal_px": float(k_left[0, 0]),
        "principal_x": float(k_left[0, 2]),
        "baseline_m": float(np.linalg.norm(t)),
    }

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote depth params to: {out_path}")
    print(
        f"focal_px={payload['focal_px']:.3f}, "
        f"principal_x={payload['principal_x']:.3f}, "
        f"baseline_m={payload['baseline_m']:.6f}"
    )


if __name__ == "__main__":
    main()
