#!/usr/bin/env python3
"""Run from project root: ``python make_hazard_map.py im0.png im1.png -o hazard_map.png``"""
from __future__ import annotations

import sys

from hazard_cv.pair_stereo import main

if __name__ == "__main__":
    raise SystemExit(main())
