"""Shared helpers: params, seeding, annotation parsing, file discovery."""
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Per-frame features written by extract_features.py (order matters).
FEATURES = [
    "face_present", "yaw", "pitch", "roll", "nose_x", "nose_y",
    "iris_l_x", "iris_l_y", "iris_r_x", "iris_r_y", "ear", "mouth_open",
]
# Extra channels added per window in make_windows.py (frame-to-frame change).
VELOCITY_FEATURES = ["d_yaw", "d_pitch"]
WINDOW_FEATURES = FEATURES + VELOCITY_FEATURES

_FRAME_RE = re.compile(r"(\d+)$")
_SEQ_RE = re.compile(r"^(.*?)\d+$")


def load_params(path=None):
    with open(path or ROOT / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def parse_frame_idx(stem: str):
    """'S1_00000264' -> 264 (trailing digits of the file name), else None."""
    m = _FRAME_RE.search(stem)
    return int(m.group(1)) if m else None


def sequence_id(stem: str) -> str:
    """'S2C_00000264' -> 'S2C_'. The prefix identifies the recording a frame belongs to;
    every recording inside a scenario folder is numbered from 0 on its own."""
    m = _SEQ_RE.match(stem)
    return m.group(1) if m else stem


def read_yolo_annotation(path):
    """Parse a YOLO-format .txt file: 'class x_center y_center width height' per line.

    Returns (boxes, errors) where boxes is a list of (cls, x, y, w, h).
    """
    boxes, errors = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{path.name}:{n} expected 5 fields, got {len(parts)}")
                continue
            try:
                cls = int(parts[0])
                x, y, w, h = (float(p) for p in parts[1:])
            except ValueError:
                errors.append(f"{path.name}:{n} non-numeric value")
                continue
            boxes.append((cls, x, y, w, h))
    return boxes, errors


def largest_box_class(boxes):
    """Class id of the box with the largest area (the head box in this dataset)."""
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[3] * b[4])[0]


def list_scenarios(raw_dir):
    """Sub-folders of raw_dir (each folder = one scenario)."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return []
    return sorted(p for p in raw_dir.iterdir() if p.is_dir())


def list_images(folder, exts):
    exts = {e.lower() for e in exts}
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in exts)