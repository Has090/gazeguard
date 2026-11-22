"""Unit tests: split leakage, annotation validation and feature maths (no MediaPipe needed)."""
import numpy as np
import pandas as pd
import pytest

from src.features import EMPTY, eye_aspect_ratio, landmarks_to_features
from src.make_windows import assert_no_leakage, assign_groups, build_windows
from src.validate_processed import validate_features
from src.validate_raw import validate_raw
from src.common import FEATURES, load_params, read_yolo_annotation, largest_box_class


def _frames(n_scen=9, n_frames=60):
    rows = []
    for s in range(n_scen):
        for f in range(n_frames):
            rows.append({"scenario": f"S{s}", "frame_idx": f, "image": f"S{s}/S{s}_{f:08d}.jpg",
                         "label": float(s % 3), **{c: float(np.random.rand()) for c in FEATURES}})
            rows[-1]["face_present"] = 1
    return pd.DataFrame(rows)


def test_split_is_group_disjoint_and_deterministic():
    df = _frames()
    X, y, meta = build_windows(df, size=30, step=10, stride=1, max_gap=5, min_purity=0.6)
    meta["label"] = y
    dominant = meta.groupby("scenario")["label"].agg(lambda s: s.mode()[0]).to_dict()
    a1 = assign_groups(dominant, 0.15, 0.15, seed=1)
    a2 = assign_groups(dominant, 0.15, 0.15, seed=1)
    assert a1 == a2
    meta["split"] = meta["scenario"].map(a1)
    assert_no_leakage(meta, lambda s: s)            # must not raise
    assert {"train", "val", "test"} <= set(a1.values())


def test_leakage_check_catches_overlap():
    meta = pd.DataFrame({"scenario": ["S1", "S1"], "split": ["train", "test"]})
    with pytest.raises(AssertionError):
        assert_no_leakage(meta, lambda s: s)


def test_windows_do_not_cross_frame_gaps():
    df = _frames(n_scen=1, n_frames=70)
    df = df[~df["frame_idx"].between(30, 49)]       # 20-frame hole
    X, y, meta = build_windows(df, size=30, step=10, stride=1, max_gap=5, min_purity=0.6)
    assert len(X) > 0
    for _, r in meta.iterrows():
        assert not (r.start_frame < 30 and r.end_frame >= 50)


def test_annotation_parsing_and_largest_box(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("5 0.4 0.4 0.05 0.05\n3 0.48 0.52 0.16 0.34\n11 0.47 0.62 0.08 0.08\n")
    boxes, errors = read_yolo_annotation(f)
    assert errors == [] and largest_box_class(boxes) == 3
    f.write_text("5 0.4 0.4\n")
    assert read_yolo_annotation(f)[1]


def test_validate_raw_flags_bad_data(tmp_path):
    from PIL import Image
    params = load_params()
    for s in ("A", "B", "C"):
        (tmp_path / s).mkdir()
        for i in range(3):
            Image.new("RGB", (32, 32)).save(tmp_path / s / f"{s}_{i:08d}.jpg")
            (tmp_path / s / f"{s}_{i:08d}.txt").write_text("3 0.5 0.5 0.2 0.2\n")
    assert validate_raw(tmp_path, params)["passed"]
    (tmp_path / "A" / "A_00000001.txt").write_text("99 2.0 0.5 0.2 0.2\n")     # bad class + coords
    (tmp_path / "B" / "B_00000002.jpg").write_bytes(b"not an image")          # corrupt image
    report = validate_raw(tmp_path, params)
    assert not report["passed"] and report["n_corrupt_images"] == 1


def test_validate_processed_flags_missing_values_and_bad_ranges():
    params = load_params()
    df = _frames()
    assert validate_features(df, params)["passed"]
    bad = df.copy()
    bad.loc[0, "yaw"] = 400.0
    assert not validate_features(bad, params)["passed"]
    nolabel = df.copy()
    nolabel["label"] = np.nan
    assert not validate_features(nolabel, params)["passed"]


def test_feature_maths_on_synthetic_face():
    pts = np.full((478, 2), 0.5)
    # eye: corners 0.1 apart, lids 0.02 apart -> EAR = 0.02+0.02 / (2*0.1) = 0.2
    pts[33], pts[133] = (0.40, 0.40), (0.50, 0.40)
    pts[160], pts[144] = (0.43, 0.39), (0.43, 0.41)
    pts[158], pts[153] = (0.47, 0.39), (0.47, 0.41)
    px = pts * np.array([100.0, 100.0])
    r = (np.linalg.norm(px[160] - px[144]) + np.linalg.norm(px[158] - px[153])) / (
        2 * np.linalg.norm(px[33] - px[133]))
    assert abs(r - 0.2) < 1e-6
    assert set(EMPTY) == set(FEATURES)
    assert landmarks_to_features(np.zeros((10, 2)), 100, 100)["face_present"] == 0


def test_single_corrupt_image_stops_pipeline(tmp_path):
    from PIL import Image
    params = load_params()
    for s in ("A", "B", "C"):
        (tmp_path / s).mkdir()
        for i in range(40):
            Image.new("RGB", (16, 16)).save(tmp_path / s / f"{s}_{i:08d}.jpg")
            (tmp_path / s / f"{s}_{i:08d}.txt").write_text("3 0.5 0.5 0.2 0.2\n")
    assert validate_raw(tmp_path, params)["passed"]
    (tmp_path / "A" / "A_00000005.jpg").write_bytes(b"garbage")    # 1 bad file out of 120
    report = validate_raw(tmp_path, params)
    assert not report["passed"] and report["n_corrupt_images"] == 1
