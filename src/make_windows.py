"""Stage 4: frames -> fixed-length windows, with a LEAKAGE-FREE split.

Frames of the same scenario (or subject, if params.split.subject_map is set) never appear
in more than one of train / val / test. Standardisation statistics come from train only.
Each recording (file-name prefix) inside a scenario is windowed on its own, so a window
never mixes frames from two different recordings.
"""
import sys
from collections import Counter

import numpy as np
import pandas as pd

from src.common import (FEATURES, ROOT, WINDOW_FEATURES, load_params, set_seed, write_json)


def build_windows(df, size, step, stride, max_gap, min_purity):
    """Return X (N,size,F), y (N,), meta DataFrame. Windows never cross a frame gap."""
    X, y, meta = [], [], []
    if "sequence" not in df.columns:   # older tables / unit tests: one sequence per scenario
        df = df.assign(sequence="")
    df = df[df["label"].notna()]
    for (scen, seq), g in df.groupby(["scenario", "sequence"], sort=True):
        g = g.sort_values("frame_idx").iloc[::stride]
        idx = g["frame_idx"].to_numpy()
        if len(idx) < size:
            continue
        breaks = np.where(np.diff(idx) > max_gap * stride)[0] + 1
        for seg in np.split(np.arange(len(g)), breaks):
            if len(seg) < size:
                continue
            sg = g.iloc[seg]
            feats = sg[FEATURES].to_numpy(dtype=np.float64)
            labels = sg["label"].to_numpy().astype(int)
            frames = sg["frame_idx"].to_numpy()
            for s in range(0, len(seg) - size + 1, step):
                w, lab = feats[s:s + size], labels[s:s + size]
                vals, counts = np.unique(lab, return_counts=True)
                if counts.max() / size < min_purity:
                    continue
                vel = np.vstack([np.zeros((1, 2)), np.diff(w[:, [1, 2]], axis=0)])  # d_yaw, d_pitch
                X.append(np.hstack([w, vel]))
                y.append(int(vals[counts.argmax()]))
                meta.append({"scenario": scen, "sequence": seq, "start_frame": int(frames[s]),
                             "end_frame": int(frames[s + size - 1])})
    if not X:
        return np.zeros((0, size, len(WINDOW_FEATURES))), np.zeros(0, dtype=int), pd.DataFrame(
            columns=["scenario", "sequence", "start_frame", "end_frame"])
    return np.stack(X), np.array(y), pd.DataFrame(meta)


def assign_groups(group_labels, val_frac, test_frac, seed, stratify=True):
    """group_labels: {group -> dominant label}. Returns {group -> 'train'|'val'|'test'}.

    With stratify=True each label's groups are spread over the splits, so rare classes do not
    vanish from val/test. A group is never split across partitions.
    """
    rng = np.random.RandomState(seed)
    assignment = {}
    buckets = {}
    for grp, lab in sorted(group_labels.items()):
        buckets.setdefault(lab if stratify else 0, []).append(grp)
    for lab, groups in sorted(buckets.items()):
        groups = [groups[i] for i in rng.permutation(len(groups))]
        n = len(groups)
        if n >= 3:
            n_test = max(1, round(n * test_frac))
            n_val = max(1, round(n * val_frac))
        elif n == 2:
            n_test, n_val = 1, 0
        else:
            n_test = n_val = 0
        for grp in groups[:n_test]:
            assignment[grp] = "test"
        for grp in groups[n_test:n_test + n_val]:
            assignment[grp] = "val"
        for grp in groups[n_test + n_val:]:
            assignment[grp] = "train"
    return assignment


def assert_no_leakage(meta, group_of):
    """Raise if any scenario/subject group appears in more than one split."""
    seen = {}
    for scen, split in zip(meta["scenario"], meta["split"]):
        grp = group_of(scen)
        if seen.setdefault(grp, split) != split:
            raise AssertionError(f"LEAKAGE: group '{grp}' appears in '{seen[grp]}' and '{split}'")


def main():
    params = load_params()
    set_seed(params["seed"])
    w_p, s_p = params["windows"], params["split"]
    df = pd.read_csv(ROOT / "data" / "processed" / "features.csv")
    if "sequence" not in df.columns:
        print("[make_windows] features.csv has no 'sequence' column - "
              "update src/extract_features.py and re-run the extract_features stage")
        sys.exit(1)
    X, y, meta = build_windows(df, w_p["size"], w_p["step"], w_p["frame_stride"],
                               w_p["max_frame_gap"], w_p["min_label_purity"])
    if len(X) == 0:
        print("[make_windows] no windows could be built (check window size / frame gaps)")
        sys.exit(1)

    subject_map = s_p.get("subject_map") or {}
    group_of = lambda scen: subject_map.get(scen, scen)
    meta["group"] = meta["scenario"].map(group_of)
    meta["label"] = y

    dominant = {}
    for grp, sub in meta.groupby("group"):
        dominant[grp] = Counter(sub["label"]).most_common(1)[0][0]
    assignment = assign_groups(dominant, s_p["val_frac"], s_p["test_frac"], params["seed"],
                               stratify=not subject_map)
    meta["split"] = meta["group"].map(assignment)
    assert_no_leakage(meta, group_of)

    classes = np.array(sorted(set(y.tolist())))
    y_idx = np.searchsorted(classes, y)
    tr = (meta["split"] == "train").to_numpy()
    if tr.sum() == 0:
        print("[make_windows] empty train split")
        sys.exit(1)
    mean = X[tr].reshape(-1, X.shape[2]).mean(axis=0)            # train statistics only
    std = X[tr].reshape(-1, X.shape[2]).std(axis=0) + 1e-6
    Xs = (X - mean) / std

    arrays = {"classes": classes, "mean": mean, "std": std,
              "feature_names": np.array(WINDOW_FEATURES)}
    for name in ("train", "val", "test"):
        m = (meta["split"] == name).to_numpy()
        arrays[f"X_{name}"], arrays[f"y_{name}"] = Xs[m].astype(np.float32), y_idx[m]
    out = ROOT / "data" / "processed"
    np.savez_compressed(out / "windows.npz", **arrays)
    meta.to_csv(out / "windows_meta.csv", index=False)

    report = {
        "leakage_check": "passed",
        "split_unit": "subject" if subject_map else "scenario",
        "n_windows": {k: int((meta["split"] == k).sum()) for k in ("train", "val", "test")},
        "groups": {k: sorted(meta.loc[meta["split"] == k, "group"].unique().tolist())
                   for k in ("train", "val", "test")},
        "class_counts": {k: {str(int(c)): int((arrays[f"y_{k}"] == i).sum())
                             for i, c in enumerate(classes)} for k in ("train", "val", "test")},
    }
    write_json(ROOT / "reports" / "split_report.json", report)
    print(f"[make_windows] windows={report['n_windows']} classes={classes.tolist()} "
          f"split_unit={report['split_unit']}")


if __name__ == "__main__":
    main()