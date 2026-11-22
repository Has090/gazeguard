"""Stage 3: schema + quality checks on the feature table BEFORE training. Exits 1 on failure."""
import sys

import pandas as pd
import pandera.pandas as pa

from src.common import FEATURES, ROOT, load_params, write_json


def build_schema():
    num = lambda lo, hi: pa.Column(float, pa.Check.in_range(lo, hi), nullable=False, coerce=True)
    return pa.DataFrameSchema(
        {
            "scenario": pa.Column(str, nullable=False),
            "frame_idx": pa.Column(int, pa.Check.ge(0), nullable=False, coerce=True),
            "image": pa.Column(str, nullable=False),
            "label": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
            "face_present": pa.Column(int, pa.Check.isin([0, 1]), nullable=False, coerce=True),
            "yaw": num(-90, 90), "pitch": num(-90, 90), "roll": num(-90, 90),
            "nose_x": num(-0.5, 1.5), "nose_y": num(-0.5, 1.5),
            "iris_l_x": num(-3, 3), "iris_l_y": num(-3, 3),
            "iris_r_x": num(-3, 3), "iris_r_y": num(-3, 3),
            "ear": num(0, 2), "mouth_open": num(0, 5),
        },
        strict=False,
        unique=["scenario", "frame_idx"],
    )


def validate_features(df, params):
    val_p = params["validation"]
    failures = []
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        failures.append(f"missing feature columns: {missing_cols}")
    else:
        try:
            build_schema().validate(df, lazy=True)
        except pa.errors.SchemaErrors as exc:
            failures.append(f"schema violations: {len(exc.failure_cases)} cases")
            failures.extend(exc.failure_cases.head(5).astype(str).agg(" | ".join, axis=1).tolist())
    face_rate = float(df["face_present"].mean()) if len(df) else 0.0
    label_rate = float(df["label"].notna().mean()) if len(df) else 0.0
    if face_rate < val_p["min_face_detect_rate"]:
        failures.append(f"face detected in only {face_rate:.1%} of frames")
    if label_rate < val_p["min_labeled_rate"]:
        failures.append(f"only {label_rate:.1%} of frames have a label")
    return {
        "passed": not failures,
        "failures": failures,
        "n_rows": int(len(df)),
        "n_scenarios": int(df["scenario"].nunique()) if len(df) else 0,
        "face_detect_rate": round(face_rate, 4),
        "labeled_rate": round(label_rate, 4),
        "n_missing_values": int(df[FEATURES].isna().sum().sum()) if not missing_cols else -1,
        "label_counts": df["label"].value_counts().sort_index().to_dict() if len(df) else {},
    }


def main():
    params = load_params()
    df = pd.read_csv(ROOT / "data" / "processed" / "features.csv")
    report = validate_features(df, params)
    report["label_counts"] = {str(int(k)): int(v) for k, v in report["label_counts"].items()}
    write_json(ROOT / "reports" / "processed_validation.json", report)
    print(f"[validate_processed] rows={report['n_rows']} passed={report['passed']}")
    if not report["passed"]:
        for f in report["failures"]:
            print("  FAIL:", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
