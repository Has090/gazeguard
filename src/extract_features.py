"""Stage 2: raw images -> one row of features + label per frame (data/processed/features.csv)."""
import sys

import cv2
import pandas as pd

from src.common import (FEATURES, ROOT, largest_box_class, list_images, list_scenarios,
                        load_params, parse_frame_idx, read_yolo_annotation)
from src.features import EMPTY, landmarks_to_features


def frame_label(img_path, scenario, params):
    labels_p = params["labels"]
    if labels_p["source"] == "folder_map":
        return labels_p["folder_map"].get(scenario)
    ann = img_path.with_suffix(".txt")
    if not ann.exists():
        return None
    boxes, _ = read_yolo_annotation(ann)
    return largest_box_class(boxes)


def main():
    import mediapipe as mp  # imported here so unit tests do not need it

    params = load_params()
    raw_dir = ROOT / params["data"]["raw_dir"]
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=params["extract"]["max_num_faces"],
        refine_landmarks=True,   # adds the 10 iris landmarks (478 total)
        min_detection_confidence=params["extract"]["min_detection_confidence"],
    )
    rows = []
    for scen in list_scenarios(raw_dir):
        images = list_images(scen, params["data"]["image_exts"])
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                feats = landmarks_to_features([(p.x, p.y) for p in lm], w, h)
            else:
                feats = dict(EMPTY)
            rows.append({
                "scenario": scen.name,
                "frame_idx": parse_frame_idx(img_path.stem),
                "image": f"{scen.name}/{img_path.name}",
                "label": frame_label(img_path, scen.name, params),
                **feats,
            })
        print(f"[extract_features] {scen.name}: {len(images)} frames", flush=True)

    df = pd.DataFrame(rows).sort_values(["scenario", "frame_idx"]).reset_index(drop=True)
    out = ROOT / "data" / "processed" / "features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[extract_features] wrote {len(df)} rows, face found in "
          f"{df['face_present'].mean():.1%} of frames -> {out}")
    if df.empty:
        sys.exit(1)


if __name__ == "__main__":
    main()
