"""Stage 1: check the raw dataset BEFORE any processing. Exits with code 1 on failure."""
import sys
from pathlib import Path

from PIL import Image

from src.common import (ROOT, list_images, list_scenarios, load_params,
                        parse_frame_idx, read_yolo_annotation, sequence_id, write_json)


def validate_raw(raw_dir, params):
    data_p, val_p = params["data"], params["validation"]
    scenarios = list_scenarios(raw_dir)
    errors = []
    n_images = n_missing = n_corrupt = n_bad_ann = n_bad_frame = n_dup = 0
    classes_seen = set()

    for scen in scenarios:
        images = list_images(scen, data_p["image_exts"])
        seen_idx = set()  # (recording prefix, frame number): each recording is numbered separately
        for img in images:
            n_images += 1
            idx = parse_frame_idx(img.stem)
            key = (sequence_id(img.stem), idx)
            if idx is None:
                n_bad_frame += 1
                errors.append(f"{scen.name}/{img.name}: file name does not end with a frame number")
            elif key in seen_idx:
                n_dup += 1
                errors.append(f"{scen.name}/{img.name}: duplicate frame number {idx} in '{key[0]}'")
            else:
                seen_idx.add(key)
            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception as exc:  # corrupted / truncated image
                n_corrupt += 1
                errors.append(f"{scen.name}/{img.name}: unreadable image ({exc.__class__.__name__})")
            ann = img.with_suffix(".txt")
            if not ann.exists():
                n_missing += 1
                continue
            boxes, ann_errors = read_yolo_annotation(ann)
            if ann_errors:
                n_bad_ann += 1
                errors.extend(ann_errors[:2])
            for cls, x, y, w, h in boxes:
                classes_seen.add(cls)
                if not (0 <= cls < data_p["num_classes"]):
                    n_bad_ann += 1
                    errors.append(f"{ann.name}: class id {cls} outside 0..{data_p['num_classes'] - 1}")
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    n_bad_ann += 1
                    errors.append(f"{ann.name}: box coordinates outside 0..1")

    missing_rate = n_missing / n_images if n_images else 1.0
    corrupt_rate = n_corrupt / n_images if n_images else 1.0
    failures = []
    if len(scenarios) < val_p["min_scenarios"]:
        failures.append(f"only {len(scenarios)} scenario folders (need >= {val_p['min_scenarios']})")
    if n_images == 0:
        failures.append("no images found")
    if missing_rate > val_p["max_missing_annotation_rate"]:
        failures.append(f"missing-annotation rate {missing_rate:.3f} too high")
    if corrupt_rate > val_p["max_corrupt_image_rate"]:
        failures.append(f"corrupt-image rate {corrupt_rate:.3f} too high")
    if n_bad_ann:
        failures.append(f"{n_bad_ann} malformed annotation entries")
    if n_bad_frame or n_dup:
        failures.append("frame numbering problems (unparsable or duplicate frame numbers)")

    return {
        "passed": not failures,
        "failures": failures,
        "n_scenarios": len(scenarios),
        "n_images": n_images,
        "n_missing_annotation": n_missing,
        "missing_annotation_rate": round(missing_rate, 4),
        "n_corrupt_images": n_corrupt,
        "corrupt_image_rate": round(corrupt_rate, 4),
        "n_bad_annotation_entries": n_bad_ann,
        "n_unparsable_frame_numbers": n_bad_frame,
        "n_duplicate_frame_numbers": n_dup,
        "class_ids_seen": sorted(classes_seen),
        "first_errors": errors[:30],
    }


def main():
    params = load_params()
    report = validate_raw(ROOT / params["data"]["raw_dir"], params)
    write_json(ROOT / "reports" / "raw_validation.json", report)
    print(f"[validate_raw] scenarios={report['n_scenarios']} images={report['n_images']} "
          f"passed={report['passed']}")
    if not report["passed"]:
        for f in report["failures"]:
            print("  FAIL:", f)
        for e in report["first_errors"][:10]:
            print("   ", e)
        sys.exit(1)


if __name__ == "__main__":
    main()