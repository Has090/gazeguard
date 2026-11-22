"""Make a SMALL fake dataset (same folder/annotation layout as the real one) from one face photo.

Only for testing the pipeline plumbing - the labels are random, so accuracy means nothing.

    python scripts/make_demo_data.py --sample-image path/to/face.jpg --out data/raw
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-image", required=True)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--scenarios", type=int, default=12)
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    base = Image.open(a.sample_image).convert("RGB")
    for s in range(1, a.scenarios + 1):
        folder = Path(a.out) / f"S{s}"
        folder.mkdir(parents=True, exist_ok=True)
        head_cls = (s - 1) % 3 + 1                      # 3 fake "classes": ids 1..3
        for f in range(a.frames):
            im = ImageEnhance.Brightness(base).enhance(rnd.uniform(0.85, 1.15))
            im = im.rotate(rnd.uniform(-3, 3), resample=Image.BILINEAR)
            stem = f"S{s}_{f + 1:08d}"
            im.save(folder / f"{stem}.jpg", quality=92)
            (folder / f"{stem}.txt").write_text(
                f"5 0.435 0.436 0.059 0.058\n5 0.523 0.443 0.058 0.061\n"
                f"{head_cls} 0.481 0.525 0.162 0.342\n11 0.477 0.621 0.082 0.086\n")
    print(f"wrote {a.scenarios} scenarios x {a.frames} frames to {a.out}")


if __name__ == "__main__":
    main()
