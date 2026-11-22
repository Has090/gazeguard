import pathlib
import zipfile

src = pathlib.Path("data/raw")
out = pathlib.Path("data/raw_zips")
out.mkdir(exist_ok=True)

for folder in sorted(p for p in src.iterdir() if p.is_dir()):
    with zipfile.ZipFile(out / f"{folder.name}.zip", "w", zipfile.ZIP_STORED) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(src).as_posix())
    print("packed", folder.name)