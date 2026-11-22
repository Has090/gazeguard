import pathlib
import zipfile

zips = sorted(pathlib.Path("data/raw_zips").glob("*.zip"))
assert zips, "No zips found in data/raw_zips"
for z in zips:
    with zipfile.ZipFile(z) as f:
        f.extractall("data/raw")