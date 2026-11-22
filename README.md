# GazeGuard – Milestone 1: reproducible ML pipeline

Head-pose / gaze behaviour flagging for exams. This milestone proves the core ML pipeline is
**solvable, leak-free and reproducible** (no frontend, no cloud autoscaling yet).

```
data/raw (DVC)  ->  validate_raw  ->  extract_features (MediaPipe)  ->  validate_processed
                ->  make_windows (leakage-free split)  ->  train (baselines + 1D-CNN+GRU, MLflow)
```

| Requirement | Where |
|---|---|
| Versioned data (DVC) | `data/raw.dvc`, `dvc.yaml`, `dvc.lock`, DagsHub remote |
| Schema / quality checks | `src/validate_raw.py`, `src/validate_processed.py` (pandera), `tests/` |
| Leakage-free split | `src/make_windows.py` – whole scenario (or subject) goes to ONE of train/val/test; stats fitted on train only; `reports/split_report.json` |
| Baselines first | `majority_class`, `random_forest` are trained/logged before the primary `cnn_gru` |
| Experiment tracking + registry | `src/train.py` -> MLflow on DagsHub; best model registered as `gazeguard-temporal-classifier` |
| One-command reproduce | `docker run --rm --env-file .env gazeguard-train` (or `dvc repro --force`) |

## Expected dataset layout

```
data/raw/
  S1/  S1_00000001.jpg  S1_00000001.txt  S1_00000002.jpg ...
  S2/  ...
  (60 scenario folders)
```
Each `.txt` is YOLO format: `class x_center y_center width height` (0–1). The file name must end
with the frame number (it is used to order frames in time).

**Label rule (check this!)** – the 13 class ids have no names in the download. By default the
frame label is the class id of the **largest box** in its `.txt` (the head box). If your folders
are named by direction instead, set `labels.source: folder_map` and fill `labels.folder_map`
in `params.yaml`.

---------------------------------------------------------------------------------------------
## Windows setup (PowerShell)

Install first: **Python 3.11** (3.9–3.12 work; 3.13 does NOT work with the pinned MediaPipe),
**Git**, **Docker Desktop**.

```powershell
cd gazeguard-mlops
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -q                 # 8 tests should pass
```

## 1. GitHub + DagsHub

1. Create an empty GitHub repo `gazeguard-mlops` (no README).
2. On DagsHub: **New Repository -> Connect a repository -> GitHub** and pick it. Your repo is then
   `https://dagshub.com/daud77245/gazeguard-mlops` (menu names may differ slightly).
3. DagsHub **Settings -> Tokens**: create an access token. Keep it private (never commit it).

```powershell
git init
git add .
git commit -m "Milestone 1 pipeline"
git branch -M main
git remote add origin https://github.com/<your-github-user>/gazeguard-mlops.git
git push -u origin main
```

## 2. DVC + DagsHub storage

```powershell
dvc init
dvc remote add -d origin https://dagshub.com/daud77245/gazeguard-mlops.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user daud77245
dvc remote modify origin --local password <YOUR_DAGSHUB_TOKEN>
```
(`--local` keeps the token in `.dvc/config.local`, which Git ignores. The remote URL itself
goes in `.dvc/config`, which you commit so the teacher can pull.)

## 3. Add the dataset and version it

Copy the 60 scenario folders into `data\raw\`, then:

```powershell
Get-ChildItem data\raw | Select-Object -First 5        # sanity check the layout
dvc add data/raw
git add data/raw.dvc data/.gitignore .dvc .dvcignore
git commit -m "Track raw dataset v1 with DVC"
dvc push
git push
```

## 4. Run the pipeline and log to DagsHub MLflow

```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/daud77245/gazeguard-mlops.mlflow"
$env:MLFLOW_TRACKING_USERNAME="daud77245"
$env:MLFLOW_TRACKING_PASSWORD="<YOUR_DAGSHUB_TOKEN>"
dvc repro
dvc metrics show
dvc dag
git add dvc.lock params.yaml reports/.gitignore data/processed/.gitignore
git commit -m "Run pipeline v1"
dvc push
git push
```
Without the three `MLFLOW_*` variables, runs go to a local `mlruns/` folder instead.
If a stage fails, read its message: validation stops the pipeline on purpose (corrupt image,
bad annotation, missing frame numbers, too few faces, empty split). Fix the data, re-run.

Open DagsHub -> **Experiments** (MLflow UI) to see the three runs and the registered model.

## 5. Docker (required)

```powershell
docker build -t gazeguard-train .
copy .env.example .env                 # then edit .env: set DAGSHUB_TOKEN
docker run --rm --env-file .env gazeguard-train
```
The container pulls the raw data with DVC, runs the unit tests, re-runs **every** stage with
`dvc repro --force`, logs to MLflow and prints the metrics. To keep local MLflow files when
running without DagsHub: `docker run --rm -v ${PWD}/mlruns:/app/mlruns gazeguard-train`.

## What to submit

- GitHub repository link (public, or give the teacher access).
- DVC files: `.dvc/config`, `dvc.yaml`, `dvc.lock`, `data/raw.dvc`, `params.yaml`.
- Screenshots: DagsHub Experiments table (3 runs with params + metrics), one run page showing the
  dataset tags (`raw_data_dvc_md5`, `git_commit`), the **Models** tab with the registered version,
  `dvc metrics show`, passing `pytest`, and the Docker run output.
- Queries for the teacher: private comment on the assignment.

## Things to state honestly in your report

- Few subjects: by default the split is **scenario-disjoint**. If you can tell which scenarios
  show the same person, fill `split.subject_map` to make it subject-disjoint.
- Neighbouring frames are nearly identical; that is exactly why random frame splits are not used.
- If the primary model does not beat the baselines, say so and show the numbers.
- Switching to your own recordings later = add them under `data/raw`, `dvc add`, commit, push;
  the pipeline re-runs and MLflow records the new dataset version.
