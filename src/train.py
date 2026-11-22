"""Stage 5: baselines + primary temporal model, all logged to MLflow; best model registered.

Candidates (each is one MLflow run):
  1. majority_class      - trivial baseline (predicts the most common training class)
  2. random_forest       - classical baseline on window statistics (mean/std/min/max per channel)
  3. cnn_gru             - PRIMARY: 1D-CNN + GRU on the raw 30-step window sequence

MLflow target comes from the environment:
  MLFLOW_TRACKING_URI (+ MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD for DagsHub);
  if MLFLOW_TRACKING_URI is not set, runs go to a local ./mlruns folder.
"""
import hashlib
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import torch
import torch.nn as nn
import yaml
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from src.common import ROOT, load_params, set_seed, write_json


class CNNGRU(nn.Module):
    """Two Conv1d layers pick up short movement patterns; a GRU models their order in time."""

    def __init__(self, n_feat, n_classes, conv=32, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, conv, 3, padding=1), nn.ReLU(),
            nn.Conv1d(conv, conv, 3, padding=1), nn.ReLU())
        self.gru = nn.GRU(conv, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):                       # x: (N, T, F)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, hn = self.gru(h)
        return self.fc(hn[-1])


def window_stats(X):
    """(N,T,F) -> (N,4F): mean, std, min, max over time for each channel."""
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1)


def metrics_for(y_true, y_pred, prefix):
    return {f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
            f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}


def log_eval_artifacts(name, y_true, y_pred, classes):
    out_dir = ROOT / "reports" / "confusion_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=[str(int(c)) for c in classes]).plot(
        ax=ax, colorbar=False)
    ax.set_title(f"{name} - test confusion matrix")
    png = out_dir / f"{name}.png"
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=[str(int(c)) for c in classes], zero_division=0)
    txt = out_dir / f"{name}_report.txt"
    txt.write_text(report, encoding="utf-8")
    mlflow.log_artifact(str(png), "evaluation")
    mlflow.log_artifact(str(txt), "evaluation")


def dataset_tags():
    tags = {}
    dvc_file = ROOT / "data" / "raw.dvc"
    if dvc_file.exists():
        info = yaml.safe_load(dvc_file.read_text(encoding="utf-8"))
        tags["raw_data_dvc_md5"] = str(info["outs"][0].get("md5", "unknown"))
    else:
        tags["raw_data_dvc_md5"] = "not-tracked"
    tags["windows_sha256"] = hashlib.sha256(
        (ROOT / "data" / "processed" / "windows.npz").read_bytes()).hexdigest()[:16]
    try:
        tags["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        tags["git_commit"] = "unknown"
    return tags


def train_cnn_gru(d, p, seed):
    set_seed(seed)
    n_classes = len(d["classes"])
    model = CNNGRU(d["X_train"].shape[2], n_classes, p["conv_channels"], p["gru_hidden"])
    counts = np.bincount(d["y_train"], minlength=n_classes).astype(np.float32)
    weights = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"])
    Xtr, ytr = torch.tensor(d["X_train"]), torch.tensor(d["y_train"])
    Xva = torch.tensor(d["X_val"])
    gen = torch.Generator().manual_seed(seed)
    best_f1, best_state, bad, history = -1.0, None, 0, []

    def predict(X):
        model.eval()
        with torch.no_grad():
            return model(X).argmax(1).numpy()

    for epoch in range(p["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr), p["batch_size"]):
            b = perm[i:i + p["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
        val_f1 = f1_score(d["y_val"], predict(Xva), average="macro", zero_division=0) \
            if len(Xva) else 0.0
        history.append((epoch, float(loss.item()), float(val_f1)))
        if val_f1 > best_f1:
            best_f1, bad = val_f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= p["patience"]:
                break
    model.load_state_dict(best_state)
    return model, predict, history


def main():
    params = load_params()
    p, seed = params["train"], params["seed"]
    set_seed(seed)
    d = dict(np.load(ROOT / "data" / "processed" / "windows.npz"))
    classes = d["classes"]
    for k in ("train", "val", "test"):
        if len(d[f"y_{k}"]) == 0:
            raise SystemExit(f"split '{k}' is empty - add more scenarios or lower val/test fractions")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or f"file:{(ROOT / 'mlruns').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    tags = dataset_tags()
    print(f"[train] MLflow: {tracking_uri}")

    common_params = {"seed": seed, "n_train": len(d["y_train"]), "n_val": len(d["y_val"]),
                     "n_test": len(d["y_test"]), "n_classes": len(classes),
                     "window_size": d["X_train"].shape[1], **{f"win_{k}": v for k, v in
                                                              params["windows"].items()}}
    results = {}   # name -> dict(run_id, val_macro_f1, test_*)

    def finish(name, run, y_val_pred, y_test_pred, extra_params, log_model_fn):
        m = {**metrics_for(d["y_val"], y_val_pred, "val"),
             **metrics_for(d["y_test"], y_test_pred, "test")}
        mlflow.log_params({**common_params, **extra_params})
        mlflow.set_tags({**tags, "model_name": name})
        mlflow.log_metrics(m)
        log_eval_artifacts(name, d["y_test"], y_test_pred, classes)
        log_model_fn()
        results[name] = {"run_id": run.info.run_id, **m}
        print(f"[train] {name}: val_macro_f1={m['val_macro_f1']:.3f} "
              f"test_macro_f1={m['test_macro_f1']:.3f}")

    # 1) trivial baseline
    with mlflow.start_run(run_name="majority_class") as run:
        dummy = DummyClassifier(strategy="most_frequent").fit(window_stats(d["X_train"]), d["y_train"])
        finish("majority_class", run, dummy.predict(window_stats(d["X_val"])),
               dummy.predict(window_stats(d["X_test"])), {"model": "majority_class"},
               lambda: mlflow.sklearn.log_model(dummy, "model"))

    # 2) classical baseline
    with mlflow.start_run(run_name="random_forest") as run:
        rf = RandomForestClassifier(n_estimators=p["rf_trees"], random_state=seed, n_jobs=1,
                                    class_weight="balanced")
        rf.fit(window_stats(d["X_train"]), d["y_train"])
        finish("random_forest", run, rf.predict(window_stats(d["X_val"])),
               rf.predict(window_stats(d["X_test"])),
               {"model": "random_forest", "rf_trees": p["rf_trees"]},
               lambda: mlflow.sklearn.log_model(rf, "model"))

    # 3) primary temporal model
    with mlflow.start_run(run_name="cnn_gru") as run:
        model, predict, history = train_cnn_gru(d, p, seed)
        n_params = int(sum(t.numel() for t in model.parameters()))
        for epoch, loss, vf1 in history:
            mlflow.log_metrics({"train_loss": loss, "val_macro_f1_epoch": vf1}, step=epoch)
        finish("cnn_gru", run, predict(torch.tensor(d["X_val"])),
               predict(torch.tensor(d["X_test"])),
               {"model": "cnn_gru", "epochs_run": len(history), "n_parameters": n_params,
                **{k: p[k] for k in ("epochs", "batch_size", "lr", "conv_channels",
                                     "gru_hidden", "patience")}},
               lambda: mlflow.pytorch.log_model(model, "model"))

    # choose + register
    policy = params["mlflow"]["register_policy"]
    candidates = {"cnn_gru": results["cnn_gru"]} if policy == "primary" else results
    best = max(candidates, key=lambda n: candidates[n]["val_macro_f1"])   # select on VALIDATION only
    reg_name = params["mlflow"]["registered_model_name"]
    mv = mlflow.register_model(f"runs:/{results[best]['run_id']}/model", reg_name)
    try:
        MlflowClient().set_registered_model_alias(reg_name, "champion", mv.version)
    except Exception as exc:  # older MLflow servers may not support aliases
        print(f"[train] could not set alias: {exc}")
    print(f"[train] registered '{best}' as {reg_name} v{mv.version}")

    flat = {"best_model": best, "registered_model": reg_name, "registered_version": str(mv.version)}
    for name, r in results.items():
        for k, v in r.items():
            if k != "run_id":
                flat[f"{name}.{k}"] = round(v, 4)
    write_json(ROOT / "metrics.json", flat)


if __name__ == "__main__":
    main()
