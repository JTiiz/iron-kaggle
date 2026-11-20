"""Publish the four forecasting models to the Hugging Face Hub, each with a model card.

    uv run --group train python training/upload_to_hub.py --dry-run   # writes the card previews
    uv run --group train python training/upload_to_hub.py             # uploads (after `hf auth login`)
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "training" / "reports"
MODELS = ROOT / "models"
PROJECT_URL = "https://github.com/JTiiz/iron-kaggle"
DATASET = "AiiN-aini/rossmann-store-sales"

DESCRIPTIONS = {
    "linear-regression": "Ordinary least squares on one-hot encoded calendar and store categories plus "
                         "standardised numeric features.",
    "decision-tree": "A single regression tree (max depth 18, at least 20 store-days per leaf).",
    "random-forest": "60 regression trees (max depth 20), each grown on half of the rows and half "
                     "of the features.",
    "gradient-boosting": "scikit-learn `HistGradientBoostingRegressor` with native categorical "
                         "support and early stopping (best of the four).",
}

CARD = """---
library_name: sklearn
pipeline_tag: tabular-regression
tags: [sklearn, tabular-regression, time-series-forecasting, retail, rossmann]
datasets: [{dataset}]
---

# Rossmann daily sales forecast · {name}

Predicts the daily sales (EUR) of any of the 1,115 Rossmann drugstores from what is known in
advance: the store, the date, whether a promotion runs and the holidays. It does **not** use the
number of customers of that day.

{description}

| Metric (last 6 weeks, held out, n={test_rows:,} store-days) | Value |
|---|---|
| R² | {r2:.3f} |
| RMSPE (Kaggle metric) | {rmspe:.3f} |
| RMSE | {rmse:,.0f} EUR |
| MAE | {mae:,.0f} EUR |

For reference, predicting each store's average for that weekday gives R² 0.696 and RMSPE 0.224.

![Actual vs predicted](actual_vs_predicted.png)

**Data:** Kaggle "Rossmann Store Sales" (`train.csv` + `store.csv`), read from
[{dataset}](https://huggingface.co/datasets/{dataset}). Open days with sales only.
Train: {train_period[0]} to {train_period[1]} ({train_rows:,} store-days).
Test: {test_period[0]} to {test_period[1]}: the last 6 weeks, never seen by the model or by the
per-store statistics.

**Target:** `log1p(sales)`; predictions are turned back into euros with `expm1`.

**Features ({n_features}):** calendar (weekday, day, month, year, week and day of year, month
start/end), promotion and holiday flags, store type, assortment, competition distance and age,
Promo2 status, and four per-store statistics from the training period (`store_stats.csv`).

## Usage

The features are built by `forecast/features.py` in the
[project repository]({project_url}); scikit-learn must be version {sklearn}.

```python
import joblib, numpy as np, pandas as pd
from huggingface_hub import snapshot_download
from forecast.features import build_features   # from the project repository

path = snapshot_download("{repo_id}")
model = joblib.load(f"{{path}}/model.joblib")   # only load pickles you trust
stats = pd.read_csv(f"{{path}}/store_stats.csv")
day = pd.DataFrame([{{"store": 336, "date": "2015-08-07", "promotion": 1,
                     "state_holiday": 0, "school_holiday": 0}}])
X = build_features(day, stats)
features = {features}
print(np.expm1(model.predict(X[features])))   # EUR
```
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--user", default="jramduran", help="Hub account (only used with --dry-run)")
    args = ap.parse_args()
    api = HfApi()
    user = args.user if args.dry_run else api.whoami()["name"]

    for key, description in DESCRIPTIONS.items():
        src = MODELS / f"rossmann-{key}"
        info = json.loads((src / "model_info.json").read_text())
        repo = f"{user}/rossmann-sales-{key}"
        card = CARD.format(name=info["name"], description=description, dataset=DATASET,
                           project_url=PROJECT_URL, repo_id=repo, n_features=len(info["features"]),
                           features=json.dumps(info["features"]), sklearn=info["sklearn"],
                           **info["metrics"], **{k: info[k] for k in
                           ("test_rows", "train_rows", "train_period", "test_period")})
        files = {p.name: p for p in src.iterdir() if p.is_file()}
        files["actual_vs_predicted.png"] = REPORTS / key / "actual_vs_predicted.png"
        size = sum(p.stat().st_size for p in files.values()) / 1e6
        print(f"{repo}: {sorted(files)} ({size:.0f} MB)")
        if args.dry_run:
            (REPORTS / key / "MODEL_CARD_PREVIEW.md").write_text(card, encoding="utf-8")
            continue
        tmp = Path(tempfile.mkdtemp(prefix="hub_"))
        for name, p in files.items():
            shutil.copy2(p, tmp / name)
        (tmp / "README.md").write_text(card, encoding="utf-8")
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=str(tmp),
                          commit_message="Upload model from the Iron Kaggle sales forecasting project")
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"  -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
