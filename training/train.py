"""Train and evaluate the four models on a time-based split.

    uv run --group train python training/train.py            # both setups, all models
    uv run --group train python training/train.py --quick    # 10 % of the stores, for a smoke test

Setups
  forecast   what a store manager knows in advance: store, date, promotion, holidays.
             These four models are saved to models/ and used by the app.
  customers  the original Iron Kaggle challenge, which also gave the number of customers
             of that same day. Reported for comparison with the notebook; not saved.

Every model learns log(1 + sales); metrics are computed on sales in euros, on the last
6 weeks of data (2015-06-20 to 2015-07-31), which no model or statistic has seen.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forecast import features as F  # noqa: E402

REPORTS = ROOT / "training" / "reports"
MODELS = ROOT / "models"
SEED = 42

MODEL_SPECS = {
    "linear-regression": "Linear regression",
    "decision-tree": "Decision tree",
    "random-forest": "Random forest",
    "gradient-boosting": "Gradient boosting",
}


def make_model(key: str, feature_names: list[str]):
    if key == "linear-regression":
        cat = [c for c in F.CATEGORICAL if c in feature_names]
        num = [c for c in feature_names if c not in cat]
        prep = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", StandardScaler(), num),
        ])
        return Pipeline([("prep", prep), ("model", LinearRegression())])
    if key == "decision-tree":
        return DecisionTreeRegressor(max_depth=18, min_samples_leaf=20, random_state=SEED)
    if key == "random-forest":
        return RandomForestRegressor(n_estimators=60, max_depth=20, min_samples_leaf=10,
                                     max_features=0.5, max_samples=0.5, n_jobs=-1, random_state=SEED)
    if key == "gradient-boosting":
        return HistGradientBoostingRegressor(
            max_iter=2000, learning_rate=0.08, max_leaf_nodes=127, min_samples_leaf=40,
            l2_regularization=1.0, categorical_features=[c for c in F.CATEGORICAL if c in feature_names],
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=SEED)
    raise KeyError(key)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        # Kaggle competition metric (root mean squared percentage error)
        "rmspe": float(np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(MODEL_SPECS))
    args = ap.parse_args()

    t0 = time.perf_counter()
    train_csv, store_csv = F.download_raw()
    sales, stores = F.load_sales(train_csv), F.load_stores(store_csv)
    if args.quick:
        keep = stores["store"].sample(frac=0.1, random_state=SEED)
        sales, stores = sales[sales["store"].isin(keep)], stores[stores["store"].isin(keep)]

    train_days = sales[sales["date"] < F.TEST_START]
    test_days = sales[sales["date"] >= F.TEST_START]
    stats = F.store_statistics(train_days, stores)  # training period only
    train = F.build_features(F.trainable(train_days), stats)
    test = F.build_features(F.trainable(test_days), stats)
    y_train, y_test = np.log1p(train["sales"].to_numpy()), test["sales"].to_numpy()

    dataset = {
        "source": f"https://huggingface.co/datasets/{F.DATASET_REPO}",
        "revision": F.DATASET_REVISION,
        "stores": int(stores["store"].nunique()),
        "rows_total": int(len(sales)),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_period": [str(train["date"].min().date()), str(train["date"].max().date())],
        "test_period": [str(test["date"].min().date()), str(test["date"].max().date())],
    }
    print(json.dumps(dataset, indent=2), flush=True)
    out_dir = REPORTS / ("quick" if args.quick else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset.json").write_text(json.dumps(dataset, indent=2))

    # Reference without any model: the store's average for that weekday in the training period.
    baseline = metrics(y_test, np.expm1(test["store_dow_log_sales"].to_numpy()))
    rows = [{"setup": "baseline", "model": "Store x weekday average", **baseline, "train_seconds": 0}]
    predictions = test[["store", "date", "sales"]].copy()

    for setup, cols in F.SETUPS.items():
        for key, name in MODEL_SPECS.items():
            if args.only and key != args.only:
                continue
            model = make_model(key, cols)
            t = time.perf_counter()
            model.fit(train[cols], y_train)
            seconds = time.perf_counter() - t
            pred = np.expm1(model.predict(test[cols]))
            m = metrics(y_test, pred)
            rows.append({"setup": setup, "model": name, **m, "train_seconds": round(seconds, 1)})
            print(f"{setup:9s} {name:18s} R2={m['r2']:.4f} RMSE={m['rmse']:7.1f} "
                  f"RMSPE={m['rmspe']:.4f} ({seconds:.0f}s)", flush=True)
            if setup != "forecast":
                continue
            predictions[key] = pred.round(1)
            folder = MODELS / f"rossmann-{key}"
            if args.quick:
                continue
            folder.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, folder / "model.joblib", compress=3)
            stats.to_csv(folder / "store_stats.csv", index=False)
            info = {"key": key, "name": name, "features": cols, "target": "log1p(sales)",
                    "sklearn": __import__("sklearn").__version__, "metrics": m,
                    "train_seconds": round(seconds, 1), **{k: dataset[k] for k in
                    ("train_rows", "test_rows", "train_period", "test_period")}}
            (folder / "model_info.json").write_text(json.dumps(info, indent=2))
            if hasattr(model, "feature_importances_"):
                imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
                imp.round(5).to_csv(out_dir / f"importance_{key}.csv", header=["importance"])

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "results.csv", index=False)
    predictions.to_csv(out_dir / "test_predictions.csv.gz", index=False)
    print(results.to_string(index=False))
    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
