"""Feature engineering shared by training and the app.

Everything a model sees is computed from a store, a date and the day's flags
(open, promotion, holidays), plus per-store statistics computed on the training
period only. Nothing from the evaluated days leaks into the features.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Pinned copy of the Kaggle "Rossmann Store Sales" files (identical bytes to the
# competition download; verified against a second mirror).
DATASET_REPO = "AiiN-aini/rossmann-store-sales"
DATASET_REVISION = "bccc4440e13d9e8e55dd2b344e6867767ffc7e48"

# The last 6 weeks of the data are the test period, as in the Kaggle competition
# (whose own test set is also 6 weeks long).
TEST_START = pd.Timestamp("2015-06-20")

STORE_TYPES = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORTMENTS = {"a": 0, "b": 1, "c": 2}  # basic, extra, extended
STATE_HOLIDAYS = {"0": 0, "a": 1, "b": 2, "c": 3}  # none, public, Easter, Christmas
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]

CALENDAR = ["day_of_week", "day", "month", "year", "week_of_year", "day_of_year",
            "is_month_start", "is_month_end"]
FLAGS = ["promotion", "state_holiday", "school_holiday"]
STORE_STATIC = ["store_type", "assortment", "competition_distance_log",
                "competition_open_months", "promo2_active", "promo2_month"]
STORE_STATS = ["store_mean_log_sales", "store_dow_log_sales", "store_promo_lift",
               "store_mean_log_customers"]
FORECAST_FEATURES = CALENDAR + FLAGS + STORE_STATIC + STORE_STATS
# The original Iron Kaggle challenge gave the number of customers of the same day.
CUSTOMERS_FEATURES = FORECAST_FEATURES + ["log_customers", "store_log_sales_per_customer"]
SETUPS = {"forecast": FORECAST_FEATURES, "customers": CUSTOMERS_FEATURES}
CATEGORICAL = ["day_of_week", "month", "state_holiday", "store_type", "assortment"]


def download_raw(cache_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Return local paths to train.csv and store.csv (downloaded once, then cached)."""
    from huggingface_hub import hf_hub_download

    kw = dict(repo_id=DATASET_REPO, repo_type="dataset", revision=DATASET_REVISION, cache_dir=cache_dir)
    return Path(hf_hub_download(filename="train.csv", **kw)), Path(hf_hub_download(filename="store.csv", **kw))


def load_sales(path: str | Path) -> pd.DataFrame:
    """Kaggle train.csv with readable column names, sorted by store and date."""
    df = pd.read_csv(path, dtype={"StateHoliday": str}, parse_dates=["Date"])
    df = df.rename(columns={
        "Store": "store", "DayOfWeek": "day_of_week", "Date": "date", "Sales": "sales",
        "Customers": "customers", "Open": "open", "Promo": "promotion",
        "StateHoliday": "state_holiday", "SchoolHoliday": "school_holiday",
    })
    df["state_holiday"] = df["state_holiday"].map(STATE_HOLIDAYS).astype("int8")
    return df.sort_values(["store", "date"]).reset_index(drop=True)


def load_stores(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path).rename(columns={"Store": "store"})


def trainable(df: pd.DataFrame) -> pd.DataFrame:
    """Days that count: open with sales (the competition ignores days with 0 sales)."""
    return df[(df["open"] == 1) & (df["sales"] > 0)]


def store_statistics(train: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """One row per (store, day_of_week): static store data + stats from the training period."""
    t = trainable(train).assign(log_sales=lambda d: np.log1p(d["sales"]),
                                log_customers=lambda d: np.log1p(d["customers"]))
    per_store = t.groupby("store").agg(
        store_mean_log_sales=("log_sales", "mean"),
        store_mean_log_customers=("log_customers", "mean"),
        sales_sum=("sales", "sum"), customers_sum=("customers", "sum"),
    )
    per_store["store_log_sales_per_customer"] = np.log(per_store["sales_sum"] / per_store["customers_sum"])
    promo = t.groupby(["store", "promotion"])["log_sales"].mean().unstack()
    per_store["store_promo_lift"] = (promo[1] - promo[0]).fillna(0)
    per_dow = t.groupby(["store", "day_of_week"])["log_sales"].mean().rename("store_dow_log_sales")

    grid = pd.MultiIndex.from_product([stores["store"], range(1, 8)], names=["store", "day_of_week"])
    out = per_dow.reindex(grid).reset_index().merge(
        per_store.drop(columns=["sales_sum", "customers_sum"]).reset_index(), on="store", how="left")
    # Days a store never opened on (most Sundays): fall back to its overall mean.
    out["store_dow_log_sales"] = out["store_dow_log_sales"].fillna(out["store_mean_log_sales"])

    s = stores.copy()
    s["store_type"] = s["StoreType"].map(STORE_TYPES)
    s["assortment"] = s["Assortment"].map(ASSORTMENTS)
    s["competition_distance"] = s["CompetitionDistance"].fillna(100_000)
    keep = ["store", "store_type", "assortment", "competition_distance", "CompetitionOpenSinceYear",
            "CompetitionOpenSinceMonth", "Promo2", "Promo2SinceYear", "Promo2SinceWeek", "PromoInterval"]
    return out.merge(s[keep], on="store", how="left")


def build_features(days: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Add every model feature to `days` (columns: store, date, promotion, state_holiday,
    school_holiday and, for the customers setup, customers)."""
    d = days.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["day_of_week"] = d["date"].dt.day_of_week + 1
    d["day"] = d["date"].dt.day
    d["month"] = d["date"].dt.month
    d["year"] = d["date"].dt.year
    d["week_of_year"] = d["date"].dt.isocalendar().week.astype(int).to_numpy()
    d["day_of_year"] = d["date"].dt.day_of_year
    d["is_month_start"] = d["date"].dt.is_month_start.astype(int)
    d["is_month_end"] = d["date"].dt.is_month_end.astype(int)

    d = d.merge(stats, on=["store", "day_of_week"], how="left", validate="many_to_one")
    if d["store_type"].isna().any():
        raise ValueError("unknown store id")

    d["competition_distance_log"] = np.log1p(d["competition_distance"])
    opened = (d["CompetitionOpenSinceYear"] - 1900) * 12 + d["CompetitionOpenSinceMonth"]
    now = (d["year"] - 1900) * 12 + d["month"]
    # months since the nearest competitor opened (0 = not open yet, -1 = unknown)
    d["competition_open_months"] = (now - opened).clip(lower=0).fillna(-1)

    since = d["Promo2SinceYear"] * 100 + d["Promo2SinceWeek"]
    current = d["year"] * 100 + d["week_of_year"]
    d["promo2_active"] = ((d["Promo2"] == 1) & (current >= since)).astype(int)
    month_name = d["month"].map(lambda m: MONTH_ABBR[m - 1])
    in_interval = [isinstance(iv, str) and mn in iv.split(",") for iv, mn in zip(d["PromoInterval"], month_name)]
    d["promo2_month"] = (d["promo2_active"].astype(bool) & np.array(in_interval)).astype(int)

    if "customers" in d:
        d["log_customers"] = np.log1p(d["customers"])
    return d
