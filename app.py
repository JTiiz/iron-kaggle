"""Store Sales Forecast: compare four models on a store's last 6 weeks and forecast any day.

    uv run streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from forecast import features as F

APP_DIR = Path(__file__).resolve().parent
MODELS_ROOT = Path(os.getenv("MODELS_ROOT", APP_DIR / "models"))
REGISTRY = APP_DIR / "models" / "models.yaml"
DEFAULT_STORE = int(os.getenv("DEFAULT_STORE", "336"))  # a store with a typical error
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
STORE_TYPE_NAMES = {0: "a", 1: "b", 2: "c", 3: "d"}
ASSORTMENT_NAMES = {0: "basic", 1: "extra", 2: "extended"}
HOLIDAYS = {"None": 0, "Public holiday": 1, "Easter": 2, "Christmas": 3}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

st.set_page_config(page_title="Store Sales Forecast", page_icon="📈", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.6rem; padding-bottom: 1rem; max-width: 1320px;}
h1 {font-size: 2rem !important; margin-bottom: 0 !important;}
[data-testid="stMetricValue"] {font-size: 1.5rem;}
.muted {color: #52514e; font-size: 0.92rem;}
.pill {display:inline-block; padding:0.1rem 0.55rem; margin:0 0.3rem 0.3rem 0; border-radius:999px;
       background:#f3f2ee; color:#52514e; font-size:0.82rem;}
.swatch {display:inline-block; width:0.7rem; height:0.7rem; border-radius:2px; margin-right:0.35rem;
         vertical-align:baseline;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Loading (cached for the life of the server)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def registry() -> list[dict]:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["models"]


def resolve(entry: dict) -> tuple[Path, str]:
    local = MODELS_ROOT / entry["folder"]
    if (local / "model_info.json").exists():
        return local, "local"
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(entry["hub_id"], allow_patterns=["*.json", "*.joblib", "*.csv"])), entry["hub_id"]


@st.cache_resource(show_spinner="Loading the four models (downloaded from Hugging Face the first time)...")
def load_models() -> dict[str, dict]:
    out = {}
    for entry in registry():
        path, source = resolve(entry)
        out[entry["key"]] = {
            **entry, "source": source,
            "model": joblib.load(path / "model.joblib"),
            "info": json.loads((path / "model_info.json").read_text(encoding="utf-8")),
            "stats_path": path / "store_stats.csv",
        }
    return out


@st.cache_resource(show_spinner="Loading store statistics...")
def load_stats(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner="Downloading the Rossmann sales history (38 MB, only the first time)...")
def load_test_period() -> pd.DataFrame:
    train_csv, _ = F.download_raw()
    sales = F.load_sales(train_csv)
    return sales[sales["date"] >= F.TEST_START].reset_index(drop=True)


def predict(models: dict, days: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    X = F.build_features(days, stats)
    out = pd.DataFrame({"date": X["date"]})
    for key, m in models.items():
        out[key] = np.expm1(m["model"].predict(X[m["info"]["features"]]))
    return out


def rmspe(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean(((y - p) / y) ** 2)))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

models = load_models()
first = next(iter(models.values()))
stats = load_stats(str(first["stats_path"]))
store_rows = stats.drop_duplicates("store").set_index("store")

st.title("Store Sales Forecast")
st.markdown(
    "<p class='muted'>Four scikit-learn models predict the daily sales of 1,115 Rossmann stores from "
    "what is known in advance: the store, the date, promotions and holidays. "
    f"Trained on {first['info']['train_period'][0]} to {first['info']['train_period'][1]}, "
    f"tested on the last 6 weeks ({first['info']['test_period'][0]} to {first['info']['test_period'][1]}).</p>",
    unsafe_allow_html=True)

left, right = st.columns([1, 3], gap="large")
with left:
    ids = store_rows.index.tolist()
    store = st.selectbox("Store", ids, index=ids.index(DEFAULT_STORE) if DEFAULT_STORE in ids else 0,
                         format_func=lambda s: f"Store {s}")
    s = store_rows.loc[store]
    comp = "unknown" if s["competition_distance"] >= 100_000 else f"{s['competition_distance'] / 1000:.1f} km"
    st.markdown(
        f"<span class='pill'>Type {STORE_TYPE_NAMES[int(s['store_type'])]}</span>"
        f"<span class='pill'>{ASSORTMENT_NAMES[int(s['assortment'])]} assortment</span>"
        f"<span class='pill'>competitor {comp}</span>"
        f"<span class='pill'>{'in' if s['Promo2'] == 1 else 'no'} Promo2</span>"
        f"<br><span class='muted'>Average day: {np.expm1(s['store_mean_log_sales']):,.0f} EUR</span>",
        unsafe_allow_html=True)
    st.markdown("**Models** · R² on the test period, all stores")
    for key, m in models.items():
        src = "local copy" if m["source"] == "local" else f"[{m['hub_id']}](https://huggingface.co/{m['hub_id']})"
        st.markdown(f"<span class='swatch' style='background:{m['color']}'></span>**{m['name']}** "
                    f"{m['info']['metrics']['r2']:.3f}<br><span class='muted'>{src}</span>",
                    unsafe_allow_html=True)

with right:
    tab_compare, tab_day = st.tabs(["Last 6 weeks: models vs actual", "Forecast a day"])

    with tab_compare:
        hist = load_test_period()
        days = hist[(hist["store"] == store) & (hist["open"] == 1) & (hist["sales"] > 0)]
        if days.empty:
            st.info("This store has no open days in the test period.")
        else:
            pred = predict(models, days, stats)
            y = days["sales"].to_numpy()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pred["date"], y=y, name="Actual", mode="lines+markers",
                                     line=dict(color=INK, width=2.5), marker=dict(size=6),
                                     hovertemplate="%{y:,.0f} EUR"))
            for key, m in models.items():
                fig.add_trace(go.Scatter(x=pred["date"], y=pred[key], name=m["name"], mode="lines",
                                         line=dict(color=m["color"], width=2), hovertemplate="%{y:,.0f} EUR"))
            fig.update_layout(
                height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=INK2),
                legend=dict(orientation="h", y=1.08, x=0, font=dict(color=INK)),
                yaxis=dict(title="Daily sales (EUR)", gridcolor=GRID, zeroline=False, tickformat=",.0f"),
                xaxis=dict(gridcolor=GRID, showspikes=True, spikecolor=MUTED, spikethickness=1))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            cols = st.columns(len(models))
            ranked = sorted(models, key=lambda k: rmspe(y, pred[k].to_numpy()))
            for col, key in zip(cols, ranked):
                err = rmspe(y, pred[key].to_numpy())
                total_err = (pred[key].sum() - y.sum()) / y.sum()
                col.metric(models[key]["name"], f"{err:.1%} error",
                           f"{total_err:+.1%} on the 6-week total", delta_color="off",
                           help="Typical daily error (RMSPE) for this store, and how far the model's "
                                "6-week total is from the real total.")

    with tab_day:
        c1, c2, c3 = st.columns(3)
        day = c1.date_input("Date", dt.date(2015, 8, 7), min_value=dt.date(2013, 1, 1),
                            max_value=dt.date(2015, 12, 31), format="DD/MM/YYYY")
        holiday = c2.selectbox("State holiday", list(HOLIDAYS))
        promo = c3.toggle("Promotion running", value=True)
        school = c3.toggle("School holiday", value=False)

        what_if = pd.DataFrame([{"store": store, "date": pd.Timestamp(day), "promotion": int(promo),
                                 "state_holiday": HOLIDAYS[holiday], "school_holiday": int(school)}])
        p = predict(models, what_if, stats).iloc[0]
        weekday = DAYS[pd.Timestamp(day).day_of_week]
        typical = float(np.expm1(stats.loc[(stats["store"] == store) &
                                           (stats["day_of_week"] == pd.Timestamp(day).day_of_week + 1),
                                           "store_dow_log_sales"].iloc[0]))
        st.markdown(f"**Store {store}, {weekday} {day:%d %B %Y}**  "
                    f"<span class='muted'>· this store's average {weekday}: {typical:,.0f} EUR</span>",
                    unsafe_allow_html=True)

        keys = list(models)
        fig = go.Figure(go.Bar(
            x=[p[k] for k in keys], y=[models[k]["name"] for k in keys], orientation="h",
            marker=dict(color=[models[k]["color"] for k in keys], line=dict(color=SURFACE, width=2),
                        cornerradius=4),
            text=[f"{p[k]:,.0f} EUR" for k in keys], textposition="inside", insidetextanchor="end",
            # white on the fills, ink on yellow (contrast)
            textfont=dict(color=[INK if models[k]["color"] == "#eda100" else "#ffffff" for k in keys], size=13),
            hovertemplate="%{y}: %{x:,.0f} EUR<extra></extra>"))
        fig.add_vline(x=typical, line=dict(color=MUTED, dash="dash", width=1), layer="below")
        fig.update_layout(
            height=260, margin=dict(l=10, r=40, t=10, b=10), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
            font=dict(color=INK2), bargap=0.35, showlegend=False,
            xaxis=dict(title="Predicted sales (EUR) · dashed line: store's average for this weekday",
                       gridcolor=GRID, tickformat=",.0f", range=[0, max(max(p[k] for k in keys), typical) * 1.08]),
            yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if holiday != "None" or pd.Timestamp(day).day_of_week == 6:
            st.caption("Most stores are closed on Sundays and public holidays; the forecast assumes the store opens.")
