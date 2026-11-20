"""Charts and the comparison table for the README and the model cards.

    uv run --group train python training/report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPORTS = Path(__file__).resolve().parent / "reports"
KEYS = {"linear-regression": "Linear regression", "decision-tree": "Decision tree",
        "random-forest": "Random forest", "gradient-boosting": "Gradient boosting"}
# Fixed categorical order (one colour per model, everywhere in the project)
COLORS = {"linear-regression": "#2a78d6", "decision-tree": "#eb6834",
          "random-forest": "#1baf7a", "gradient-boosting": "#eda100"}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.axisbelow": True, "axes.spines.right": False, "font.size": 10,
})


def comparison(results: pd.DataFrame) -> None:
    fc = results[results.setup == "forecast"].set_index("model")
    cu = results[results.setup == "customers"].set_index("model")
    base = results[results.setup == "baseline"].iloc[0]
    names = list(KEYS.values())
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 3.9))
    h = 0.36
    ax.barh(y - h / 2, fc.loc[names, "r2"], h - 0.04, color="#2a78d6", label="Forecast (no customers)")
    ax.barh(y + h / 2, cu.loc[names, "r2"], h - 0.04, color="#eb6834", label="With same-day customers")
    for i, n in enumerate(names):
        ax.text(fc.loc[n, "r2"] + 0.004, i - h / 2, f"{fc.loc[n, 'r2']:.3f}", va="center", fontsize=9, color=INK2)
        ax.text(cu.loc[n, "r2"] + 0.004, i + h / 2, f"{cu.loc[n, 'r2']:.3f}", va="center", fontsize=9, color=INK2)
    ax.axvline(base.r2, color=MUTED, lw=1, ls="--")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0.65, 1.0)
    ax.set_xlabel(f"R² on the last 6 weeks (held out)  ·  dashed line: store x weekday average ({base.r2:.3f})")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS / "comparison.png", dpi=150)
    plt.close(fig)


def daily_total(pred: pd.DataFrame) -> None:
    daily = pred.groupby("date")[["sales", *KEYS]].sum() / 1e6
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(daily.index, daily["sales"], color=INK, lw=2, label="Actual")
    for k in ("linear-regression", "gradient-boosting"):
        ax.plot(daily.index, daily[k], color=COLORS[k], lw=2, label=KEYS[k])
    ax.set_ylabel("Sales, all open stores (M EUR)")
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(REPORTS / "daily_total.png", dpi=150)
    plt.close(fig)


def actual_vs_predicted(pred: pd.DataFrame, key: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lim = float(np.quantile(pred["sales"], 0.998))
    hb = ax.hexbin(pred["sales"], pred[key], gridsize=55, extent=(0, lim, 0, lim), mincnt=1,
                   cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                       "blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]), bins="log")
    ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual daily sales (EUR)")
    ax.set_ylabel("Predicted (EUR)")
    ax.set_title(KEYS[key], fontsize=11, loc="left")
    cb = fig.colorbar(hb, ax=ax, fraction=0.045)
    cb.set_label("store-days (log)", color=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    results = pd.read_csv(REPORTS / "results.csv")
    pred = pd.read_csv(REPORTS / "test_predictions.csv.gz", parse_dates=["date"])
    comparison(results)
    daily_total(pred)
    for key in KEYS:
        folder = REPORTS / key
        folder.mkdir(exist_ok=True)
        actual_vs_predicted(pred, key, folder / "actual_vs_predicted.png")
        r = results[(results.setup == "forecast") & (results.model == KEYS[key])].iloc[0]
        (folder / "metrics.json").write_text(json.dumps(
            {k: round(float(r[k]), 4) for k in ("r2", "rmse", "mae", "rmspe", "train_seconds")}, indent=2))

    table = ["| Model | Forecast R² | RMSE (EUR) | MAE (EUR) | RMSPE | With customers R² | Train time |",
             "|---|---|---|---|---|---|---|"]
    base = results[results.setup == "baseline"].iloc[0]
    table.append(f"| Store x weekday average | {base.r2:.3f} | {base.rmse:.0f} | {base.mae:.0f} | "
                 f"{base.rmspe:.3f} | - | - |")
    for name in KEYS.values():
        f = results[(results.setup == "forecast") & (results.model == name)].iloc[0]
        c = results[(results.setup == "customers") & (results.model == name)].iloc[0]
        table.append(f"| {name} | {f.r2:.3f} | {f.rmse:.0f} | {f.mae:.0f} | {f.rmspe:.3f} | "
                     f"{c.r2:.3f} | {f.train_seconds:.0f} s |")
    (REPORTS / "comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print("\n".join(table))


if __name__ == "__main__":
    main()
