# %% [markdown]
# # Air Quality Analysis and Prediction: next-day PM2.5 forecasting (ES 678)
#
# This file holds every function used in the project. `scripts/run_all.py` runs the
# experiments, and `notebooks/air_quality_colab.ipynb` is built from these same two files.
#
# **Question.** Can today's pollutant readings and calendar information predict tomorrow's
# city-level PM2.5 in Indian cities more accurately than persistence (tomorrow = today)?
# One row is one city on one day. Everything is built so that the features of a row use only
# information available at the end of that day, and every learned step is fit on training data only.

# %%
import json
import os
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # running inside a notebook
    ROOT = Path(".")

# %% [markdown]
# ## 0. Configuration
# All choices that matter are set here (and nowhere else), so they can be audited or changed.

# %%
SEED = 42

# Chronological split, defined by the date of the TARGET (the day being predicted).
TRAIN_END = "2018-12-31"      # train: 2015-2018
VAL_END = "2019-12-31"        # validation (tuning + conformal calibration): 2019
TEST_END = "2020-07-01"       # test: 2020 (the file ends on 2020-07-01); touched once
LOCKDOWN_START = "2020-03-25" # India's nationwide COVID-19 lockdown began on this date

MIN_TRAIN_ROWS = 200      # a city needs this many usable training rows to be kept
MAX_MISSING_TRAIN = 0.50  # drop a numeric feature if more than this share is missing in TRAIN
FFILL_LIMIT = 3           # past-only forward fill, at most 3 days
HIGH_Q = 0.90             # "high-pollution" = target above this quantile of TRAINING targets
ALPHA = 0.10              # intervals have 90% nominal coverage
BLOCK = 7                 # block length (days) for the moving-block bootstrap
NBOOT = 2000

POLL = ["PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
CAL = ["month_sin", "month_cos", "dow_sin", "dow_cos"]
BASE = ["pm25_today"] + POLL + CAL + ["City"]
FEATURE_SETS = {
    "today only": BASE,
    "lags 1-3": BASE + ["pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_roll3"],
    "lags 1-7": BASE + [f"pm25_lag{k}" for k in range(1, 8)] + ["pm25_roll3", "pm25_roll7"],
}
MAIN_SET = "lags 1-7"
MODEL_NAMES = ["ridge", "rf", "gb"]
MODEL_LABELS = {"persistence": "Persistence", "city-month": "City-month mean",
                "ridge": "Ridge", "rf": "Random forest", "gb": "Gradient boosting"}

# Hyperparameter grids, tuned on the 2019 validation year only.
GRIDS = {
    "ridge": [{"alpha": a} for a in (0.1, 1, 10, 100, 1000)],
    "rf": [{"max_depth": d, "min_samples_leaf": l} for d in (8, 14, None) for l in (3, 10)],
    "gb": [{"n_estimators": n, "learning_rate": lr, "max_depth": md}
           for n in (150, 400) for lr in (0.03, 0.1) for md in (2, 4)],
}

# %% [markdown]
# ## 1. Data loading and quality audit

# %%
def find_data(path=None):
    """Locate city_day.csv. In Google Colab, offer an upload dialog if it is not found."""
    cands = [path, os.environ.get("AQ_DATA"), ROOT / "data" / "city_day.csv", "city_day.csv",
             "data/city_day.csv", "/content/city_day.csv", "../data/city_day.csv"]
    for c in cands:
        if c and Path(c).exists():
            return str(c)
    try:
        from google.colab import files  # type: ignore
        print("city_day.csv not found. Please upload it.")
        up = files.upload()
        return [k for k in up if k.endswith(".csv")][0]
    except ImportError:
        raise FileNotFoundError(
            "city_day.csv not found. Download 'Air Quality Data in India (2015-2020)' from Kaggle "
            "and place city_day.csv in data/ (see data/README.md).")


def load_raw(path):
    raw = pd.read_csv(path, parse_dates=["Date"])
    return raw.sort_values(["City", "Date"]).reset_index(drop=True)


def to_daily(raw):
    """Reindex every city to a complete daily calendar so that lags are in calendar days."""
    cols = ["PM2.5"] + POLL
    parts = []
    for city, g in raw.groupby("City"):
        g = g.set_index("Date")[cols].sort_index()
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(idx)
        g.insert(0, "City", city)
        g.index.name = "Date"
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def audit(raw, daily):
    """Data-quality checks reported in the paper: missingness, duplicates, gaps, extremes."""
    obs = raw.dropna(subset=["PM2.5"])
    miss = (raw[["PM2.5"] + POLL + ["AQI"]].isna().mean() * 100).round(1).rename("missing_pct")
    per_city = pd.DataFrame({
        "rows": raw.groupby("City").size(),
        "calendar_days": daily.groupby("City").size(),
        "pm25_observed": obs.groupby("City").size(),
    }).fillna(0).astype(int)
    per_city["missing_calendar_days"] = per_city["calendar_days"] - per_city["rows"]
    by_year = obs.pivot_table(index="City", columns=obs["Date"].dt.year, values="PM2.5",
                              aggfunc="count").fillna(0).astype(int)
    pm = obs["PM2.5"]
    summary = {
        "rows": int(len(raw)), "cities": int(raw["City"].nunique()),
        "date_min": str(raw["Date"].min().date()), "date_max": str(raw["Date"].max().date()),
        "duplicate_city_dates": int(raw.duplicated(["City", "Date"]).sum()),
        "pm25_missing_pct": float(round(raw["PM2.5"].isna().mean() * 100, 1)),
        "pm25_min": float(pm.min()), "pm25_median": float(pm.median()),
        "pm25_p99": float(pm.quantile(0.99)), "pm25_max": float(pm.max()),
        "pm25_nonpositive": int((pm <= 0).sum()),
        "calendar_days_missing_total": int(per_city["missing_calendar_days"].sum()),
    }
    return summary, miss, per_city, by_year

# %% [markdown]
# ## 2. Leak-free features and target
# * Features of the row dated *t* use only data up to day *t* (past-only forward fill,
#   lags >= 0, trailing rolling means).
# * The target is PM2.5 on day *t+1*. Rows need an observed PM2.5 on day *t* and on day *t+1*, so that
#   persistence is defined and every method is scored on exactly the same rows.
# * `AQI` and `AQI_Bucket` are never used (AQI is computed from the same pollutants).

# %%
def build_features(daily):
    d = daily.sort_values(["City", "Date"]).reset_index(drop=True)
    cols = ["PM2.5"] + POLL
    ff = d.groupby("City")[cols].ffill(limit=FFILL_LIMIT)  # past-only by construction
    city = d["City"]
    X = pd.DataFrame({"City": city, "Date": d["Date"]})
    X["pm25_today"] = ff["PM2.5"]
    for p in POLL:
        X[p] = ff[p]
    pm = ff["PM2.5"]
    for k in range(1, 8):
        X[f"pm25_lag{k}"] = pm.groupby(city).shift(k)
    for w in (3, 7):
        X[f"pm25_roll{w}"] = pm.groupby(city).transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
    m, wd = d["Date"].dt.month, d["Date"].dt.dayofweek
    X["month_sin"], X["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    X["dow_sin"], X["dow_cos"] = np.sin(2 * np.pi * wd / 7), np.cos(2 * np.pi * wd / 7)
    X["y_today"] = d["PM2.5"]                              # raw observed PM2.5 today
    X["y"] = d.groupby("City")["PM2.5"].shift(-1)          # raw observed PM2.5 tomorrow (target)
    X["target_date"] = d["Date"] + pd.Timedelta(days=1)
    return X


def make_dataset(daily):
    F = build_features(daily)
    return F.dropna(subset=["y_today", "y"]).reset_index(drop=True)


def split(F, train_end=TRAIN_END, val_end=VAL_END, test_end=TEST_END):
    """Chronological split by TARGET date, so no target date falls in two splits."""
    td = F["target_date"]
    tr = F[td <= pd.Timestamp(train_end)]
    va = F[(td > pd.Timestamp(train_end)) & (td <= pd.Timestamp(val_end))]
    te = F[(td > pd.Timestamp(val_end)) & (td <= pd.Timestamp(test_end))]
    return tr.copy(), va.copy(), te.copy()


def select_numeric(train, cols, thr=MAX_MISSING_TRAIN):
    """Drop numeric features that are mostly missing IN TRAINING. Learned from train only."""
    num = [c for c in cols if c != "City"]
    miss = train[num].isna().mean()
    kept = [c for c in num if miss[c] <= thr]
    return kept, miss


def cols_for(train, set_name, kept_num=None):
    if kept_num is None:
        kept_num, _ = select_numeric(train, FEATURE_SETS[MAIN_SET])
    return [c for c in FEATURE_SETS[set_name] if c == "City" or c in kept_num]

# %% [markdown]
# ## 3. Baselines and models
# * **Persistence**: tomorrow = today.
# * **City-month mean**: mean target for the same city and calendar month, computed on training data only.
# * **Ridge, random forest, gradient boosting**, inside a pipeline whose imputer (train medians),
#   scaler (ridge only) and one-hot encoder are fit on training data only.

# %%
def persistence(df):
    return df["y_today"].to_numpy(dtype=float)


def city_month_fit(train):
    t = train.assign(m=train["target_date"].dt.month)
    return t.groupby(["City", "m"])["y"].mean(), t.groupby("m")["y"].mean(), float(train["y"].mean())


def city_month_predict(model, df):
    cm, mm, g = model
    m = df["target_date"].dt.month
    idx = pd.MultiIndex.from_arrays([df["City"], m])
    p = pd.Series(cm.reindex(idx).to_numpy(), index=df.index)
    return p.fillna(m.map(mm)).fillna(g).to_numpy(dtype=float)


def _ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipe(cols, model, scale):
    num = [c for c in cols if c != "City"]
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer([("num", Pipeline(steps), num), ("city", _ohe(), ["City"])])
    return Pipeline([("pre", pre), ("model", model)])


def make_model(name, hp, seed=SEED):
    if name == "ridge":
        return Ridge(**hp)
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, max_features=0.5, n_jobs=-1, random_state=seed, **hp)
    if name == "gb":
        return GradientBoostingRegressor(subsample=0.8, random_state=seed, **hp)
    raise ValueError(name)


def fit_model(name, hp, cols, train):
    pipe = build_pipe(cols, make_model(name, hp), scale=(name == "ridge"))
    return pipe.fit(train[cols], train["y"])


def tune(name, cols, train, val):
    """Grid search on the validation year only. Returns best hyperparameters and the full table."""
    rows = []
    for hp in GRIDS[name]:
        m = fit_model(name, hp, cols, train)
        rows.append({**hp, "val_mae": mae(val["y"].to_numpy(), m.predict(val[cols]))})
    tab = pd.DataFrame(rows)
    best = GRIDS[name][int(tab["val_mae"].idxmin())]
    return best, tab

# %% [markdown]
# ## 4. Metrics and uncertainty of the comparison
# MAE and RMSE are in PM2.5 units (ug/m3). "High-pollution days" are days whose target is above the
# 90th percentile of the TRAINING targets. The skill score is `1 - MAE_model / MAE_persistence`
# (positive = better than persistence). Confidence intervals come from a moving-block bootstrap over
# test dates (blocks of 7 days, all cities of a date resampled together), because consecutive days
# are strongly correlated.

# %%
def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def _safe(a, b):
    return a / b if b > 0 else np.nan


def metrics(df, pred, hi_thr, ref=None):
    y = df["y"].to_numpy(float)
    e = np.asarray(pred, float) - y
    hi = y >= hi_thr
    out = {"n": int(len(y)), "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
           "n_high": int(hi.sum()), "MAE_high": float(np.mean(np.abs(e[hi]))) if hi.any() else np.nan,
           "bias": float(np.mean(e))}
    if ref is not None:
        out["skill"] = 1 - out["MAE"] / float(np.mean(np.abs(np.asarray(ref, float) - y)))
    return out


def block_bootstrap(dates, arrays, stat_fns, B=NBOOT, block=BLOCK, seed=SEED, level=0.95):
    """Moving-block bootstrap over dates. `arrays` are per-row values, summed per date."""
    u, inv = np.unique(np.asarray(dates), return_inverse=True)
    D = len(u)
    block = min(block, D)
    agg = {k: np.bincount(inv, weights=np.asarray(v, float), minlength=D) for k, v in arrays.items()}
    agg["n"] = np.bincount(inv, minlength=D).astype(float)
    tot = {k: v.sum() for k, v in agg.items()}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(D / block))
    draws = {k: np.empty(B) for k in stat_fns}
    for b in range(B):
        starts = rng.integers(0, D - block + 1, size=nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:D]
        s = {k: v[idx].sum() for k, v in agg.items()}
        for name, fn in stat_fns.items():
            draws[name][b] = fn(s)
    out = {}
    for name, fn in stat_fns.items():
        lo, hi = np.nanquantile(draws[name], [(1 - level) / 2, 1 - (1 - level) / 2])
        out[name] = (float(fn(tot)), float(lo), float(hi))
    return out


COMPARE_STATS = {
    "MAE": lambda s: _safe(s["am"], s["n"]),
    "skill": lambda s: 1 - _safe(s["am"], s["ap"]),
    "dMAE": lambda s: _safe(s["am"] - s["ap"], s["n"]),
    "skill_high": lambda s: 1 - _safe(s["amh"], s["aph"]),
}


def boot_vs_persistence(df, pred, hi_thr):
    y = df["y"].to_numpy(float)
    am, ap = np.abs(np.asarray(pred, float) - y), np.abs(persistence(df) - y)
    hi = (y >= hi_thr).astype(float)
    return block_bootstrap(df["target_date"].to_numpy(),
                           {"am": am, "ap": ap, "amh": am * hi, "aph": ap * hi}, COMPARE_STATS)


def evaluate(df, preds, hi_thr, with_ci=True):
    """One row per model. `preds` must contain 'persistence' (the reference)."""
    rows = []
    for name, p in preds.items():
        r = {"model": name, **metrics(df, p, hi_thr, ref=preds["persistence"])}
        if with_ci:
            ci = boot_vs_persistence(df, p, hi_thr)
            for k in ("MAE", "skill", "dMAE", "skill_high"):
                r[k + "_lo"], r[k + "_hi"] = ci[k][1], ci[k][2]
        rows.append(r)
    return pd.DataFrame(rows)

# %% [markdown]
# ## 5. Error and failure-mode analysis helpers

# %%
def group_errors(df, pred, groups):
    """Bias, MAE and under-prediction share for named boolean masks (a dict name -> mask)."""
    y = df["y"].to_numpy(float)
    p = np.asarray(pred, float)
    e = p - y
    rows = []
    for name, mask in groups.items():
        mask = np.asarray(mask, bool)
        if mask.sum() == 0:
            continue
        rows.append({"group": name, "n": int(mask.sum()), "MAE": float(np.mean(np.abs(e[mask]))),
                     "bias": float(np.mean(e[mask])),
                     "share_underpredicted": float(np.mean(e[mask] < 0)),
                     "pred_over_obs": float(p[mask].mean() / y[mask].mean())})
    return pd.DataFrame(rows)


def per_city_table(df, pred, ref):
    t = df[["City", "y"]].copy()
    t["am"], t["ap"] = np.abs(np.asarray(pred, float) - t["y"]), np.abs(np.asarray(ref, float) - t["y"])
    g = t.groupby("City").agg(n=("y", "size"), MAE_model=("am", "mean"), MAE_persistence=("ap", "mean"))
    g["skill"] = 1 - g["MAE_model"] / g["MAE_persistence"]
    return g.reset_index()

# %% [markdown]
# ## 6. Prediction intervals: quantile gradient boosting and split-conformal
# * **Quantile gradient boosting** fits the 5% and 95% conditional quantiles directly (no calibration).
# * **Split-conformal** uses absolute residuals of the point model on a calibration year and returns
#   `prediction +/- q` where `q` is the `ceil((n+1)(1-alpha))/n` empirical quantile.
#   Its coverage guarantee assumes exchangeability, which a time series does not satisfy, so coverage is
#   measured on later periods (rolling windows and the 2020 lockdown) instead of being assumed.

# %%
def conformal_q(residuals, alpha=ALPHA):
    r = np.sort(np.asarray(residuals, float))
    n = len(r)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(r[min(k, n) - 1])


def fit_quantile_gb(hp, cols, train, q):
    model = GradientBoostingRegressor(loss="quantile", alpha=q, subsample=0.8, random_state=SEED, **hp)
    return build_pipe(cols, model, scale=False).fit(train[cols], train["y"])


WINDOWS = [  # (label, train_end, calibration_end, test_end): train < calibration < test, all by target date
    ("2018 backtest", "2016-12-31", "2017-12-31", "2018-12-31"),
    ("2019 backtest", "2017-12-31", "2018-12-31", "2019-12-31"),
    ("2020 test", "2018-12-31", "2019-12-31", "2020-07-01"),
]


def coverage_rows(window, method, df, lo, hi, hi_thr, extra_groups=None):
    y = df["y"].to_numpy(float)
    covered = ((y >= lo) & (y <= hi)).astype(float)
    width = np.clip(hi, 0, None) - np.clip(lo, 0, None)
    high = y >= hi_thr
    groups = {"all": np.ones(len(y), bool), "high-pollution days": high}
    groups.update(extra_groups or {})
    rows = []
    for gname, mask in groups.items():
        if mask.sum() == 0:
            continue
        r = {"window": window, "method": method, "group": gname, "n": int(mask.sum()),
             "coverage": float(covered[mask].mean()), "mean_width": float(width[mask].mean())}
        if gname == "all":
            ci = block_bootstrap(df["target_date"].to_numpy(), {"c": covered},
                                 {"cov": lambda s: _safe(s["c"], s["n"])}, B=1000)["cov"]
            r["coverage_lo"], r["coverage_hi"] = ci[1], ci[2]
        rows.append(r)
    return rows

# %% [markdown]
# ## 7. Plain-text helpers for saving results and LaTeX tables

# %%
def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "seed": SEED}


def fmt(x, nd=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"
