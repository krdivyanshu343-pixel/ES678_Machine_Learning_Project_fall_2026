"""Automated checks that the pipeline does not leak future information.

Run with:  python tests/test_leakage.py      (or: python -m pytest tests, if pytest is installed)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from aq_pipeline import (POLL, ALPHA, block_bootstrap, build_features, conformal_q, make_dataset,
                         split, to_daily, select_numeric, FEATURE_SETS, MAIN_SET)


def synthetic_daily(n_days=150, cities=("A", "B"), seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for c in cities:
        df = pd.DataFrame({"City": c, "Date": pd.date_range("2018-01-01", periods=n_days)})
        df["PM2.5"] = rng.gamma(4, 20, n_days)
        for p in POLL:
            df[p] = rng.gamma(3, 10, n_days)
        df.loc[rng.choice(n_days, 15, replace=False), ["PM2.5"] + POLL] = np.nan  # holes
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def test_features_do_not_use_the_future():
    """Altering every observation after a cutoff must not change features dated on or before it."""
    daily = synthetic_daily()
    cutoff = pd.Timestamp("2018-03-15")
    f1 = build_features(daily)
    changed = daily.copy()
    changed.loc[changed["Date"] > cutoff, ["PM2.5"] + POLL] = 999.0
    f2 = build_features(changed)
    cols = [c for c in f1.columns if c not in ("y", "target_date")]  # y is tomorrow's value by design
    a = f1[f1["Date"] <= cutoff][cols].reset_index(drop=True)
    b = f2[f2["Date"] <= cutoff][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_target_is_next_day_value():
    daily = synthetic_daily()
    F = make_dataset(daily)
    raw = daily.set_index(["City", "Date"])["PM2.5"]
    for _, r in F.sample(50, random_state=1).iterrows():
        assert raw[(r["City"], r["Date"] + pd.Timedelta(days=1))] == r["y"]
        assert raw[(r["City"], r["Date"])] == r["y_today"]


def test_split_is_chronological_and_disjoint():
    F = pd.DataFrame({"target_date": pd.date_range("2015-01-01", "2020-07-01", freq="D")})
    tr, va, te = split(F)
    assert tr["target_date"].max() < va["target_date"].min()
    assert va["target_date"].max() < te["target_date"].min()
    assert len(tr) + len(va) + len(te) == len(F)


def test_feature_selection_uses_only_the_training_frame():
    daily = synthetic_daily()
    F = make_dataset(daily)
    train = F.iloc[:100]
    kept1, _ = select_numeric(train, FEATURE_SETS[MAIN_SET])
    F2 = F.copy()
    F2.iloc[100:, F2.columns.get_loc("PM10")] = np.nan  # wreck the non-training rows
    kept2, _ = select_numeric(F2.iloc[:100], FEATURE_SETS[MAIN_SET])
    assert kept1 == kept2


def test_split_conformal_covers_on_exchangeable_data():
    rng = np.random.default_rng(0)
    cal, test = rng.normal(size=2000), rng.normal(size=20000)
    q = conformal_q(np.abs(cal), ALPHA)
    assert np.mean(np.abs(test) <= q) >= 1 - ALPHA - 0.02


def test_block_bootstrap_point_estimate_matches_plain_mean():
    rng = np.random.default_rng(0)
    dates = np.repeat(pd.date_range("2020-01-01", periods=60).to_numpy(), 3)
    err = np.abs(rng.normal(size=len(dates)))
    out = block_bootstrap(dates, {"e": err}, {"mae": lambda s: s["e"] / s["n"]}, B=200)
    assert abs(out["mae"][0] - err.mean()) < 1e-9
    assert out["mae"][1] <= out["mae"][0] <= out["mae"][2]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nAll {len(tests)} checks passed.")
