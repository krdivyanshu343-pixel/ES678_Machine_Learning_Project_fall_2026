# %% [markdown]
# # Run the full experiment
# Executes the pipeline end to end: data audit -> chronological split -> leak-free features ->
# baselines and models -> tuning on 2019 -> single evaluation on 2020 -> graduate-depth analyses
# (lag ablation, failure audit, prediction intervals) -> figures and LaTeX tables.

# %%
import sys; from pathlib import Path  # script-only
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # script-only
from aq_pipeline import *  # script-only
import time

T0 = time.time()
RESULTS = ROOT / "results"
TABLES = ROOT / "paper" / "tables"
FIGS = ROOT / "paper" / "figures"
for p in (RESULTS, TABLES, FIGS):
    p.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)
SUMMARY = {"versions": versions()}

# %% [markdown]
# ## 1. Load data and audit quality

# %%
DATA = find_data()
raw = load_raw(DATA)
daily = to_daily(raw)
a_sum, a_miss, a_city, a_year = audit(raw, daily)
SUMMARY["data_audit"] = a_sum
print(json.dumps(a_sum, indent=2))
print("\nMissing % per column:\n", a_miss.to_string())
a_miss.to_csv(RESULTS / "audit_missing_pct.csv")
a_city.to_csv(RESULTS / "audit_per_city.csv")
a_year.to_csv(RESULTS / "audit_pm25_counts_city_year.csv")

# %% [markdown]
# ## 2. Dataset, city filter and chronological split
# Cities with fewer than `MIN_TRAIN_ROWS` usable training rows are dropped (this uses training data only).

# %%
F_all = make_dataset(daily)
tr0, _, _ = split(F_all)
counts = tr0["City"].value_counts()
KEEP = sorted(counts[counts >= MIN_TRAIN_ROWS].index)
DROPPED = sorted(set(F_all["City"].unique()) - set(KEEP))
F = F_all[F_all["City"].isin(KEEP)].reset_index(drop=True)
train, val, test = split(F)

# Hard checks that the split is chronological and non-overlapping.
assert train["target_date"].max() <= pd.Timestamp(TRAIN_END)
assert val["target_date"].min() > train["target_date"].max()
assert test["target_date"].min() > val["target_date"].max()

HI = float(np.quantile(train["y"], HIGH_Q))
LOCK = pd.Timestamp(LOCKDOWN_START)
pre_mask = (test["target_date"] < LOCK).to_numpy()
lock_mask = ~pre_mask

city_rows = pd.DataFrame({
    "train_rows": train.groupby("City").size(), "val_rows": val.groupby("City").size(),
    "test_rows": test.groupby("City").size()}).fillna(0).astype(int)
city_rows.to_csv(RESULTS / "kept_cities_rows.csv")
SUMMARY["split"] = {
    "rows_before_filter": int(len(F_all)), "cities_kept": KEEP, "cities_dropped": DROPPED,
    "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
    "test_pre_lockdown_rows": int(pre_mask.sum()), "test_lockdown_rows": int(lock_mask.sum()),
    "high_threshold_ugm3": HI, "high_days_test": int((test["y"] >= HI).sum()),
    "train_target_dates": [str(train["target_date"].min().date()), str(train["target_date"].max().date())],
    "val_target_dates": [str(val["target_date"].min().date()), str(val["target_date"].max().date())],
    "test_target_dates": [str(test["target_date"].min().date()), str(test["target_date"].max().date())],
}
print(f"kept {len(KEEP)} cities, dropped {len(DROPPED)}: {DROPPED}")
print(f"rows: train {len(train)}, val {len(val)}, test {len(test)} (pre-lockdown {pre_mask.sum()}, lockdown {lock_mask.sum()})")
print(f"high-pollution threshold (train 90th percentile): {HI:.1f} ug/m3; high days in test: {(test['y'] >= HI).sum()}")
print(city_rows.to_string())

# %% [markdown]
# ## 3. Baselines, tuning on validation (2019), final evaluation on test (2020)

# %%
KEPT_NUM, miss_train = select_numeric(train, FEATURE_SETS[MAIN_SET])
DROPPED_FEATURES = [c for c in miss_train.index if c not in KEPT_NUM]
COLS = cols_for(train, MAIN_SET, KEPT_NUM)
SUMMARY["features"] = {"kept_numeric": KEPT_NUM, "dropped_for_missingness_in_train": DROPPED_FEATURES,
                       "train_missing_share": miss_train.round(3).to_dict()}
print("dropped features (missing share in train > %.0f%%):" % (MAX_MISSING_TRAIN * 100), DROPPED_FEATURES)

cm = city_month_fit(train)
BEST_HP, TUNING = {}, {}
for name in MODEL_NAMES:
    BEST_HP[name], TUNING[name] = tune(name, COLS, train, val)
    TUNING[name].to_csv(RESULTS / f"tuning_{name}.csv", index=False)
    print(f"{name}: best {BEST_HP[name]}  val MAE {TUNING[name]['val_mae'].min():.2f}")
FITTED = {n: fit_model(n, BEST_HP[n], COLS, train) for n in MODEL_NAMES}


def predict_all(df):
    p = {"persistence": persistence(df), "city-month": city_month_predict(cm, df)}
    for n in MODEL_NAMES:
        p[n] = FITTED[n].predict(df[COLS])
    return p


P_val, P_test = predict_all(val), predict_all(test)
pred_out = test[["City", "target_date", "y_today", "y"]].reset_index(drop=True).copy()
for k, v in P_test.items():
    pred_out[f"pred_{k}"] = v
pred_out.to_csv(RESULTS / "test_predictions.csv", index=False)
val_tab = evaluate(val, P_val, HI, with_ci=False)
BEST = val_tab[val_tab["model"].isin(MODEL_NAMES)].sort_values("MAE").iloc[0]["model"]
print("\nVALIDATION 2019\n", val_tab.round(3).to_string(index=False))
print("\nbest learned model on validation:", BEST)

# The test set is evaluated here, once, after models and hyperparameters are frozen.
test_tab = evaluate(test, P_test, HI, with_ci=True)
pre_tab = evaluate(test[pre_mask], {k: v[pre_mask] for k, v in P_test.items()}, HI, with_ci=True)
lock_tab = evaluate(test[lock_mask], {k: v[lock_mask] for k, v in P_test.items()}, HI, with_ci=True)
test_tab.assign(period="all 2020 test").to_csv(RESULTS / "test_metrics_all.csv", index=False)
pre_tab.assign(period="pre-lockdown").to_csv(RESULTS / "test_metrics_pre_lockdown.csv", index=False)
lock_tab.assign(period="lockdown").to_csv(RESULTS / "test_metrics_lockdown.csv", index=False)
val_tab.to_csv(RESULTS / "val_metrics.csv", index=False)
print("\nTEST 2020 (all)\n", test_tab.round(3).to_string(index=False))
print("\nTEST pre-lockdown\n", pre_tab[["model", "n", "MAE", "skill", "skill_lo", "skill_hi"]].round(3).to_string(index=False))
print("\nTEST lockdown\n", lock_tab[["model", "n", "MAE", "skill", "skill_lo", "skill_hi"]].round(3).to_string(index=False))
SUMMARY["best_hyperparameters"] = BEST_HP
SUMMARY["best_model_on_validation"] = BEST

# %% [markdown]
# ## 4. Stability check: rolling-origin validation inside the training period
# Train on all earlier years, validate on the next year, for 2017 and 2018, using the selected hyperparameters.

# %%
ro = []
for label, tr_end, va_end in [("2017", "2016-12-31", "2017-12-31"), ("2018", "2017-12-31", "2018-12-31")]:
    ftr, fva, _ = split(F, tr_end, va_end, va_end)
    kn, _ = select_numeric(ftr, FEATURE_SETS[MAIN_SET])
    cols = cols_for(ftr, MAIN_SET, kn)
    hi_f = float(np.quantile(ftr["y"], HIGH_Q))
    preds = {"persistence": persistence(fva), "city-month": city_month_predict(city_month_fit(ftr), fva)}
    for n in MODEL_NAMES:
        preds[n] = fit_model(n, BEST_HP[n], cols, ftr).predict(fva[cols])
    t = evaluate(fva, preds, hi_f, with_ci=False)
    t.insert(0, "fold", f"train<= {tr_end[:4]}, validate {label}")
    ro.append(t)
ro = pd.concat(ro, ignore_index=True)
ro.to_csv(RESULTS / "rolling_origin.csv", index=False)
print(ro[["fold", "model", "n", "MAE", "RMSE", "skill"]].round(3).to_string(index=False))

# %% [markdown]
# ## 5. Graduate depth analysis 1: lag-feature ablation
# Each feature set is re-tuned on 2019 for each model family, then scored once on the 2020 test.

# %%
abl = []
for sname in FEATURE_SETS:
    cols = cols_for(train, sname, KEPT_NUM)
    for n in MODEL_NAMES:
        if sname == MAIN_SET:
            hp, model, ptest = BEST_HP[n], FITTED[n], P_test[n]
            vmae = float(TUNING[n]["val_mae"].min())
        else:
            hp, tab = tune(n, cols, train, val)
            model = fit_model(n, hp, cols, train)
            ptest, vmae = model.predict(test[cols]), float(tab["val_mae"].min())
        ci = boot_vs_persistence(test, ptest, HI)
        abl.append({"feature_set": sname, "model": n, "n_features": len(cols), "val_MAE": vmae,
                    "test_MAE": mae(test["y"], ptest), "test_skill": ci["skill"][0],
                    "skill_lo": ci["skill"][1], "skill_hi": ci["skill"][2], "hyperparameters": str(hp)})
abl = pd.DataFrame(abl)
abl.to_csv(RESULTS / "ablation_lags.csv", index=False)
print(abl.drop(columns="hyperparameters").round(3).to_string(index=False))

# %% [markdown]
# ## 6. Graduate depth analysis 2: failure-mode audit
# Where does the best model fail? High-pollution days, winter episodes, particular months and cities.

# %%
pb, pp = P_test[BEST], P_test["persistence"]
tm = test["target_date"].dt.month.to_numpy()
hi_t = (test["y"] >= HI).to_numpy()
groups_test = {"all test days": np.ones(len(test), bool), "high-pollution days": hi_t, "other days": ~hi_t,
               "Jan-Feb 2020": np.isin(tm, [1, 2]), "lockdown period": lock_mask}
fail = pd.concat([group_errors(test, pb, groups_test).assign(model=BEST, split="test 2020"),
                  group_errors(test, pp, groups_test).assign(model="persistence", split="test 2020")])
tv = val["target_date"].dt.month.to_numpy()
hi_v = (val["y"] >= HI).to_numpy()
groups_val = {"all validation days": np.ones(len(val), bool), "high-pollution days": hi_v,
              "other days": ~hi_v, "Nov-Dec 2019 (winter)": np.isin(tv, [11, 12])}
pv, ppv = P_val[BEST], P_val["persistence"]
fail = pd.concat([fail, group_errors(val, pv, groups_val).assign(model=BEST, split="validation 2019"),
                  group_errors(val, ppv, groups_val).assign(model="persistence", split="validation 2019")])
fail.to_csv(RESULTS / "failure_audit_groups.csv", index=False)
print(fail.round(3).to_string(index=False))

month_rows = []
for m in sorted(set(tm)):
    mask = tm == m
    month_rows.append({"month": int(m), "n": int(mask.sum()), "MAE_model": mae(test["y"][mask], pb[mask]),
                       "MAE_persistence": mae(test["y"][mask], pp[mask])})
month_tab = pd.DataFrame(month_rows)
month_tab.to_csv(RESULTS / "failure_audit_by_month.csv", index=False)

city_test = per_city_table(test, pb, pp)
city_val = per_city_table(val, pv, ppv)
city_test.to_csv(RESULTS / "per_city_test.csv", index=False)
city_val.to_csv(RESULTS / "per_city_val.csv", index=False)
print("\nper-city test skill vs persistence:\n", city_test.round(3).to_string(index=False))
SUMMARY["per_city"] = {"test_cities_better_than_persistence": int((city_test["skill"] > 0).sum()),
                       "test_cities": int(len(city_test)),
                       "val_cities_better_than_persistence": int((city_val["skill"] > 0).sum()),
                       "val_cities": int(len(city_val))}

worst = test[["City", "target_date", "y"]].copy()
worst["pred"], worst["persistence"] = pb, pp
worst["abs_err"] = np.abs(worst["pred"] - worst["y"])
worst.sort_values("abs_err", ascending=False).head(15).to_csv(RESULTS / "failure_audit_worst_days.csv", index=False)
print("\nlargest test errors:\n", worst.sort_values("abs_err", ascending=False).head(8).to_string(index=False, float_format=lambda v: f"{v:.1f}"))

# %% [markdown]
# ## 7. Graduate depth analysis 3: prediction intervals and their coverage
# 90% intervals from (a) split-conformal around the best point model and (b) quantile gradient boosting
# (5% and 95% quantiles). Windows: train < calibration year < test year, all chronological.

# %%
cov_rows, CONF_Q = [], {}
for label, tr_end, cal_end, te_end in WINDOWS:
    wtr, wcal, wte = split(F, tr_end, cal_end, te_end)
    kn, _ = select_numeric(wtr, FEATURE_SETS[MAIN_SET])
    cols = cols_for(wtr, MAIN_SET, kn)
    hi_w = float(np.quantile(wtr["y"], HIGH_Q))
    point = fit_model(BEST, BEST_HP[BEST], cols, wtr)
    q = conformal_q(np.abs(wcal["y"].to_numpy() - point.predict(wcal[cols])))
    CONF_Q[label] = q
    yhat = point.predict(wte[cols])
    qlo = fit_quantile_gb(BEST_HP["gb"], cols, wtr, ALPHA / 2).predict(wte[cols])
    qhi = fit_quantile_gb(BEST_HP["gb"], cols, wtr, 1 - ALPHA / 2).predict(wte[cols])
    extra = {}
    if label == "2020 test":
        pre_w = (wte["target_date"] < LOCK).to_numpy()
        extra = {"pre-lockdown": pre_w, "lockdown": ~pre_w}
    cov_rows += coverage_rows(label, "split-conformal", wte, yhat - q, yhat + q, hi_w, extra)
    cov_rows += coverage_rows(label, "quantile GB", wte, np.minimum(qlo, qhi), np.maximum(qlo, qhi), hi_w, extra)
cov = pd.DataFrame(cov_rows)
cov.to_csv(RESULTS / "interval_coverage.csv", index=False)
print("conformal half-widths q (ug/m3):", {k: round(v, 1) for k, v in CONF_Q.items()})
print(cov.round(3).to_string(index=False))
SUMMARY["conformal_halfwidth"] = CONF_Q
SUMMARY["nominal_coverage"] = 1 - ALPHA

# %% [markdown]
# ## 8. Figures

# %%
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7})
UNIT = r"$\mu$g/m$^3$"

# Figure 1: (a) observed vs predicted on the test set, (b) MAE by month vs persistence.
y, p = test["y"].to_numpy(), pb
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.45), gridspec_kw={"width_ratios": [1, 1.1]})
ax[0].scatter(y[~hi_t], p[~hi_t], s=3, alpha=0.35, c="tab:blue", label="other days", rasterized=True)
ax[0].scatter(y[hi_t], p[hi_t], s=6, alpha=0.7, c="tab:red", label="high-pollution days", rasterized=True)
lim = float(max(y.max(), p.max()) * 1.02)
ax[0].plot([0, lim], [0, lim], "k--", lw=0.8)
ax[0].set_xlabel(f"observed next-day PM2.5 ({UNIT})")
ax[0].set_ylabel(f"predicted ({UNIT})")
ax[0].set_title(f"(a) {MODEL_LABELS[BEST]}, 2020 test")
ax[0].legend(loc="upper left", frameon=False, markerscale=2)
mt = month_tab[month_tab["month"] <= 6]
xs = np.arange(len(mt))
ax[1].bar(xs - 0.2, mt["MAE_persistence"], 0.4, label="persistence", color="tab:gray")
ax[1].bar(xs + 0.2, mt["MAE_model"], 0.4, label=MODEL_LABELS[BEST], color="tab:blue")
ax[1].set_xticks(xs)
ax[1].set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun"][: len(mt)])
ax[1].axvline(2.5, color="k", ls=":", lw=0.8)
ax[1].set_ylabel(f"MAE ({UNIT})")
ax[1].set_title("(b) MAE by target month (dotted: ~lockdown start)")
ax[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGS / "fig_errors.pdf")
fig.savefig(FIGS / "fig_errors.png", dpi=200)

# Figure 2: empirical coverage of the nominal 90% intervals.
def _get(window, group, method):
    r = cov[(cov["window"] == window) & (cov["group"] == group) & (cov["method"] == method)]
    return float(r["coverage"].iloc[0]) if len(r) else np.nan


cats_a = [("2018 backtest", "all", "2018"), ("2019 backtest", "all", "2019"),
          ("2020 test", "pre-lockdown", "2020\npre-lockdown"), ("2020 test", "lockdown", "2020\nlockdown")]
cats_b = [("2018 backtest", "high-pollution days", "2018"), ("2019 backtest", "high-pollution days", "2019"),
          ("2020 test", "high-pollution days", "2020")]
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw={"width_ratios": [1.3, 1]})
for a, cats, title in ((ax[0], cats_a, "(a) all days"), (ax[1], cats_b, "(b) high-pollution days")):
    xs = np.arange(len(cats))
    a.bar(xs - 0.2, [_get(w, g, "split-conformal") for w, g, _ in cats], 0.4, label="split-conformal", color="tab:blue")
    a.bar(xs + 0.2, [_get(w, g, "quantile GB") for w, g, _ in cats], 0.4, label="quantile GB", color="tab:orange")
    a.axhline(1 - ALPHA, color="k", ls="--", lw=0.8)
    a.set_xticks(xs)
    a.set_xticklabels([c[2] for c in cats])
    a.set_ylim(0, 1.25)
    a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a.set_title(title)
ax[0].set_ylabel("empirical coverage")
ax[0].legend(frameon=False, loc="upper center", ncol=2)
fig.tight_layout()
fig.savefig(FIGS / "fig_coverage.pdf")
fig.savefig(FIGS / "fig_coverage.png", dpi=200)
if "ipykernel" in sys.modules:
    plt.show()
plt.close("all")

# %% [markdown]
# ## 9. LaTeX tables for the paper (generated from the saved results)

# %%
def _row(tab, name):
    return tab[tab["model"] == name].iloc[0]


best_mae = min(float(_row(test_tab, n)["MAE"]) for n in MODEL_NAMES)
lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Model & MAE & RMSE & MAE$_{\mathrm{high}}$ & Skill [95\% CI] & MAE$_{\mathrm{pre}}$ & MAE$_{\mathrm{lock}}$ \\",
         r"\midrule"]
for n in ["persistence", "city-month"] + MODEL_NAMES:
    r, rp, rl = _row(test_tab, n), _row(pre_tab, n), _row(lock_tab, n)
    mae_s = fmt(r["MAE"])
    if n in MODEL_NAMES and abs(float(r["MAE"]) - best_mae) < 1e-9:
        mae_s = r"\textbf{" + mae_s + "}"
    skill_s = "0 (ref.)" if n == "persistence" else f"{r['skill']:+.3f} [{r['skill_lo']:+.3f}, {r['skill_hi']:+.3f}]"
    lines.append(f"{MODEL_LABELS[n]} & {mae_s} & {fmt(r['RMSE'])} & {fmt(r['MAE_high'])} & {skill_s} & "
                 f"{fmt(rp['MAE'])} & {fmt(rl['MAE'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "table_main.tex").write_text("\n".join(lines) + "\n")

lines = [r"\begin{tabular}{lccc}", r"\toprule",
         r"Features & Ridge & Random forest & Grad.\ boosting \\", r"\midrule"]
for sname in FEATURE_SETS:
    cells = []
    for n in MODEL_NAMES:
        r = abl[(abl["feature_set"] == sname) & (abl["model"] == n)].iloc[0]
        cells.append(f"{r['test_MAE']:.1f} ({r['test_skill']:+.3f})")
    lines.append(f"{sname} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "table_ablation.tex").write_text("\n".join(lines) + "\n")

def _cell(window, group, method, col):
    r = cov[(cov["window"] == window) & (cov["group"] == group) & (cov["method"] == method)]
    return float(r[col].iloc[0]) if len(r) else np.nan


rows_spec = [("2018 backtest", "2018 backtest", "all"), ("2019 backtest", "2019 backtest", "all"),
             ("2020 test", "2020 test (all)", "all"), ("2020 test", "\\ \\ pre-lockdown", "pre-lockdown"),
             ("2020 test", "\\ \\ lockdown", "lockdown")]
lines = [r"\begin{tabular}{lrcccccc}", r"\toprule",
         r" & & \multicolumn{2}{c}{Split-conformal} & \multicolumn{2}{c}{Quantile GB} & \multicolumn{2}{c}{Coverage, high days} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Period & $n$ & Cov. & Width & Cov. & Width & Conf. & QGB \\", r"\midrule"]
for window, label, group in rows_spec:
    n_ = int(_cell(window, group, "split-conformal", "n"))
    hi_cells = ("--", "--") if group != "all" and window == "2020 test" else (
        fmt(_cell(window, "high-pollution days", "split-conformal", "coverage"), 2),
        fmt(_cell(window, "high-pollution days", "quantile GB", "coverage"), 2))
    if label == "2020 test (all)":
        hi_cells = (fmt(_cell(window, "high-pollution days", "split-conformal", "coverage"), 2),
                    fmt(_cell(window, "high-pollution days", "quantile GB", "coverage"), 2))
    lines.append(f"{label} & {n_} & {fmt(_cell(window, group, 'split-conformal', 'coverage'), 2)} & "
                 f"{fmt(_cell(window, group, 'split-conformal', 'mean_width'), 0)} & "
                 f"{fmt(_cell(window, group, 'quantile GB', 'coverage'), 2)} & "
                 f"{fmt(_cell(window, group, 'quantile GB', 'mean_width'), 0)} & {hi_cells[0]} & {hi_cells[1]} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "table_intervals.tex").write_text("\n".join(lines) + "\n")

SUMMARY["runtime_seconds"] = round(time.time() - T0, 1)
SUMMARY["test_metrics_all"] = test_tab.round(4).to_dict(orient="records")
save_json(SUMMARY, RESULTS / "run_summary.json")
print(f"\nDone in {SUMMARY['runtime_seconds']} s. Results in {RESULTS}, paper tables in {TABLES}.")
