"""Independent re-computation of the headline numbers from the saved test predictions.

Uses only numpy/pandas on results/test_predictions.csv (no pipeline code) and compares with
results/test_metrics_all.csv. Run:  python scripts/verify_headline.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
p = pd.read_csv(R / "test_predictions.csv")
m = pd.read_csv(R / "test_metrics_all.csv").set_index("model")
y = p["y"].to_numpy()
mae_pers = np.mean(np.abs(p["pred_persistence"] - y))
ok = True
print(f"{'model':<12} {'MAE (here)':>11} {'MAE (saved)':>12} {'skill (here)':>13} {'skill (saved)':>14}")
for name in ["persistence", "city-month", "ridge", "rf", "gb"]:
    mae = np.mean(np.abs(p[f"pred_{name}"] - y))
    skill = 1 - mae / mae_pers
    same = abs(mae - m.loc[name, "MAE"]) < 1e-6 and abs(skill - m.loc[name, "skill"]) < 1e-6
    ok &= same
    print(f"{name:<12} {mae:11.4f} {m.loc[name, 'MAE']:12.4f} {skill:13.4f} {m.loc[name, 'skill']:14.4f}  {'OK' if same else 'MISMATCH'}")
# persistence must literally equal today's observed value
assert np.allclose(p["pred_persistence"], p["y_today"])
print("\nrows:", len(p), "| date range:", p["target_date"].min(), "to", p["target_date"].max())
print("ALL HEADLINE NUMBERS REPRODUCED" if ok else "MISMATCH FOUND")
