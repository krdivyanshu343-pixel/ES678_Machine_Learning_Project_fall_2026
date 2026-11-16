"""Build notebooks/air_quality_colab.ipynb from src/aq_pipeline.py and scripts/run_all.py.

Cells are separated by lines starting with '# %%' ('# %% [markdown]' for text cells).
Lines tagged 'script-only' are dropped because they only matter when running as a script.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse(path):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if kind == "markdown":
            text = "\n".join(re.sub(r"^# ?", "", l) for l in text.splitlines())
        else:
            text = "\n".join(l for l in text.splitlines() if "script-only" not in l)
        if text.strip():
            cells.append((kind, text))

    for line in path.read_text().splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def cell(kind, text):
    src = text.splitlines(keepends=True)
    if src and src[-1].endswith("\n"):
        src[-1] = src[-1].rstrip("\n")
    base = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


intro = """# Air Quality Analysis and Prediction (ES 678): all code in one notebook

Next-day PM2.5 forecasting for Indian cities (CPCB data, 2015-2020). This notebook contains **every function and every experiment** of the project, in the order they run. Run all cells top to bottom (about 10 minutes on Colab's CPU).

**How to use it in Google Colab**
1. Open the notebook in Colab (Runtime > Run all).
2. When the data-loading cell asks, upload `city_day.csv` (download *Air Quality Data in India (2015-2020)* from Kaggle first; a free account is needed).
3. Outputs are written to `results/` and `paper/` inside the Colab session; download them from the Files panel if you want them.

**Playing around:** all settings are in the *Configuration* cell (split dates, city threshold, grids, interval level, bootstrap size). Change one, then re-run from that cell downwards. A small plotting cell at the very end lets you inspect any city."""

extra = """# Play around: compare observed PM2.5, persistence and the selected model for one city on the 2020 test set.
CITY = "Delhi"   # try any of: " ".join(KEEP)
m = (test["City"] == CITY).to_numpy()
d = test.loc[m, "target_date"]
plt.figure(figsize=(9, 3))
plt.plot(d, test.loc[m, "y"], label="observed", lw=1.5, color="black")
plt.plot(d, P_test["persistence"][m], label="persistence", lw=1, color="tab:gray")
plt.plot(d, P_test[BEST][m], label=MODEL_LABELS[BEST], lw=1, color="tab:blue")
plt.axvline(LOCK, color="red", ls=":", lw=1)
plt.ylabel("PM2.5 (ug/m3)"); plt.title(f"{CITY}: next-day PM2.5, 2020 test (red dotted: lockdown start)")
plt.legend(); plt.tight_layout(); plt.show()"""

cells = [("markdown", intro)]
cells += parse(ROOT / "src" / "aq_pipeline.py")
cells += parse(ROOT / "scripts" / "run_all.py")
cells += [("markdown", "## 10. Play around"), ("code", extra)]

nb = {
    "cells": [cell(k, t) for k, t in cells],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
out = ROOT / "notebooks" / "air_quality_colab.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} with {len(nb['cells'])} cells")
