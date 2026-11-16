"""
Build the distribution + geometric-fit CSVs from raw nonproductive counts.

For CDP and CHS, reads {material}_nonproductive_counts.csv (one row per polyanion-
pair residence interval tau_{P-P}: the number N of nonproductive O-pair switches
that occurred inside it), and writes:

  {mat}_nonproductive_distribution.csv  the plotted distribution  (temperature_K, N_nonproductive, count, probability)
  {mat}_geometric_fit.csv               the geometric fit per T   (temperature_K, p, R2, slope, intercept, count_threshold, n_events)

The distribution P(N) is fit (log-linear, on N-values whose occurrence count >= 10)
to a geometric law P(N) = p (1-p)^N, giving the per-attempt productive-rotation
probability p = 1 - exp(slope).

Generate plot inputs from the bundled count tables, then plot::

    python figures/make_nonproductive_csv.py --out results/reference/nonproductive
    SUPERPROTON_REFERENCE_DIR=results/reference python figures/plot_nonproductive.py

For your own count tables, add ``--raw results/raw``. ``--out`` is required;
use a separate output directory to preserve the bundled reference data.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import each_material, ref  # noqa: E402


import numpy as np
import pandas as pd
from scipy import stats

COUNT_THRESHOLD = 10  # fit N-values whose occurrence count >= this (the "count_10" figure)


def build(material, raw_dir, out_dir):
    counts = pd.read_csv(raw_dir / f"{material}_nonproductive_counts.csv")
    dist_rows, fit_rows = [], []

    for temp in sorted(counts["temperature_K"].unique()):
        per_interval = counts.loc[counts["temperature_K"] == temp, "N_nonproductive"].to_numpy()
        n_values, occ = np.unique(per_interval, return_counts=True)
        prob = occ / len(per_interval)
        for nv, oc, pr in zip(n_values, occ, prob):
            dist_rows.append({"temperature_K": temp, "N_nonproductive": int(nv),
                              "count": int(oc), "probability": pr})

        mask = occ >= COUNT_THRESHOLD
        if mask.sum() < 2:
            fit_rows.append({"temperature_K": temp, "p": np.nan, "R2": np.nan,
                             "slope": np.nan, "intercept": np.nan,
                             "count_threshold": COUNT_THRESHOLD, "n_events": len(per_interval)})
            continue
        slope, intercept, r, _, _ = stats.linregress(n_values[mask], np.log(prob[mask]))
        p = 1 - np.exp(slope)
        fit_rows.append({"temperature_K": temp, "p": p, "R2": r ** 2,
                         "slope": slope, "intercept": intercept,
                         "count_threshold": COUNT_THRESHOLD, "n_events": len(per_interval)})
        print(f"  {material} {temp}K: p={p:.3f}, R2={r**2:.3f}, n_events={len(per_interval)}")

    pd.DataFrame(dist_rows).to_csv(out_dir / f"{material}_nonproductive_distribution.csv",
                                   index=False, float_format="%.6e")
    pd.DataFrame(fit_rows).to_csv(out_dir / f"{material}_geometric_fit.csv",
                                  index=False, float_format="%.6f")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=None,
                   help="Directory holding the raw *_nonproductive_counts.csv (default: data/reference/nonproductive).")
    p.add_argument("--out", type=Path, required=True,
                   help="Directory for the derived CSVs. Required, so a "
                        "reproducibility check cannot overwrite data/reference/.")
    p.add_argument("--material", choices=["CDP", "CHS"], action="append",
                   help="Restrict to one material; repeatable. Default: both.")
    args = p.parse_args(argv)

    # --raw defaults to the shipped reference data so the check is one command;
    # --out is required so the check cannot overwrite what it is verifying.
    raw_dir = args.raw if args.raw is not None else ref("nonproductive")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    each_material(args.material or ("CDP", "CHS"),
                  lambda material: build(material, raw_dir, out_dir),
                  what="CSVs", where=raw_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
