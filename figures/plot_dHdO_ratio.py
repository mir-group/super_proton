"""
Counts vs. mean dH/dO displacement ratio, one figure per material (4 ns MLMD):
CDP (CsH2PO4) at 525 K and CHS (CsHSO4) at 450 K.

Three series per figure:
  baseline               - per-event mean dH/dO over the whole rotation period,
                           pooled over productive and nonproductive events
  outlier (productive)   - per-event mean over the instants with Z-score > 2,
                           for productive (polyanion-pair) rotations
  outlier (non-productive) - the same, for nonproductive rotations

The outlier peaks (~4-6) sit well above the baseline (~1.5): during a transfer
the proton moves far more than its carrier oxygen, which is the slingshot.

Reads the binned distributions shipped in ``data/reference/dhdo_ratio/``,
whose columns are ``series, bin_left, bin_right, bin_center, count_per_event``
(``count_per_event`` = histogram count / number of events, the plotted y).
The two outlier series share bin edges, auto-binned on the productive set. The
baseline uses a wider adaptive bin width chosen so its peak height is
comparable to the outlier peak height -- with a fixed narrow binning the
baseline peak is a visualization artifact. x is truncated at the 97th
percentile of the productive outliers.

To recompute the per-event ratios from a trajectory, see
``superproton.reorientation.compute_reorientation``.

Run:
    python figures/plot_dHdO_ratio.py
Produces: CDP_525K_dHdO_ratio.png, CHS_450K_dHdO_ratio.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import each_material, out, ref  # noqa: E402

DATA = ref("dhdo_ratio")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MATERIALS = {"CDP": 525, "CHS": 450}


def _percentile_from_bins(g, q):
    """Interpolate the ``q``-th percentile out of a binned count distribution."""
    cdf = np.cumsum(g["count_per_event"].to_numpy())
    cdf = cdf / cdf[-1]
    return float(np.interp(q / 100.0, cdf, g["bin_right"].to_numpy()))


def run(material, temp):
    d = pd.read_csv(DATA / f"{material}_{temp}K_dHdO_ratio.csv")
    base = d[d["series"] == "baseline"]
    np_out = d[d["series"] == "outlier_nonproductive"]
    p_out = d[d["series"] == "outlier_productive"]

    plt.figure(figsize=(7, 5))
    plt.rcParams["font.size"] = 16

    # baseline as filled bars at its own (wider) bin width; outliers as steps
    # sharing the productive bin edges.
    h_base = plt.bar(
        base["bin_left"], base["count_per_event"],
        width=(base["bin_right"] - base["bin_left"]).to_numpy(),
        align="edge", alpha=0.5, edgecolor="none", color="grey", label="baseline",
    )
    h_np = plt.step(np_out["bin_left"], np_out["count_per_event"], linewidth=2,
                    label="outlier (non-productive)")[0]
    h_p = plt.step(p_out["bin_left"], p_out["count_per_event"], linewidth=3,
                   label="outlier (productive)")[0]

    plt.xlabel("Mean dH/dO displacement ratio")
    plt.ylabel("Count / number of events")
    pct = _percentile_from_bins(p_out, 97)
    plt.xlim(-4e-2 * pct, pct)
    plt.tick_params(which="both", direction="in", length=4, width=1.2)
    plt.legend(
        [h_np, h_p, h_base],
        ["outlier (non-productive)", "outlier (productive)", "baseline"],
        frameon=False, fontsize=14, loc="upper right",
    )
    plt.tight_layout()
    plt.savefig(out(f"{material}_{temp}K_dHdO_ratio.png"), dpi=300, transparent=True)
    plt.close()

    def peak(g):
        return float(g.loc[g["count_per_event"].idxmax(), "bin_center"])

    print(f"{material} {temp}K: baseline peak={peak(base):.2f}, "
          f"outlier P peak={peak(p_out):.2f}, NP peak={peak(np_out):.2f} "
          f"-> {material}_{temp}K_dHdO_ratio.png")


if __name__ == "__main__":
    each_material(MATERIALS, run)
