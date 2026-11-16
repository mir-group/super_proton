"""
Probability density vs. rotation period, for CDP (525 K) and CHS (450 K), 4 ns MLMD.

Two distributions per material:
  O-pair switch   - time between the start of a reorientation and the next O-pair
                    (hydrogen-bond partner) switch
  P/S-pair switch - same, but for the polyanion pair switch (PO4 for CDP, SO4 for CHS)

NOTE: the polyanion is PO4 in CDP and SO4 in CHS, so the polyanion curve is
labelled "P-pair switch" for CDP and "S-pair switch" for CHS.

Reads the binned distributions shipped in ``data/reference/rotation_period/``.
The distributions histogram ``switch_frame - reorient_start_frame`` with
``bins="auto"`` and ``density=True``. To regenerate them from a trajectory, see
``scripts/run_analysis.py --analysis rotation_period``.

Run:
    python figures/plot_rotation_period.py
Produces: CDP_525_rot_period_prob.png, CHS_450_rot_period_prob.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import each_material, out, ref  # noqa: E402

DATA = ref("rotation_period")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MATERIALS = {"CDP": 525, "CHS": 450}
POLY = {"CDP": "P-pair", "CHS": "S-pair"}  # polyanion pair label per material


def _percentile_from_bins(left, right, density, q):
    """Interpolate the ``q``-th percentile out of a binned density."""
    widths = right - left
    cdf = np.cumsum(density * widths)
    cdf /= cdf[-1]
    return float(np.interp(q / 100.0, cdf, right))


def plot(material, temp):
    d = pd.read_csv(DATA / f"{material}_{temp}_rot_period.csv")
    poly_label = f"{POLY[material]} switch"

    plt.figure(figsize=(7, 5))
    plt.rcParams.update({"font.size": 15})

    # O-pair: outline only; polyanion pair: filled.
    o = d[d["switch_type"] == "O-pair"]
    x = d[d["switch_type"] == POLY[material]]
    # Draw the filled polyanion histogram first, then the O-pair outline on top
    # with a raised zorder: in the target figure the outline stays crisp where
    # the two overlap rather than being dimmed by the fill's alpha.
    plt.stairs(
        x["prob_density"],
        np.append(x["bin_left_ps"].to_numpy(), x["bin_right_ps"].iloc[-1]),
        label=poly_label,
        fill=True,
        alpha=0.8,
        color="tab:orange",
    )
    plt.stairs(
        o["prob_density"],
        np.append(o["bin_left_ps"].to_numpy(), o["bin_right_ps"].iloc[-1]),
        label="O-pair switch",
        linewidth=1.5,
        color="tab:blue",
        zorder=3,
    )

    # truncate x at the 99th percentile of the polyanion-pair distribution
    p99 = _percentile_from_bins(
        x["bin_left_ps"].to_numpy(), x["bin_right_ps"].to_numpy(),
        x["prob_density"].to_numpy(), 99,
    )
    plt.xlim(-4e-2 * p99, p99)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [labels.index("O-pair switch"), labels.index(poly_label)]
    plt.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False)
    plt.ylabel("Probability density")
    plt.xlabel("Rotation period [ps]")
    plt.tick_params(which="both", direction="in", length=4, width=1.2)
    plt.tight_layout()
    plt.savefig(out(f"{material}_{temp}_rot_period_prob.png"), dpi=300)
    plt.close()

    def mean_from_bins(g):
        w = g["bin_right_ps"] - g["bin_left_ps"]
        return float((g["bin_center_ps"] * g["prob_density"] * w).sum()
                     / (g["prob_density"] * w).sum())

    print(f"{material} {temp}K: O-pair mean={mean_from_bins(o):.2f} ps, "
          f"{POLY[material]} mean={mean_from_bins(x):.2f} ps "
          f"-> {material}_{temp}_rot_period_prob.png")


if __name__ == "__main__":
    each_material(MATERIALS, plot)
