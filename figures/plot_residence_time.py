"""
Residence-time distributions with exponential-rate fits, on a log-y axis.
Reproduces Fig. 4a (CDP, 525 K) and Fig. 7a (CHS, 450 K).

Two overlaid histograms per material:
  - O-pair residence time tau_{O-O}   (blue)
  - polyanion-pair residence time     (orange): P-P for CDP, S-S for CHS
and the exponential fits of each:
  - CDP: O-pair is bi-exponential -> 1/lambda1 (blue solid, fast) + 1/lambda2
    (cyan dash-dot, slow); polyanion pair -> 1/lambda3 (orange dashed)
  - CHS: O-pair single exponential -> 1/lambda1 (blue solid); polyanion -> 1/lambda3

Reads (produced by make_residence_csv.py):
  {mat}_residence_density.csv   pair_type, bin_left_ps, bin_right_ps, bin_center_ps, probability_density
  {mat}_fit_lines.csv           component (lambda1|lambda2|lambda3), tau_ps, rho

Run:
    python plot_residence_time.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import each_material, out, ref  # noqa: E402

DATA = ref("residence")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# per material: (temperature, polyanion pair code, x-limits, x-ticks)
CONFIG = {
    "CDP": dict(temp=525, poly="P", xlim=(-50, 1800), xticks=np.arange(0, 1800, 500),
                mat_name=r"CsH$_2$PO$_4$"),
    "CHS": dict(temp=450, poly="S", xlim=(-20, 800), xticks=np.arange(0, 800, 200),
                mat_name=r"CsHSO$_4$"),
}
HIST_COLOR = {"O": "#1f77b4", "poly": "#ff7f0e"}
FIT_STYLE = {  # component -> (color, linestyle, lambda-subscript)
    "lambda1": ("blue", "-", "1"),
    "lambda2": ("cyan", "-.", "2"),
    "lambda3": ("orange", "--", "3"),
}


def inv_rate_ps(g):
    """1/lambda in ps from an analytic exponential segment rho = A*exp(-lambda*tau)."""
    slope = stats.linregress(g["tau_ps"], np.log(g["rho"])).slope
    return -1.0 / slope


def plot(material):
    c = CONFIG[material]
    temp, poly = c["temp"], c["poly"]
    dens = pd.read_csv(DATA / f"{material}_{temp}K_residence_density.csv")
    fits = pd.read_csv(DATA / f"{material}_{temp}K_fit_lines.csv")

    plt.figure()
    plt.rcParams.update({"font.size": 15})

    # histograms: polyanion pair first (drawn under), then O-pair on top
    poly_label = rf"{poly}-pair: $\tau_{{\mathrm{{{poly}-{poly}}}}}$"
    h_poly = _bar(dens, poly, HIST_COLOR["poly"], 0.5, poly_label)
    h_O = _bar(dens, "O", HIST_COLOR["O"], 0.6, r"O-pair: $\tau_{\mathrm{O-O}}$")

    # fit lines, in component order lambda1, (lambda2), lambda3
    fit_handles = []
    for comp in ("lambda1", "lambda2", "lambda3"):
        g = fits[fits["component"] == comp].sort_values("tau_ps")
        if g.empty:
            continue
        color, ls, sub = FIT_STYLE[comp]
        inv = inv_rate_ps(g)
        (line,) = plt.plot(g["tau_ps"], g["rho"], color=color, linestyle=ls,
                           linewidth=2, label=rf"1/$\lambda_{sub}$={inv:.0f} ps")
        fit_handles.append(line)

    # legend order matches the paper figure: O-pair, polyanion pair, then fits
    handles = [h_O, h_poly] + fit_handles
    labels = [h.get_label() for h in handles]

    plt.xlabel(r"Residence times $\tau$ [ps]")
    plt.ylabel(r"$\rho(\tau)$")
    plt.xlim(c["xlim"])
    plt.xticks(c["xticks"])
    plt.yscale("log")
    plt.ylim(6e-5, 1e-1)
    plt.minorticks_on()
    plt.tick_params(which="both", direction="in", length=4, width=1.2)
    plt.legend(frameon=False, fontsize=14, handles=handles, labels=labels)
    plt.tight_layout()
    plt.savefig(out(f"{material}_{temp}K_log.png"), dpi=300, transparent=True)
    plt.close()
    print(f"wrote {material}_{temp}K_log.png")


def _bar(dens, pair_type, color, alpha, label):
    g = dens[dens["pair_type"] == pair_type].sort_values("bin_left_ps")
    widths = g["bin_right_ps"].to_numpy() - g["bin_left_ps"].to_numpy()
    return plt.bar(g["bin_left_ps"], g["probability_density"], width=widths,
                   align="edge", color=color, alpha=alpha, linewidth=0, label=label)


if __name__ == "__main__":
    each_material(("CDP", "CHS"), plot)
