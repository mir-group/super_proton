"""
Polyanion reorientation rates vs. temperature (Arrhenius), for CDP and CHS.

Reproduces the four figures:
  (a) CDP_CHS_arrhenius_2e-4_5e-5_count_10.png   both materials (CDP partial)
  (b) CDP_arrhenius_2e-4_5e-5_count_10.png       CDP, all rate terms
  (c) CDP_arrhenius_2e-4_5e-5_count_10_avg.png   CDP + average-method cross-check
  (d) CHS_arrhenius_2e-4_5e-5_count_10_avg.png   CHS + average-method cross-check

Rate terms (units ps^-1):
  lambda_1 (O-O)   O-pair (polyanion oxygen) reorientation, fast component
  lambda_2 (O-O)   slow component (CDP O-pair is bi-exponential)
  lambda_P (P-P / S-S)   *productive* polyanion (PO4 / SO4) rotation
  lambda_NP        *nonproductive* rotation (does not advance a proton)
  bar-lambda_*     same rate from a simple global-average method, drawn with
                   open markers + solid line as a cross-check of the fitted rates

Each E_a is an Arrhenius fit ln(rate) = -E_a/(kT) + const over all temperatures.

Run:
    python plot_rotation_rate.py
Reads CDP_rates_2e-4_5e-5_count_10.csv and CHS_rates_2e-4_5e-5_count_10.csv.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import ref, out  # noqa: E402
# The fit object is needed for the drawn line, so the regression stays local;
# only the constant is shared, so it cannot drift from the library's value.
from superproton.residence import K_B_EV_PER_K as K_B  # noqa: E402

DATA = ref("rotation_rates")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def inv1000(x):
    """1000/T <-> T map for the secondary axis (safe at 0)."""
    x = np.asarray(x, dtype=float)
    nz = np.isclose(x, 0)
    out = np.full_like(x, np.inf)
    out[~nz] = 1e3 / x[~nz]
    return out


def arrhenius(T, rate):
    """Fit ln(rate) vs 1/T; return activation energy, its error, and the fit."""
    fit = stats.linregress(1.0 / T, np.log(rate))
    return -fit.slope * K_B, abs(K_B * fit.stderr), fit


def ea_str(Ea, err):
    return f"{Ea:.2f}({err * 1e2:.0f})"


def fit_line(ax, T, fit, color, ls="--"):
    x = 1e3 / T
    xf = np.linspace(x.min(), x.max(), 100)
    yf = np.exp(fit.slope * (xf / 1e3) + fit.intercept)
    ax.plot(xf, yf, linestyle=ls, color=color, alpha=0.7, zorder=1, linewidth=1)


def series(ax, T, rate, err, fmt, color, base):
    """Marker series (no connecting line) + dashed Arrhenius fit."""
    rate = np.asarray(rate, float)
    Ea, e, fit = arrhenius(T, rate)
    ax.errorbar(1e3 / T, rate, yerr=(None if err is None else np.asarray(err, float)),
                fmt=fmt, capsize=0, color=color,
                label=base + rf", $E_a$={ea_str(Ea, e)}eV")
    fit_line(ax, T, fit, color, "--")


def avg_series(ax, T, rate, fmt, color, base):
    """Open-marker average-method series + solid Arrhenius fit (cross-check)."""
    rate = np.asarray(rate, float)
    Ea, e, fit = arrhenius(T, rate)
    ax.plot(1e3 / T, rate, marker=fmt, linestyle="none", color=color,
            fillstyle="none", zorder=10,
            label=base + rf", $E_a$={ea_str(Ea, e)}eV")
    fit_line(ax, T, fit, color, "-")


def plot_cdp(ax, df, partial=False, plot_np=True, plot_average=False):
    T = df["T"].values
    series(ax, T, df["O_rate1"], df["O_rate1_err"], "o", "dodgerblue",
           r"$\lambda_1^\mathrm{CDP}$(O-O)")
    if not partial:
        series(ax, T, df["O_rate2"], df["O_rate2_err"], "D", "dodgerblue",
               r"$\lambda_2$(O-O)")
        if plot_np:
            series(ax, T, df["lambda_NP"], None, "x", "orange", r"$\lambda_\mathrm{NP}$")
    series(ax, T, df["P_rate"], df["P_rate_err"], ">", "dodgerblue",
           r"$\lambda_\mathrm{P}^\mathrm{CDP}$(P-P)")
    if plot_average and not partial:
        if plot_np:
            avg_series(ax, T, df["avg_NP_rate"], "X", "orange", r"$\bar{\lambda}_\mathrm{NP}$")
        avg_series(ax, T, df["avg_P_rate"], ">", "dodgerblue", r"$\bar{\lambda}_\mathrm{P}$")


def plot_chs(ax, df, plot_np=True, plot_average=False):
    T = df["T"].values
    series(ax, T, df["O_rate"], df["O_rate_err"], "o", "darkviolet", r"$\lambda_1$(O-O)")
    if plot_np:
        series(ax, T, df["lambda_NP"], None, "x", "red", r"$\lambda_\mathrm{NP}$")
    series(ax, T, df["P_rate"], df["P_rate_err"], ">", "darkviolet", r"$\lambda_\mathrm{P}$(S-S)")
    if plot_average:
        if plot_np:
            avg_series(ax, T, df["avg_NP_rate"], "X", "red", r"$\bar{\lambda}_\mathrm{NP}$")
        avg_series(ax, T, df["avg_P_rate"], ">", "darkviolet", r"$\bar{\lambda}_\mathrm{P}$")


def format_plot(ax, material):
    ax.set_ylim(*{"cdp": (0.8e-3, 1), "chs": (2e-3, 2.5e-1), "both": (1e-3, 6e-1)}[material])
    ax.set_yscale("log")
    ax.set_xlabel(r"1000/T [K$^{-1}$]")
    ax.set_ylabel(r"Rotation rate [ps$^{-1}$]")
    ax2 = ax.secondary_xaxis("top", functions=(inv1000, inv1000))
    ax2.set_xlabel("T [K]")
    ax.tick_params(which="both", direction="in", length=7, width=1.5)
    ax2.tick_params(which="both", direction="in", length=7, width=1.5)

    # fitted (lambda) entries first, average (bar-lambda) entries last
    handles, labels = ax.get_legend_handles_labels()
    err = [(h, l) for h, l in zip(handles, labels) if r"\bar" not in l]
    bar = [(h, l) for h, l in zip(handles, labels) if r"\bar" in l]
    items = err + bar
    hs, ls = [i[0] for i in items], [i[1] for i in items]
    n = len(hs)
    if n <= 4:
        ax.legend(hs, ls, frameon=False, loc="best", fontsize=11)
    else:
        mid = n // 2 if material == "both" else n // 2 + 1
        leg1 = ax.legend(hs[:mid], ls[:mid], frameon=False, loc="upper right", fontsize=11)
        ax.add_artist(leg1)
        ax.legend(hs[mid:], ls[mid:], frameon=False, loc="upper left", fontsize=11)


def make(figname, material, plotter):
    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots()
    plotter(ax)
    format_plot(ax, material)
    fig.tight_layout()
    fig.savefig(out(figname), dpi=300)
    plt.close(fig)
    print(f"wrote {figname}")


if __name__ == "__main__":
    cdp = pd.read_csv(DATA / "CDP_rates_2e-4_5e-5_count_10.csv")
    chs = pd.read_csv(DATA / "CHS_rates_2e-4_5e-5_count_10.csv")

    # (a) both materials (CDP shown partial: only lambda_1 and lambda_P)
    make("a_CDP_CHS_arrhenius_2e-4_5e-5_count_10.png", "both",
         lambda ax: (plot_cdp(ax, cdp, partial=True, plot_np=True),
                     plot_chs(ax, chs, plot_np=True)))
    # (b) CDP, all rate terms
    make("b_CDP_arrhenius_2e-4_5e-5_count_10.png", "cdp",
         lambda ax: plot_cdp(ax, cdp, partial=False, plot_np=True, plot_average=False))
    # (c) CDP + average-method cross-check
    make("c_CDP_arrhenius_2e-4_5e-5_count_10_avg.png", "cdp",
         lambda ax: plot_cdp(ax, cdp, partial=False, plot_np=True, plot_average=True))
    # (d) CHS + average-method cross-check
    make("d_CHS_arrhenius_2e-4_5e-5_count_10_avg.png", "chs",
         lambda ax: plot_chs(ax, chs, plot_np=True, plot_average=True))
