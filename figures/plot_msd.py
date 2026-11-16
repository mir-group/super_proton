"""
MSD vs delta-time at a representative temperature,
for both CDP (CsH2PO4, 525 K) and CHS (CsHSO4, 450 K).

Data: MLMD (machine-learned MD), 4 ns trajectories.
Each CSV holds the mean-squared displacement of every atomic species vs lag time
Delta t. Column order differs between the two materials:
    CDP_525K_msd.csv : time_ps, MSD_Cs, MSD_P, MSD_H, MSD_O   [A^2]
    CHS_450K_msd.csv : time_ps, MSD_Cs, MSD_H, MSD_S, MSD_O   [A^2]

Run:
    python plot_msd.py
Produces: msd_CDP_525K.png, msd_CHS_450K.png (and a side-by-side msd_CDP_CHS.png)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_csv_columns as load, ref, out  # noqa: E402

DATA = ref("msd")


import matplotlib.pyplot as plt

# ---- plotting window (matches the manuscript figure) ---------------------
SIM_LEN = 4.0            # ns, trajectory length
T_START = 2.1            # ps, drop the ballistic/short-time noise
T_END = SIM_LEN * 0.7e3  # ps, cut the poorly averaged long-time tail (2800 ps)
X_MAX = SIM_LEN * 0.8e3  # ps, x-axis upper limit (3200 ps)

# species -> (csv column name, color); order sets the legend/draw order
CDP = {
    "csv": DATA / "CDP_525K_msd.csv",
    "title": "CDP (CsH$_2$PO$_4$), 525 K",
    "series": [
        ("H",  "MSD_H_A2",  "dodgerblue"),
        ("O",  "MSD_O_A2",  "red"),
        ("P",  "MSD_P_A2",  "orange"),
        ("Cs", "MSD_Cs_A2", "purple"),
    ],
}
CHS = {
    "csv": DATA / "CHS_450K_msd.csv",
    "title": "CHS (CsHSO$_4$), 450 K",
    "series": [
        ("H",  "MSD_H_A2",  "dodgerblue"),
        ("O",  "MSD_O_A2",  "red"),
        ("S",  "MSD_S_A2",  "goldenrod"),
        ("Cs", "MSD_Cs_A2", "purple"),
    ],
}


def draw(ax, spec):
    d = load(spec["csv"])
    t = d["time_ps"]
    mask = (t > T_START) & (t < T_END)
    for label, col, color in spec["series"]:
        ax.plot(t[mask], d[col][mask], label=label, color=color, lw=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(T_START, X_MAX)
    ax.set_xlabel(r"$\Delta t$ [ps]")
    ax.set_ylabel(r"MSD [$\AA^2$]")
    ax.set_title(spec["title"], fontsize=14)
    ax.legend(frameon=False, ncol=2)
    ax.tick_params(which="both", direction="in", length=4, width=1.2)
    ax.grid(True, which="both", alpha=0.3)


if __name__ == "__main__":
    plt.rcParams["font.size"] = 17

    # individual figures (one per material)
    for spec, png in [(CDP, "msd_CDP_525K.png"), (CHS, "msd_CHS_450K.png")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        draw(ax, spec)
        fig.tight_layout()
        fig.savefig(out(png), dpi=300)
        plt.close(fig)
        print(f"wrote {png}")

    # side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    draw(axes[0], CDP)
    draw(axes[1], CHS)
    fig.tight_layout()
    fig.savefig(out("msd_CDP_CHS.png"), dpi=300)
    plt.close(fig)
    print("wrote msd_CDP_CHS.png")
