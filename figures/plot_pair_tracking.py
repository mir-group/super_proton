"""
#14 - single-proton O-O and X-X pair tracking vs. simulation time.
  (a) CDP: H id 268 at 525 K   (b) CHS: H id 342 at 500 K   (1.5 ns MLMD)

Four stacked panels sharing a time axis:
  1. H+ displacement [A]
  2. O nearest neighbor  (which oxygen label the proton sits on)
  3. thin band: tau_{O-O} intervals (blue) and tau_{P-P}/tau_{S-S} intervals (orange)
  4. P/S nearest neighbor (which polyanion label the proton sits on)
Red dashed lines mark polyanion (X-X) pair switches; the red shaded strips are the
O-H reorientation windows leading into each switch. The oxygen label changes many
times (fast O-O hopping) while the polyanion label changes only at the rarer,
productive X-X switches.

Reads {mat}_H{cand}_track.csv (t_ps, H_displacement_A, O_nearest_neighbor,
X_nearest_neighbor) and {mat}_H{cand}_events.csv (kind, time_ps), produced by
track_candidate.py.

Run:
    python plot_pair_tracking.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import each_material, out, ref  # noqa: E402

DATA = ref("pair_tracking")


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

CFG = {
    "CDP": dict(cand=268, poly="P", lw=0.5, xlim=(450, 820), o_top=None, x_top=None, prefix="a"),
    "CHS": dict(cand=342, poly="S", lw=1.5, xlim=(280, 500), o_top=13, x_top=5, prefix="b"),
}


def plot(material):
    c = CFG[material]
    tag = f"{material}_H{c['cand']}"
    d = pd.read_csv(DATA / f"{tag}_track.csv")
    ev = pd.read_csv(DATA / f"{tag}_events.csv")
    t = d["t_ps"].to_numpy()
    H_dis, O_bond, X_host = (d["H_displacement_A"].to_numpy(),
                             d["O_nearest_neighbor"].to_numpy(),
                             d["X_nearest_neighbor"].to_numpy())
    O_sw = ev.loc[ev["kind"] == "O_pair_switch", "time_ps"].to_numpy()
    X_sw = ev.loc[ev["kind"] == "X_pair_switch", "time_ps"].to_numpy()
    reorient = ev.loc[ev["kind"] == "reorient_start", "time_ps"].to_numpy()

    plt.rcParams["font.size"] = 15
    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 0.1, 1], hspace=0.08)
    ax0, ax1, ax2, ax3 = (fig.add_subplot(gs[i], sharex=(None if i == 0 else fig.axes[0]))
                          for i in range(4))
    for ax in (ax0, ax1, ax2):
        plt.setp(ax.get_xticklabels(), visible=False)

    # panel 1: H+ displacement
    ax0.plot(t, H_dis)
    ax0.set_ylabel("H$^+$\ndisplacement [Å]")
    ax0.vlines(X_sw, 0, H_dis.max(), color="r", linestyle="--", linewidth=2)

    # panel 2: O nearest neighbor
    ax1.plot(t, O_bond, linewidth=c["lw"])
    ax1.vlines(X_sw, 1, O_bond.max(), color="r", linestyle="--", linewidth=2)
    ax1.set_ylabel("O nearest\n neighbor")
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
    if c["o_top"]:
        ax1.set_ylim(top=c["o_top"])

    # panel 3: timeline band of tau intervals
    o_times = [t[0]] + list(O_sw) + [t[-1]]
    x_times = [t[0]] + list(X_sw) + [t[-1]]
    for i in range(len(o_times) - 1):
        ax2.plot([o_times[i], o_times[i + 1]], [4.0, 4.0], color="#1f77b4",
                 marker="|", markersize=10, markeredgewidth=2,
                 label=r"$\tau_{O-O}$ intervals" if i == 0 else None)
    for i in range(len(x_times) - 1):
        ax2.plot([x_times[i], x_times[i + 1]], [0, 0], color="#ff7f0e",
                 marker="|", markersize=10, markeredgewidth=2,
                 label=(rf"$\tau_{{{c['poly']}-{c['poly']}}}$ intervals") if i == 0 else None)
    ax2.set_ylim(-0.5, 4.5)
    ax2.yaxis.set_visible(False)
    ax2.xaxis.set_visible(False)
    for s in ("top", "right", "bottom", "left"):
        ax2.spines[s].set_visible(False)

    # panel 4: polyanion nearest neighbor
    ax3.plot(t, X_host, linewidth=c["lw"])
    ax3.vlines(X_sw, 1, X_host.max(), color="r", linestyle="--", linewidth=2)
    for t_start, t_end in zip(reorient, X_sw):
        for ax in (ax0, ax1, ax3):
            ax.axvspan(t_start, t_end, color="r", alpha=0.1)
    ax3.set_ylabel(f"{c['poly']} nearest\n neighbor")
    ax3.set_xlim(*c["xlim"])
    if c["x_top"]:
        ax3.set_ylim(top=c["x_top"])
    ax3.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
    ax3.set_xlabel("time [ps]")

    handles, labels = ax2.get_legend_handles_labels()
    ax3.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.67, 1.0),
               frameon=False, fontsize=12, ncol=2)

    png = f"{c['prefix']}_{material}_H{c['cand']}_pair_tracking.png"
    fig.savefig(out(png), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    each_material(("CDP", "CHS"), plot)
