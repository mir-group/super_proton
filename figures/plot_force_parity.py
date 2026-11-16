"""
Force-parity plots for the Allegro MLFF on the DFT test sets: CDP and CHS.

Each point is one Cartesian force component (eV/A) of one atom in one test-set
frame, DFT (x) vs. Allegro prediction (y); the red line is y = x. Test-set force
errors:
    CDP: RMSE 44.6 meV/A, MAE 31.2 meV/A  (34896 components)
    CHS: RMSE 114.4 meV/A, MAE 56.2 meV/A (66864 components)

Reads {CDP,CHS}_force_parity.csv (columns: F_dft_eV_A, F_allegro_eV_A).

Run:
    python plot_force_parity.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ref, out  # noqa: E402

DATA = ref("model_validation")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LIM = 20
MAT_NAME = {"CDP": r"CsH$_2$PO$_4$", "CHS": r"CsHSO$_4$"}


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


if __name__ == "__main__":
    for material in ("CDP", "CHS"):
        d = pd.read_csv(DATA / f"{material}_force_parity.csv")
        fd, fn = d["F_dft_eV_A"].to_numpy(), d["F_allegro_eV_A"].to_numpy()
        print(f"{material}: RMSE {rmse(fd, fn)*1e3:.1f} meV/Å, "
              f"MAE {np.mean(np.abs(fd-fn))*1e3:.1f} meV/Å, n={fd.size}")

        plt.figure()
        plt.rcParams["font.size"] = 14
        plt.plot(np.linspace(-LIM, LIM), np.linspace(-LIM, LIM), "r")
        plt.scatter(fd, fn, color="C0", s=8)
        plt.xlabel("DFT forces per component [eV/Å]")
        plt.ylabel("Allegro forces per component [eV/Å]")
        plt.title(f"{MAT_NAME[material]} Forces Parity on Test Set")
        plt.axis([-LIM, LIM, -LIM, LIM])
        plt.tight_layout()
        plt.savefig(out(f"parity_{material}_test.png"), dpi=300)
        plt.close()
        print(f"  -> parity_{material}_test.png")
