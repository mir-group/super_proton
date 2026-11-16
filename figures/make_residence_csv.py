"""Derive plot-ready residence-time CSVs from raw per-event residence times.

``scripts/run_analysis.py --analysis residence`` writes one row per event::

    {material}_{temp}K_residence_time.csv    pair_type, residence_time_ps

``plot_residence_time.py`` draws from the binned distribution and fitted curves::

    {material}_{temp}K_residence_density.csv  pair_type, bin_left_ps, bin_right_ps,
                                              bin_center_ps, probability_density
    {material}_{temp}K_fit_lines.csv          component, tau_ps, rho

This script performs that conversion. From a downloaded Zenodo trajectory::

    python scripts/run_analysis.py --analysis residence \\
        --dump data/zenodo/CDP_525K_4ns.trj \\
        --material CDP --temperature 525 --out results/raw
    python figures/make_residence_csv.py --raw results/raw \\
        --out results/reference/residence --material CDP
    SUPERPROTON_REFERENCE_DIR=results/reference python figures/plot_residence_time.py

``--out`` is required. Use a separate directory to preserve the bundled
reference data. To regenerate from the bundled per-event times instead::

    python figures/make_residence_csv.py --out results/reference/residence

The histogram and fits use ``superproton.residence``:

- O-pair, CsH2PO4: piecewise-linear in log-rho, giving a fast ``lambda1`` below
  the crossover ``t_c`` and a slow ``lambda2`` above it.
- O-pair, CsHSO4: single exponential, ``lambda1``.
- Polyanion pair (P-P for CDP, S-S for CHS): single exponential, ``lambda3``.

The ``lambda3`` and CsHSO4 ``lambda1`` curves are drawn as normalized
exponentials with amplitude ``lambda``. The piecewise branches use the fitted
intercept because a two-branch curve has no single normalization.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import each_material, ref  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from superproton.io import polyanion_element  # noqa: E402
from superproton.residence import (  # noqa: E402
    DEFAULT_RHO_FLOOR_OO,
    DEFAULT_RHO_FLOOR_PP,
    fit_piecewise,
    fit_single_exp,
)

MATERIALS = {"CDP": 525, "CHS": 450}


def _load_raw(path: Path, material: str) -> tuple[np.ndarray, np.ndarray]:
    poly = polyanion_element(material)
    d = pd.read_csv(path)
    tau_O = d.loc[d["pair_type"] == "O", "residence_time_ps"].to_numpy(float)
    tau_X = d.loc[d["pair_type"] == poly, "residence_time_ps"].to_numpy(float)
    if tau_O.size == 0 or tau_X.size == 0:
        raise ValueError(
            f"{path} has {tau_O.size} O-pair and {tau_X.size} {poly}-pair rows; "
            f"expected both. Is pair_type using '{poly}' for the polyanion?"
        )
    return tau_O, tau_X


def build(material: str, temp: int, raw_dir: Path, out_dir: Path) -> None:
    tau_O, tau_X = _load_raw(
        raw_dir / f"{material}_{temp}K_residence_time.csv", material
    )
    poly = polyanion_element(material)

    # Freedman-Diaconis density histograms, exactly as plotted.
    pdf_O, edge_O = np.histogram(tau_O[tau_O > 0], bins="fd", density=True)
    pdf_X, edge_X = np.histogram(tau_X[tau_X > 0], bins="fd", density=True)

    dens_rows = []
    for pair_type, pdf, edge in (("O", pdf_O, edge_O), (poly, pdf_X, edge_X)):
        for left, right, density in zip(edge[:-1], edge[1:], pdf):
            dens_rows.append({
                "pair_type": pair_type,
                "bin_left_ps": left,
                "bin_right_ps": right,
                "bin_center_ps": 0.5 * (left + right),
                "probability_density": density,
            })
    pd.DataFrame(dens_rows).to_csv(
        out_dir / f"{material}_{temp}K_residence_density.csv",
        index=False, float_format="%.6e",
    )

    fit_rows: list[dict] = []
    if material == "CDP":
        piecewise = fit_piecewise(tau_O, rho_floor=DEFAULT_RHO_FLOOR_OO)
        k1 = -piecewise.lambda_1_per_ps
        k2 = -piecewise.lambda_2_per_ps
        t_c = piecewise.tau_c_ps
        b1 = piecewise.intercept
        b2 = (k1 - k2) * t_c + b1  # continuity at the crossover
        below, above = edge_O[edge_O < t_c], edge_O[edge_O >= t_c]
        for tau, rho in zip(below, np.exp(b1) * np.exp(below * k1)):
            fit_rows.append({"component": "lambda1", "tau_ps": tau, "rho": rho})
        for tau, rho in zip(above, np.exp(b2) * np.exp(above * k2)):
            fit_rows.append({"component": "lambda2", "tau_ps": tau, "rho": rho})
        summary = (f"1/l1={1 / piecewise.lambda_1_per_ps:.0f} ps, "
                   f"1/l2={1 / piecewise.lambda_2_per_ps:.0f} ps, t_c={t_c:.0f} ps")
    else:
        o_fit = fit_single_exp(
            tau_O, rho_floor=DEFAULT_RHO_FLOOR_OO, trim_to_peak=True
        )
        slope = -o_fit.lambda_per_ps
        for tau, rho in zip(edge_O, -slope * np.exp(edge_O * slope)):
            fit_rows.append({"component": "lambda1", "tau_ps": tau, "rho": rho})
        summary = f"1/l1={1 / o_fit.lambda_per_ps:.0f} ps"

    x_fit = fit_single_exp(tau_X, rho_floor=DEFAULT_RHO_FLOOR_PP)
    slope_X = -x_fit.lambda_per_ps
    for tau, rho in zip(edge_X, -slope_X * np.exp(edge_X * slope_X)):
        fit_rows.append({"component": "lambda3", "tau_ps": tau, "rho": rho})
    pd.DataFrame(fit_rows).to_csv(
        out_dir / f"{material}_{temp}K_fit_lines.csv",
        index=False, float_format="%.6e",
    )

    print(f"{material} {temp}K: {summary}, 1/l3={1 / x_fit.lambda_per_ps:.0f} ps "
          f"(n_O={tau_O.size}, n_{poly}={tau_X.size}) -> {out_dir}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw", type=Path, default=None,
                   help="Directory holding the raw *_residence_time.csv "
                        "(default: data/reference/residence).")
    p.add_argument("--out", type=Path, required=True,
                   help="Directory for the derived CSVs. Required: without it an "
                        "earlier version defaulted to data/reference/ and silently "
                        "overwrote the published files it was meant to verify.")
    p.add_argument("--material", choices=["CDP", "CHS"], action="append",
                   help="Restrict to one material; repeatable. Default: both.")
    p.add_argument("--temperature", type=int, default=None,
                   help="Temperature label of the raw file. Applies to a single "
                        "--material only. Default: 525 (CDP) / 450 (CHS).")
    args = p.parse_args(argv)

    # One --temperature cannot label two materials: CDP and CHS were run at
    # different temperatures, so applying a single value to both would mislabel
    # the output and silently look for files that do not exist.
    if args.temperature is not None and len(args.material or []) != 1:
        p.error(
            "--temperature applies to one material; pass exactly one --material "
            "with it (e.g. --material CDP --temperature 540), or omit it to use "
            "the representative temperature of each."
        )

    raw_dir = args.raw if args.raw is not None else ref("residence")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = {
        material: (args.temperature if args.temperature is not None
                   else MATERIALS[material])
        for material in (args.material or list(MATERIALS))
    }
    each_material(requested,
                  lambda material, temp: build(material, temp, raw_dir, out_dir),
                  what="CSVs", where=raw_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
