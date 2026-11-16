"""The manuscript's reported values, in one place.

Both ``scripts/run_analysis.py --compare`` and ``tests/test_reference.py`` check
computed results against the paper. Keeping the numbers here means a correction
during review is made once, rather than in two files that can silently diverge.

These are the *published* values. Regression checks and their tolerances are
in ``tests/test_reference.py``.

Keys are ``(material, temperature_K)``. Temperatures are the representative ones
the paper reports in full: 525 K for CsH2PO4, 450 K for CsHSO4.
"""

from __future__ import annotations

#: Inverse rates in ps, from the residence-time fits of Figure 4a (CsH2PO4) and
#: Figure 7a (CsHSO4). CsH2PO4's O-pair distribution is bi-exponential, so it
#: has a fast branch, a slow branch and a crossover; CsHSO4's is single.
#: Quoted to the nearest ps, so a computed value has to round to these.
RATES_PS: dict[tuple[str, int], dict[str, int]] = {
    ("CDP", 525): {"inv_lambda1": 21, "inv_lambda2": 156, "t_c": 72, "inv_lambda3": 494},
    ("CHS", 450): {"inv_lambda1": 37, "inv_lambda3": 126},
}

#: Covalent X-O-H geometry from SI Table S6, as ``(mean, standard deviation)``.
#: Lengths in angstrom, angle in degrees. The standard deviation is the width of
#: the distribution, not the error on the mean, which is negligible at >1e6
#: samples per entry.
XOH_GEOMETRY: dict[tuple[str, int], dict[str, tuple[float, float]]] = {
    ("CDP", 525): {"d_OH": (1.032, 0.054), "d_XO": (1.609, 0.046), "angle_deg": (115.4, 8.1)},
    ("CHS", 450): {"d_OH": (1.018, 0.042), "d_XO": (1.623, 0.048), "angle_deg": (109.1, 6.1)},
}

#: Proton tracer diffusivity in cm^2/s, from the ML MD column of Figure 2.
#: See :func:`superproton.msd.diffusivity`: D shifts by a few percent with the
#: MSD fit window, which is larger than the error on any single fit.
DIFFUSIVITY_CM2_S: dict[tuple[str, int], float] = {
    ("CDP", 525): 1.848e-07,
    ("CHS", 450): 3.749e-07,
}

#: Arrhenius activation energies in eV, each fitted over every temperature that
#: ``data/reference/`` carries, so these are keyed by material rather than by a
#: single temperature. The rotation rates are the four of Figure 4b (CsH2PO4)
#: and the overall rate of Figure 7c (CsHSO4); ``diffusion`` is the proton
#: tracer diffusivity of Figure 2c, quoted in the text as 0.43(2) eV and
#: 0.29(1) eV. Quoted to two decimals, so a computed value has to round to
#: these. Rates the paper does not state an E_a for are omitted.
ACTIVATION_ENERGIES_EV: dict[str, dict[str, float]] = {
    "CDP": {"lambda_1": 0.16, "lambda_2": 0.35, "lambda_P": 0.36,
            "lambda_NP": 0.23, "diffusion": 0.43},
    "CHS": {"lambda_1": 0.16, "diffusion": 0.29},
}

#: Pointing-attempt outcomes for CsH2PO4 525 K over the full 4 ns trajectory,
#: from ``data/reference/pointing/attempt_outcome.csv``. Note that a shorter
#: window is not directly comparable: the first 500 ps runs about 1.6x above the
#: 4 ns mean attempt rate, because only 10 ps of equilibration is dropped.
POINTING_OUTCOMES: dict[tuple[str, int], dict[str, float]] = {
    ("CDP", 525): {"n_decided": 3030, "committed": 0.8436,
                   "reverted": 0.1142, "superseded": 0.0422},
}


#: Corrected SI Tables S2-S3: NP-centred overlap percentages, keyed by
#: (NP role, P role). Bin order is 0, 1, 2, >=3 overlapping productive events.
#: Rounded percentages can sum to 100.01 rather than exactly 100.
OVERLAP_PERCENT: dict[tuple[str, int], dict[tuple[str, str], tuple[float, ...]]] = {
    ("CDP", 525): {
        ("exchanger", "carrier"): (99.13, 0.84, 0.03, 0.01),
        ("exchanger", "recipient"): (93.66, 6.22, 0.11, 0.00),
        ("exchanger", "provider"): (93.72, 6.19, 0.09, 0.00),
        ("host", "carrier"): (98.48, 1.45, 0.07, 0.00),
        ("host", "recipient"): (97.53, 2.44, 0.03, 0.00),
        ("host", "provider"): (97.46, 2.51, 0.03, 0.00),
    },
    ("CHS", 450): {
        ("exchanger", "carrier"): (80.14, 17.81, 1.60, 0.45),
        ("exchanger", "recipient"): (79.72, 17.88, 2.12, 0.29),
        ("exchanger", "provider"): (78.51, 19.07, 2.12, 0.30),
        ("host", "carrier"): (97.98, 1.91, 0.11, 0.00),
        ("host", "recipient"): (70.42, 25.44, 3.83, 0.31),
        ("host", "provider"): (69.51, 26.11, 4.03, 0.34),
    },
}
