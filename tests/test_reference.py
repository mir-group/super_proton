"""Regression tests against the manuscript's published values.

The other test modules validate the analysis code against synthetic inputs with
known ground truth. This one pins it to the numbers actually reported in the
paper, so a refactor that silently changes a published result fails here.

Two tiers:

**Tier A** runs on the reference data shipped in ``data/reference/`` and needs
no trajectory, so it runs everywhere. It covers the analysis stages that turn
intermediate arrays into the quoted numbers.

**Tier B** runs the whole pipeline from a production trajectory and is skipped
unless you point it at one. The trajectories are 1-3 GB and live on Zenodo
(record 14649023); see ``data/README.md``. Enable with::

    export SUPERPROTON_CDP_525K_TRJ=/path/to/CDP_525K_4ns.trj
    export SUPERPROTON_CHS_450K_TRJ=/path/to/CHS_450K_4ns.trj
    pytest tests/test_reference.py

Tolerances are stated per assertion and reflect why the number can move, not a
tuning knob. Where the pipeline is exact -- event counts, residence times -- the
tolerance is exact.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest

REFERENCE = Path(__file__).resolve().parent.parent / "data" / "reference"

from superproton.published import ACTIVATION_ENERGIES_EV, RATES_PS, XOH_GEOMETRY

# Keyed by material at its representative temperature, which is what every
# assertion below uses.
PUBLISHED_RATES = {"CDP": RATES_PS[("CDP", 525)], "CHS": RATES_PS[("CHS", 450)]}
PUBLISHED_XOH = {
    "CDP": {k.replace("angle_deg", "angle"): v for k, v in XOH_GEOMETRY[("CDP", 525)].items()},
    "CHS": {k.replace("angle_deg", "angle"): v for k, v in XOH_GEOMETRY[("CHS", 450)].items()},
}


def _load_residence_taus(material: str, temperature: int):
    """O-pair and polyanion-pair residence times from the shipped reference CSV."""
    path = REFERENCE / "residence" / f"{material}_{temperature}K_residence_time.csv"
    poly = "P" if material == "CDP" else "S"
    tau_O, tau_X = [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            value = float(row["residence_time_ps"])
            if row["pair_type"] == "O":
                tau_O.append(value)
            elif row["pair_type"] == poly:
                tau_X.append(value)
    return np.array(tau_O), np.array(tau_X)


# ---------------------------------------------------------------------------
# Tier A -- shipped reference data only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("material", "quantity", "source", "column"),
    [
        # Rotation rates, Figure 4b (CDP) and Figure 7c (CHS). CDP resolves a
        # fast and a slow overall rate; CHS has one, so its column is unnumbered.
        ("CDP", "lambda_1", "rotation_rates", "O_rate1"),
        ("CDP", "lambda_2", "rotation_rates", "O_rate2"),
        ("CDP", "lambda_P", "rotation_rates", "P_rate"),
        ("CDP", "lambda_NP", "rotation_rates", "lambda_NP"),
        ("CHS", "lambda_1", "rotation_rates", "O_rate"),
        # Proton tracer diffusivity, Figure 2c.
        ("CDP", "diffusion", "diffusion", "D_cm2_s"),
        ("CHS", "diffusion", "diffusion", "D_cm2_s"),
    ],
)
def test_activation_energies_match_publication(material, quantity, source, column):
    """Arrhenius slopes over the full temperature series reproduce the paper.

    These are the numbers the manuscript states in text -- 0.36 eV for the
    productive rotation, 0.43(2) eV for CsH2PO4 proton diffusion -- and the
    argument that productive rotation is the rate-limiting step rests on the
    gap between them. Nothing else in the suite covers the multi-temperature
    fit: every other test pins a single representative temperature.

    Exact to the two decimals the paper quotes.
    """
    from superproton.residence import arrhenius_fit

    if source == "rotation_rates":
        path = REFERENCE / source / f"{material}_rates_2e-4_5e-5_count_10.csv"
        T_key = "T"
    else:
        path = REFERENCE / source / f"{material}_MLMD_diffusion.csv"
        T_key = "T_K"

    T, values = [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            T.append(float(row[T_key]))
            values.append(float(row[column]))

    fit = arrhenius_fit(np.array(T), np.array(values))
    expected = ACTIVATION_ENERGIES_EV[material][quantity]
    assert round(fit.E_a_eV, 2) == expected, (
        f"{material} {quantity}: E_a = {fit.E_a_eV:.4f} +/- {fit.stderr_E_a_eV:.4f} eV, "
        f"paper reports {expected} eV"
    )


def test_cdp_residence_rates_match_publication():
    """CsH2PO4's bi-exponential O-pair fit and single-exponential P-pair fit.

    The published 21 / 156 / 72 / 494 ps are quoted to the nearest ps, so each
    fitted value has to round to the published integer.
    """
    from superproton.residence import fit_piecewise, fit_single_exp

    tau_O, tau_P = _load_residence_taus("CDP", 525)
    expected = PUBLISHED_RATES["CDP"]

    piecewise = fit_piecewise(tau_O)
    assert round(1.0 / piecewise.lambda_1_per_ps) == expected["inv_lambda1"]
    assert round(1.0 / piecewise.lambda_2_per_ps) == expected["inv_lambda2"]
    assert round(piecewise.tau_c_ps) == expected["t_c"]

    single = fit_single_exp(tau_P)
    assert round(1.0 / single.lambda_per_ps) == expected["inv_lambda3"]


def test_chs_residence_rates_match_publication():
    """CsHSO4's O-pair distribution is single-exponential, unlike CsH2PO4's."""
    from superproton.residence import (
        DEFAULT_RHO_FLOOR_OO,
        fit_single_exp,
    )

    tau_O, tau_S = _load_residence_taus("CHS", 450)
    expected = PUBLISHED_RATES["CHS"]

    o_fit = fit_single_exp(tau_O, rho_floor=DEFAULT_RHO_FLOOR_OO, trim_to_peak=True)
    assert round(1.0 / o_fit.lambda_per_ps) == expected["inv_lambda1"]

    s_fit = fit_single_exp(tau_S)
    assert round(1.0 / s_fit.lambda_per_ps) == expected["inv_lambda3"]


def test_piecewise_seed_scan_is_reproduced_verbatim():
    """The tau_c seed governs which local optimum the four-parameter fit reaches.

    ``_seed_tau_c`` reproduces a quirk of the source script (the loss lags one
    candidate behind). Rewriting it "correctly" moves tau_c by ~3 ps and
    1/lambda_2 by ~2 ps off the published values, so the seed is pinned here.
    """
    from superproton.residence import _log_density, _seed_tau_c

    tau_O, _ = _load_residence_taus("CDP", 525)
    edges, log_rho = _log_density(tau_O, 2e-4, trim_to_peak=True)
    seed, _, _ = _seed_tau_c(edges, log_rho)
    assert round(seed) == PUBLISHED_RATES["CDP"]["t_c"]


@pytest.mark.parametrize("material,temperature", [("CDP", 525), ("CHS", 450)])
def test_reference_residence_counts_are_stable(material, temperature):
    """Guard the shipped reference CSVs themselves against accidental edits."""
    tau_O, tau_X = _load_residence_taus(material, temperature)
    expected = {("CDP", 525): (17999, 2094), ("CHS", 450): (24202, 7090)}
    assert (tau_O.size, tau_X.size) == expected[(material, temperature)]


def test_msd_fft_matches_the_naive_window_average():
    """The FFT estimator must equal the O(n^2) roll loop it replaces.

    ``superproton.msd.window_averaged_msd`` computes the same quantity as
    ``cal_msd.window_avg_msd`` in the source repo by FFT. Their agreement is
    what licenses the substitution, so it is asserted directly rather than
    trusted.
    """
    from superproton.msd import window_averaged_msd

    def naive(pos, frac):
        n = pos.shape[0]
        out = [
            np.mean((((np.roll(pos, -lag, axis=0) - pos)[: n - lag]) ** 2).sum(axis=2))
            for lag in range(int(frac * n))
        ]
        return np.array(out)[1:]

    rng = np.random.default_rng(0)
    positions = np.cumsum(rng.normal(size=(400, 11, 3)), axis=0)
    for frac in (0.7, 1.0):
        # Chunking is a memory knob and must not change the result.
        for chunk in (3, 64, 10_000):
            got = window_averaged_msd(positions, frac=frac, atom_chunk=chunk)
            np.testing.assert_allclose(got, naive(positions, frac), rtol=1e-10)


def test_published_msd_curve_yields_published_diffusivity():
    """D from the published MSD curve, within the estimator's window spread.

    D is sensitive to the fit window at the few-percent level -- larger than any
    single fit's standard error. The 6% tolerance is that spread, measured in
    ``superproton.msd.diffusivity``'s docstring, not a fudge factor.
    """
    from superproton.msd import diffusivity

    with (REFERENCE / "msd" / "CDP_525K_msd.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    time_ps = np.array([float(r["time_ps"]) for r in rows])
    msd_H = np.array([float(r["MSD_H_A2"]) for r in rows])

    published_D = 1.848e-07  # data/reference/diffusion/CDP_MLMD_diffusion.csv, 525 K
    fit = diffusivity(time_ps, msd_H, skip_ps=10.0)
    assert fit.D_cm2_s == pytest.approx(published_D, rel=0.06)
    assert fit.r2 > 0.99

    # An explicit window must be exactly reproducible, unlike the automatic scan.
    pinned = diffusivity(time_ps, msd_H, fit_range_ps=(100.0, 1000.0))
    assert pinned.D_cm2_s == pytest.approx(1.832e-07, rel=1e-3)


def test_oxygen_roles_follow_the_pair_transition():
    """carrier / acceptor / donor assignment at a pair switch.

    A proton on the oxygen pair {10, 20} that moves to {20, 30} keeps oxygen 20
    (the carrier), gains 30 (the acceptor) and leaves 10 (the donor).
    """
    from superproton.hopping import _roles_for_switches

    carrier, acceptor, donor = _roles_for_switches(
        np.array([[10, 20], [20, 30], [30, 40]])
    )
    np.testing.assert_array_equal(carrier, [20, 30])
    np.testing.assert_array_equal(acceptor, [30, 40])
    np.testing.assert_array_equal(donor, [10, 20])

    # A pair that turns over completely has no carrier and is dropped downstream.
    carrier, acceptor, donor = _roles_for_switches(np.array([[10, 20], [30, 40]]))
    np.testing.assert_array_equal(carrier, [-1])
    np.testing.assert_array_equal(acceptor, [-1])
    np.testing.assert_array_equal(donor, [-1])


# ---------------------------------------------------------------------------
# Tier B -- full pipeline from a production trajectory
# ---------------------------------------------------------------------------

CDP_TRJ = os.environ.get("SUPERPROTON_CDP_525K_TRJ")
CHS_TRJ = os.environ.get("SUPERPROTON_CHS_450K_TRJ")

needs_cdp = pytest.mark.skipif(
    not CDP_TRJ,
    reason="set SUPERPROTON_CDP_525K_TRJ to a 4 ns CsH2PO4 525 K dump (Zenodo 14649023)",
)
needs_chs = pytest.mark.skipif(
    not CHS_TRJ,
    reason="set SUPERPROTON_CHS_450K_TRJ to a 4 ns CsHSO4 450 K dump (Zenodo 14649023)",
)

# The paper drops 10 ps of equilibration; at 50 fs per frame that is 200 frames.
EQUILIBRATION_FRAMES = 200


def _pipeline_taus(dump: str, material: str):
    from superproton.io import read_lammps_dump
    from superproton.residence import collect_taus
    from superproton.rotation import classify_events
    from superproton.tracking import proton_neighbors

    pipe = read_lammps_dump(dump)
    tracks = proton_neighbors(
        pipe,
        material=material,
        start_frame=EQUILIBRATION_FRAMES,
        frame_to_ps=0.05,
        show_progress=False,
    )
    return collect_taus(classify_events(tracks))


@needs_cdp
def test_pipeline_reproduces_reference_residence_times_and_rates():
    """The whole chain: trajectory -> tracking -> classification -> tau -> rates.

    Two assertions over one trajectory walk. First that the residence times are
    exact, not approximate: event counts and every individual tau match the
    published intermediate arrays, so drift in the neighbour tracking or the
    pair-switch detector shows up immediately. Then that the fits of those taus
    land on the published rates.

    The rate half is implied by the tau half plus
    ``test_cdp_residence_rates_match_publication`` (same fit code, same input),
    but it is asserted here anyway: it is the single place that demonstrates
    trajectory in, published number out. It shares this test's trajectory walk
    rather than paying for a second one, which on the 4 ns dump costs about
    three minutes and 2 GB.
    """
    from superproton.residence import fit_piecewise, fit_single_exp

    taus = _pipeline_taus(CDP_TRJ, "CDP")
    reference_O, reference_P = _load_residence_taus("CDP", 525)

    assert taus.tau_OO_ps.size == reference_O.size
    assert taus.tau_PP_ps.size == reference_P.size
    # The reference CSV stores 4 decimal places, hence the absolute tolerance.
    np.testing.assert_allclose(
        np.sort(taus.tau_OO_ps), np.sort(reference_O), atol=5e-5
    )
    np.testing.assert_allclose(
        np.sort(taus.tau_PP_ps), np.sort(reference_P), atol=5e-5
    )

    expected = PUBLISHED_RATES["CDP"]
    piecewise = fit_piecewise(taus.tau_OO_ps)
    assert round(1.0 / piecewise.lambda_1_per_ps) == expected["inv_lambda1"]
    assert round(1.0 / piecewise.lambda_2_per_ps) == expected["inv_lambda2"]
    assert round(piecewise.tau_c_ps) == expected["t_c"]
    assert round(1.0 / fit_single_exp(taus.tau_PP_ps).lambda_per_ps) == expected["inv_lambda3"]


@pytest.mark.parametrize(
    "material,dump_env,n_frames",
    [
        pytest.param("CDP", "SUPERPROTON_CDP_525K_TRJ", 20_200, marks=needs_cdp),
        pytest.param("CHS", "SUPERPROTON_CHS_450K_TRJ", 40_200, marks=needs_chs),
    ],
)
def test_xoh_geometry_matches_table_s6(material, dump_env, n_frames):
    """Covalent X-O-H geometry over stable-bond intervals, versus SI Table S6.

    Uses a leading window of the trajectory rather than all 4 ns: the geometry
    is a per-frame equilibrium property with >10^5 samples in that window, so
    it converges long before the trajectory ends. ``n_frames`` differs by
    material because CsHSO4's 10 ps stable-bond dwell needs a longer window
    than CsH2PO4's 2 ps to accumulate intervals.

    Tolerances are the last digit the SI quotes: 0.001 A on bond lengths and
    0.1 degrees on the angle.
    """
    from superproton.correlation import polyanion_centers
    from superproton.geometry import xoh_geometry
    from superproton.io import read_lammps_dump
    from superproton.reorientation import read_HO_positions
    from superproton.tracking import proton_neighbors

    dump = os.environ[dump_env]
    dwell_ps = 2.0 if material == "CDP" else 10.0

    pipe = read_lammps_dump(dump)
    common = dict(
        material=material,
        start_frame=EQUILIBRATION_FRAMES,
        end_frame=n_frames,
        show_progress=False,
    )
    tracks = proton_neighbors(pipe, frame_to_ps=0.05, **common)
    positions = read_HO_positions(pipe, **common)
    centers = polyanion_centers(pipe, material, frame=EQUILIBRATION_FRAMES)

    geometry = xoh_geometry(positions, tracks, centers, dwell_ps=dwell_ps)
    expected = PUBLISHED_XOH[material]

    assert geometry.n_samples > 100_000
    assert geometry.d_OH_mean_A == pytest.approx(expected["d_OH"][0], abs=1e-3)
    assert geometry.d_OH_std_A == pytest.approx(expected["d_OH"][1], abs=1e-3)
    assert geometry.d_XO_mean_A == pytest.approx(expected["d_XO"][0], abs=1e-3)
    assert geometry.d_XO_std_A == pytest.approx(expected["d_XO"][1], abs=1e-3)
    assert geometry.angle_XOH_mean_deg == pytest.approx(expected["angle"][0], abs=0.1)
    assert geometry.angle_XOH_std_deg == pytest.approx(expected["angle"][1], abs=0.1)


@needs_cdp
def test_xoh_geometry_rejects_single_frame_polyanion_centers():
    """A frame-0 X position inflates the X-O spread ~7x and must not be accepted.

    This was a real defect: ``xoh_geometry`` used ``PolyanionCenters.positions``
    directly, giving d_XO = 1.579 +/- 0.335 A against the published
    1.609 +/- 0.046 A. The guard keeps it from coming back.
    """
    from dataclasses import replace

    from superproton.correlation import polyanion_centers
    from superproton.geometry import xoh_geometry
    from superproton.io import read_lammps_dump
    from superproton.reorientation import read_HO_positions
    from superproton.tracking import proton_neighbors

    pipe = read_lammps_dump(CDP_TRJ)
    common = dict(
        material="CDP", start_frame=EQUILIBRATION_FRAMES, end_frame=1_200,
        show_progress=False,
    )
    tracks = proton_neighbors(pipe, frame_to_ps=0.05, **common)
    positions = read_HO_positions(pipe, **common)
    centers = polyanion_centers(pipe, "CDP", frame=EQUILIBRATION_FRAMES)

    without_X = replace(positions, X_pos=None, X_ids=None)
    with pytest.raises(ValueError, match="per-frame polyanion-center positions"):
        xoh_geometry(without_X, tracks, centers, dwell_ps=2.0)


@needs_cdp
def test_species_msd_matches_published_curves():
    """Per-species MSD from the trajectory, against the published curves.

    Compares all four species over the 2.1-2800 ps window the paper plots.
    Outside that window the comparison is not meaningful: below 2.1 ps the
    motion is ballistic and above 2800 ps the estimator averages over too few
    time origins, which is exactly why the paper trims both ends.

    The 1% tolerance is loose relative to the measured agreement (median
    5e-5 to 6e-4, worst 2.5e-3) and exists so that a different FFT backend or
    a change in numpy's ``bins="fd"`` behaviour does not fail the suite.

    Reads every frame's positions, so this needs roughly 2 GB.
    """
    import csv as _csv

    from superproton.io import read_lammps_dump
    from superproton.msd import species_msd

    pipe = read_lammps_dump(CDP_TRJ)
    # frac=1.0 reports every lag, as the published CSVs do.
    computed = species_msd(
        pipe, material="CDP", frame_to_ps=0.05, frac=1.0, show_progress=False
    )

    with (REFERENCE / "msd" / "CDP_525K_msd.csv").open() as fh:
        rows = list(_csv.DictReader(fh))
    published_t = np.array([float(r["time_ps"]) for r in rows])

    n = min(len(rows), computed.time_ps.size)
    window = (computed.time_ps[:n] >= 2.1) & (computed.time_ps[:n] <= 2800.0)
    assert window.sum() > 10_000, "plot window should cover most of the trajectory"
    np.testing.assert_allclose(computed.time_ps[:n], published_t[:n], rtol=1e-12)

    for element in ("Cs", "P", "H", "O"):
        published = np.array([float(r[f"MSD_{element}_A2"]) for r in rows])[:n]
        got = computed.msd_A2[element][:n]
        np.testing.assert_allclose(
            got[window], published[window], rtol=0.01,
            err_msg=f"{element} MSD disagrees with the published curve",
        )

    # Atom counts follow from the 5x5x5 CsH2PO4 supercell stoichiometry.
    assert computed.n_atoms == {"Cs": 125, "H": 250, "P": 125, "O": 500}


@needs_cdp
def test_pointing_outcomes_match_the_origin_algorithm():
    """Pointing attempt counts and outcome fractions, on a 500 ps window.

    Pinned against the origin script
    (``residence_times/pointing_transfer_prod.py``) run over the same frames,
    which gives 691 decided attempts with 89.6% committed. The shipped
    reference CSV is a full 4 ns run (3030 attempts, 84.4% committed) and is
    not directly comparable to a 500 ps window: the first 500 ps runs about
    1.6x above the 4 ns mean attempt rate because only 10 ps of equilibration
    is dropped.

    Two defects used to sit here, and the tolerances below are what separates
    the fixed code from either of them:

    - ``hbond_distances`` measured raw separations on unwrapped coordinates,
      so 13% of covalent O-H "bonds" came out as lattice vectors up to 35 A,
      permanently breaking the hysteresis.
    - the hysteresis substituted the proton's own donor polyanion whenever no
      acceptor was committed, collapsing the pair to a degenerate {A, A}.

    Together they gave 15995 attempts at 14.4% committed -- a 23x over-count
    with the commit/revert ratio inverted.
    """
    from superproton.io import cell, read_lammps_dump
    from superproton.pointing import (
        detect_attempts,
        hbond_distances,
        outcome_fractions,
    )
    from superproton.reorientation import read_HO_positions
    from superproton.tracking import proton_neighbors

    pipe = read_lammps_dump(CDP_TRJ)
    common = dict(
        material="CDP", start_frame=EQUILIBRATION_FRAMES, end_frame=10_200,
        show_progress=False,
    )
    tracks = proton_neighbors(pipe, frame_to_ps=0.05, **common)
    positions = read_HO_positions(pipe, with_X=False, **common)
    distances = hbond_distances(positions, tracks, cell=cell(pipe))

    # A covalent O-H bond is about 1 A. Anything near a box edge means the
    # minimum-image folding is not being applied.
    assert distances.donor_O_dist.max() < 2.0

    data = detect_attempts(tracks, distances)
    fractions = outcome_fractions(data)
    assert len(data.attempts) == pytest.approx(691, rel=0.06)
    assert fractions["committed"] == pytest.approx(0.896, abs=0.02)
    assert fractions["reverted"] < 0.12
    # No attempt may open on the "no acceptor committed yet" sentinel: a
    # proton's first commitment starts its history rather than pointing away
    # from an earlier pair. A momentary {A, A} pair from two nearest oxygens on
    # the same polyanion is legitimate and is NOT excluded here.
    assert all(a.polyanion_pair_at_fresh != (-1, -1) for a in data.attempts)


@needs_cdp
def test_dhdo_ratios_match_the_published_distributions():
    """dH/dO baseline, outliers and cos(dH, dO), against the reference CSVs.

    ``compute_reorientation`` used to read the carrier oxygen at ``t_f``, the
    switch frame itself, which is by construction the oxygen the proton lands
    on *after* the switch rather than the one it rode through the rotation.
    It therefore measured the proton against a spectator oxygen, inflating the
    baseline to 1.79 and pushing CsHSO4's baseline above CsH2PO4's rigid
    co-rotation value, which inverts the ordering the paper argues from.
    """
    from superproton.io import read_lammps_dump
    from superproton.reorientation import compute_reorientation, read_HO_positions
    from superproton.rotation import classify_events
    from superproton.tracking import proton_neighbors

    pipe = read_lammps_dump(CDP_TRJ)
    common = dict(
        material="CDP", start_frame=EQUILIBRATION_FRAMES, end_frame=10_200,
        show_progress=False,
    )
    tracks = proton_neighbors(pipe, frame_to_ps=0.05, **common)
    positions = read_HO_positions(pipe, with_X=False, **common)
    result = compute_reorientation(positions, tracks, classify_events(tracks))

    finite = ~np.isnan(result.baseline)
    assert finite.sum() > 1000
    # Reference mass-weighted mean from CDP_525K_dHdO_ratio.csv is 1.633.
    assert result.baseline[finite].mean() == pytest.approx(1.633, rel=0.02)
    # The baseline must sit just above the rigid co-rotation value of 1.40.
    assert 1.40 < result.baseline[finite].mean() < 1.80

    has_cos = ~np.isnan(result.cos_HO)
    # The published cos(dH, dO) distribution has mean +0.520: reorientation
    # acts with the carrier, not against it (SI Figure cos_HO).
    assert result.cos_HO[has_cos].mean() == pytest.approx(0.520, abs=0.03)


@pytest.mark.parametrize('temperature',[525,550,575])
def test_si10_fits_reproduce_archived_comparison(temperature):
    """Pin the distinct SI policy to the original-script-validated fit curves."""
    from superproton.residence import residence_fit_comparison
    with np.load(REFERENCE/'residence_comparison'/f'CDP_{temperature}K_comparison.npz') as saved:
        fitted=residence_fit_comparison(saved['tau_ps'])
        for key in ('piecewise_parameters','biexp_parameters','piecewise_survival','biexp_survival'):
            np.testing.assert_allclose(fitted[key],saved[key],rtol=1e-5,atol=1e-10)


@pytest.mark.parametrize('material',['CDP','CHS'])
def test_orientation_reference_is_normalized_angular_density(material):
    for scope in ('single','all'):
        with np.load(REFERENCE/'orientation'/f'{material}_550K_{scope}.npz') as d:
            integral=np.sum(d['density']*np.diff(d['phi_edges_deg'])[:,None]
                            *np.diff(d['theta_edges_deg'])[None,:])
            assert integral == pytest.approx(1,abs=1e-12)
