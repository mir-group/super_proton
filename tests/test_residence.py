"""Synthetic-event validation of superproton.residence.

These tests construct ``RotationEvents`` from a thinned Poisson process with
known overall rate, productive probability, and number of protons. Every fit
in ``residence.py`` is then asserted to recover the synthetic ground truth
within a tolerance that reflects the available sample size. This locks in the
behavior of the residence-time pipeline before any real-trajectory tracking
code (the ``rotation`` / ``tracking`` modules) is wired up.
"""

from __future__ import annotations

import numpy as np
import pytest

from superproton.residence import (
    K_B_EV_PER_K,
    arrhenius_fit,
    collect_taus,
    fit_biexp_survival,
    fit_piecewise,
    fit_single_exp,
    global_average_rate,
    nonproductive_rate,
    productive_probability,
)
from superproton.rotation import RotationEvents


def _make_events(
    *,
    n_protons: int = 200,
    sim_time_ps: float = 4000.0,
    lambda_o_per_ps: float = 0.05,
    productive_p: float = 0.10,
    frame_to_ps: float = 0.05,
    seed: int = 0,
) -> RotationEvents:
    """Thinned Poisson process: O-switches at rate lambda_o, marked productive with prob p."""
    rng = np.random.default_rng(seed)
    n_frames = int(sim_time_ps / frame_to_ps) + 1

    o_frames: list[np.ndarray] = []
    p_frames: list[np.ndarray] = []
    prod_ts: list[np.ndarray] = []
    nonprod_ts: list[np.ndarray] = []
    for _ in range(n_protons):
        n_expected = int(3 * lambda_o_per_ps * sim_time_ps)
        gaps_ps = rng.exponential(1.0 / lambda_o_per_ps, size=n_expected)
        arrivals_ps = np.cumsum(gaps_ps)
        arrivals_ps = arrivals_ps[arrivals_ps < sim_time_ps]
        arrivals_fr = arrivals_ps / frame_to_ps
        # Quantize to integer frames and drop duplicates that fall in the same bin.
        arrivals_fr = np.unique(arrivals_fr.astype(int))
        is_productive = rng.random(arrivals_fr.size) < productive_p
        o_frames.append(arrivals_fr)
        p_frames.append(arrivals_fr[is_productive])
        prod_ts.append(np.clip(arrivals_fr[is_productive] - 1, 0, n_frames - 1))
        nonprod_ts.append(np.clip(arrivals_fr[~is_productive] - 1, 0, n_frames - 1))

    # Synthetic pair ids: arbitrary placeholders. They don't affect any of
    # the residence-time fits, which only consume switch frames.
    placeholder_pairs = [np.zeros((0, 2), dtype=np.int64) for _ in range(n_protons)]
    return RotationEvents(
        n_protons=n_protons,
        n_frames=n_frames,
        frame_to_ps=frame_to_ps,
        o_pair_switch_frames=o_frames,
        p_pair_switch_frames=p_frames,
        productive_t_s_frames=prod_ts,
        nonproductive_t_s_frames=nonprod_ts,
        o_pair_ids=placeholder_pairs,
        p_pair_ids=[a.copy() for a in placeholder_pairs],
    )


# ---------------------------------------------------------------------------
# collect_taus
# ---------------------------------------------------------------------------


def test_collect_taus_recovers_mean_residence_times():
    events = _make_events()
    tau = collect_taus(events)

    expected_mean_OO = 1.0 / 0.05  # 20 ps
    expected_mean_PP = 1.0 / (0.05 * 0.10)  # 200 ps

    assert tau.tau_OO_ps.size > 5_000
    assert tau.tau_PP_ps.size > 500
    assert abs(tau.tau_OO_ps.mean() - expected_mean_OO) / expected_mean_OO < 0.05
    assert abs(tau.tau_PP_ps.mean() - expected_mean_PP) / expected_mean_PP < 0.10


def test_collect_taus_categorization_sizes_add_up():
    events = _make_events()
    tau = collect_taus(events)
    assert (
        tau.tau_OO_PP_ps.size
        + tau.tau_OO_NN_ps.size
        + tau.tau_OO_PN_ps.size
        == tau.tau_OO_ps.size
    )
    # For p = 0.10 the (NN, PN, PP) split is roughly (0.81, 0.18, 0.01).
    n = tau.tau_OO_ps.size
    assert tau.tau_OO_NN_ps.size / n == pytest.approx(0.81, abs=0.03)
    assert tau.tau_OO_PN_ps.size / n == pytest.approx(0.18, abs=0.03)
    assert tau.tau_OO_PP_ps.size / n == pytest.approx(0.01, abs=0.01)


# ---------------------------------------------------------------------------
# fit_single_exp
# ---------------------------------------------------------------------------


def test_fit_single_exp_recovers_known_lambda_PP():
    events = _make_events()
    tau = collect_taus(events)
    fit = fit_single_exp(tau.tau_PP_ps, rho_floor=1e-5)
    expected = 0.005  # 1/ps
    assert abs(fit.lambda_per_ps - expected) / expected < 0.10
    assert fit.r2 > 0.95


def test_fit_single_exp_recovers_known_lambda_OO():
    events = _make_events()
    tau = collect_taus(events)
    # OO is a single-rate Poisson process in this synthetic, so single-exp must work.
    fit = fit_single_exp(tau.tau_OO_ps, rho_floor=1e-4)
    expected = 0.05
    assert abs(fit.lambda_per_ps - expected) / expected < 0.05
    assert fit.r2 > 0.98


# ---------------------------------------------------------------------------
# fit_piecewise
# ---------------------------------------------------------------------------


def test_fit_piecewise_runs_on_single_rate_and_recovers_fast_branch():
    """On a true single-rate process the piecewise model is over-parameterized;
    the slow branch is unconstrained, but the fast (well-sampled) branch must
    still recover the truth."""
    events = _make_events()
    tau = collect_taus(events)
    fit = fit_piecewise(tau.tau_OO_ps, rho_floor=1e-4)
    expected = 0.05
    assert fit.lambda_1_per_ps == pytest.approx(expected, rel=0.20)


def test_fit_piecewise_on_two_rate_mixture():
    """An 80/20 mixture with a 10x rate ratio should yield clearly separated branches."""
    rng = np.random.default_rng(1)
    n = 50_000
    lam_fast, lam_slow = 0.05, 0.005
    tau = np.concatenate(
        [
            rng.exponential(1.0 / lam_fast, size=int(0.8 * n)),
            rng.exponential(1.0 / lam_slow, size=int(0.2 * n)),
        ]
    )
    fit = fit_piecewise(tau, rho_floor=1e-5)
    # Fast (small-tau) branch is lambda_1; slow tail branch is lambda_2.
    assert fit.lambda_1_per_ps > fit.lambda_2_per_ps
    assert fit.lambda_1_per_ps == pytest.approx(lam_fast, rel=0.40)
    assert fit.lambda_2_per_ps == pytest.approx(lam_slow, rel=0.40)
    assert 10 < fit.tau_c_ps < 500


# ---------------------------------------------------------------------------
# fit_biexp_survival
# ---------------------------------------------------------------------------


def test_fit_biexp_survival_two_rate_mixture():
    rng = np.random.default_rng(2)
    n = 10_000
    a_true = 0.8
    lam_fast = 0.1
    lam_slow = 0.005
    tau = np.concatenate(
        [
            rng.exponential(1.0 / lam_fast, size=int(a_true * n)),
            rng.exponential(1.0 / lam_slow, size=int((1 - a_true) * n)),
        ]
    )
    fit = fit_biexp_survival(tau)
    assert fit.amplitude == pytest.approx(a_true, abs=0.10)
    assert fit.lambda_1_per_ps == pytest.approx(lam_fast, rel=0.25)
    assert fit.lambda_2_per_ps == pytest.approx(lam_slow, rel=0.30)
    # Canonicalization: lambda_1 must be the larger rate.
    assert fit.lambda_1_per_ps > fit.lambda_2_per_ps


# ---------------------------------------------------------------------------
# productive_probability
# ---------------------------------------------------------------------------


def test_productive_probability_recovers_known_p():
    events = _make_events()
    prob = productive_probability(events, min_bin_count=20)
    assert prob.p == pytest.approx(0.10, abs=0.02)
    assert prob.counts.size > 1_000


# ---------------------------------------------------------------------------
# nonproductive_rate
# ---------------------------------------------------------------------------


def test_nonproductive_rate_formula():
    assert nonproductive_rate(0.005, 0.10) == pytest.approx(0.045, abs=1e-9)
    with pytest.raises(ValueError):
        nonproductive_rate(0.005, 0.0)
    with pytest.raises(ValueError):
        nonproductive_rate(0.005, 1.0)


# ---------------------------------------------------------------------------
# global_average_rate
# ---------------------------------------------------------------------------


def test_global_average_rate_matches_synthetic_rates():
    events = _make_events()
    g = global_average_rate(events)
    assert g["overall"] == pytest.approx(0.05, rel=0.05)
    assert g["productive"] == pytest.approx(0.005, rel=0.10)
    assert g["nonproductive"] == pytest.approx(0.045, rel=0.05)


# ---------------------------------------------------------------------------
# arrhenius_fit
# ---------------------------------------------------------------------------


def test_arrhenius_fit_recovers_known_E_a():
    rng = np.random.default_rng(3)
    E_a_true = 0.43  # eV
    A_true = 1e5  # per ps
    T = np.array([525, 540, 550, 575, 600, 625, 650], dtype=float)
    rates = A_true * np.exp(-E_a_true / (K_B_EV_PER_K * T))
    rates *= 1.0 + 0.05 * rng.standard_normal(T.size)
    fit = arrhenius_fit(T, rates)
    assert fit.E_a_eV == pytest.approx(E_a_true, abs=0.03)


def test_arrhenius_fit_rejects_nonpositive_rates():
    T = np.array([300.0, 400.0])
    with pytest.raises(ValueError):
        arrhenius_fit(T, np.array([1e-3, -1e-3]))
