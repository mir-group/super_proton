"""Residence-time histograms and rate fits.

Consumes :class:`superproton.rotation.RotationEvents` and produces the
residence-time distributions, exponential fits, productive-probability
estimate, and Arrhenius activation energies reported in the manuscript.

Fit conventions

- Two-rate fits (:func:`fit_piecewise`, :func:`fit_biexp_survival`) accept any
  tau array, but the published analysis only applies them to ``tau_O-O``.
  ``tau_P-P`` is fit single-exponentially.
- The density tail floor (``rho_floor``) is an explicit keyword argument.
  Defaults reproduce the paper exactly:
  ``rho_floor=2e-4`` for O-pair, ``rho_floor=5e-5`` for P-pair.
- Inputs are pre-computed events; no trajectory walking happens here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

from superproton.rotation import RotationEvents

K_B_EV_PER_K = 8.617333262e-5
DEFAULT_RHO_FLOOR_OO = 2e-4
DEFAULT_RHO_FLOOR_PP = 5e-5


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidenceTimes:
    """Residence-time arrays in picoseconds.

    All arrays concatenate intervals across every proton in the events.
    Intervals that touch a trajectory boundary are dropped (collection is
    strictly between consecutive switches per proton).
    """

    tau_OO_ps: np.ndarray
    tau_PP_ps: np.ndarray
    tau_OO_PP_ps: np.ndarray  # both bounding switches productive
    tau_OO_NN_ps: np.ndarray  # both bounding switches nonproductive
    tau_OO_PN_ps: np.ndarray  # one of each


@dataclass(frozen=True)
class SingleExpFit:
    """Single-exponential fit of the log-density of residence times."""

    lambda_per_ps: float
    intercept: float
    r2: float
    stderr_lambda: float


@dataclass(frozen=True)
class PiecewiseFit:
    """Piecewise-linear fit of ``log rho(tau)`` with a transition at ``tau_c``."""

    lambda_1_per_ps: float  # fast rate (small tau)
    lambda_2_per_ps: float  # slow rate (large tau)
    tau_c_ps: float
    intercept: float
    stderr_lambda_1: float
    stderr_lambda_2: float
    stderr_tau_c: float


@dataclass(frozen=True)
class BiExpFit:
    """Bi-exponential survival-function fit ``S = A exp(-l1 t) + (1-A) exp(-l2 t)``."""

    amplitude: float  # weight A of the fast component, in [0, 1]
    lambda_1_per_ps: float
    lambda_2_per_ps: float
    stderr_amplitude: float
    stderr_lambda_1: float
    stderr_lambda_2: float


@dataclass(frozen=True)
class ProductiveProb:
    """Geometric-distribution fit for the productive-event probability ``p``."""

    p: float
    stderr_p: float
    counts: np.ndarray  # observed counts of N nonproductive events per productive interval


@dataclass(frozen=True)
class ArrheniusFit:
    """Arrhenius fit ``log lambda = log A - E_a / (k_B T)``."""

    E_a_eV: float
    A_per_ps: float
    stderr_E_a_eV: float


# ---------------------------------------------------------------------------
# 1. collect_taus
# ---------------------------------------------------------------------------


def collect_taus(events: RotationEvents) -> ResidenceTimes:
    """Build O-pair and P-pair residence times, plus the categorized O-pair subsets.

    For each proton, residence intervals are taken strictly between
    consecutive switches; opening (frame 0 to first switch) and closing
    (last switch to ``n_frames - 1``) intervals are excluded per the
    manuscript convention.
    """
    dt = float(events.frame_to_ps)

    oo_chunks: list[np.ndarray] = []
    pp_chunks: list[np.ndarray] = []
    oo_pp: list[np.ndarray] = []
    oo_nn: list[np.ndarray] = []
    oo_pn: list[np.ndarray] = []

    for h in range(events.n_protons):
        o_sw = np.asarray(events.o_pair_switch_frames[h], dtype=int)
        p_sw = np.asarray(events.p_pair_switch_frames[h], dtype=int)

        if o_sw.size >= 2:
            tau_oo_h = (o_sw[1:] - o_sw[:-1]).astype(float) * dt
            oo_chunks.append(tau_oo_h)

            is_productive = np.isin(o_sw, p_sw)
            left = is_productive[:-1]
            right = is_productive[1:]
            oo_pp.append(tau_oo_h[left & right])
            oo_nn.append(tau_oo_h[(~left) & (~right)])
            oo_pn.append(tau_oo_h[left ^ right])

        if p_sw.size >= 2:
            pp_chunks.append((p_sw[1:] - p_sw[:-1]).astype(float) * dt)

    return ResidenceTimes(
        tau_OO_ps=_concat(oo_chunks),
        tau_PP_ps=_concat(pp_chunks),
        tau_OO_PP_ps=_concat(oo_pp),
        tau_OO_NN_ps=_concat(oo_nn),
        tau_OO_PN_ps=_concat(oo_pn),
    )


def _concat(chunks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=float)


# ---------------------------------------------------------------------------
# Histogram utilities
# ---------------------------------------------------------------------------


def _log_density(
    tau: np.ndarray,
    rho_floor: float,
    *,
    trim_to_peak: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Freedman-Diaconis log-density of ``tau``, prepared for a log-linear fit.

    The manuscript's quoted rates use the following histogram and selection
    conventions:

    1. Only strictly positive residence times enter the histogram.
    2. Bins come from ``np.histogram(..., bins="fd", density=True)``.
    3. The fit abscissa is the **right** bin edge, not the bin center. The
       plotted fit lines in the paper are drawn against bin edges, so the
       regression has to use the same abscissa.
    4. The last 10% of bins are dropped (poorly sampled tail).
    5. Bins whose density falls below ``rho_floor`` are dropped.
    6. For O-pair distributions the leading rise up to the density peak is
       dropped (``trim_to_peak``); the polyanion-pair distribution decays
       monotonically from the first bin and is not trimmed.

    Returns
    -------
    (edges, log_rho)
        Right bin edges and the natural log of the density in those bins.
    """
    tau = np.asarray(tau, dtype=float)
    pdf, edges = np.histogram(tau[tau > 0], bins="fd", density=True)
    keep = pdf > 0
    x = edges[1:][keep]
    y = np.log(pdf[keep])

    # Drop the poorly averaged tail. NOTE: the slice length is derived from the
    # full edge array, not from the filtered arrays.
    cut = int(len(edges) * 0.9)
    x, y = x[:cut], y[:cut]

    above = y > np.log(rho_floor)
    x, y = x[above], y[above]

    if trim_to_peak and y.size:
        peak = int(np.argmax(y))
        x, y = x[peak:], y[peak:]

    if x.size < 3:
        raise ValueError(
            f"Only {x.size} bins survive rho_floor={rho_floor}; "
            "lower the floor or supply more data."
        )
    return x, y


def _seed_tau_c(
    x: np.ndarray, y: np.ndarray, start_idx: int = 4
) -> tuple[float, object, object]:
    """Locate a seed crossover ``tau_c`` for the piecewise fit.

    The Figure 4a scan advances a candidate crossover one bin at a time,
    accepting it when the loss ``R^2(left) - R^2(right)`` does not increase.
    Both regressions are evaluated at the *running* ``mid`` rather than at
    the ``candidate``, so the loss lags one step behind.

    The result seeds ``curve_fit`` in :func:`fit_piecewise`, which refines all
    four parameters. This fit is not globally convex, so the seed affects the
    local optimum and the resulting rates.

    The running-mid and candidate-split conventions give different fits.
    The candidate-split variant gives ``1/lambda_1 = 20.6 ps``,
    ``1/lambda_2 = 153.8 ps`` and ``tau_c = 68.8 ps``; the running-mid variant
    gives 21.1 / 155.5 / 72.0 ps, matching the paper's rounded values
    (21 / 156 / 72). The running-mid convention is therefore required to
    reproduce the main-text fit.
    """
    def loss(r_left, r_right) -> float:
        return -(r_right.rvalue ** 2) + (r_left.rvalue ** 2)

    mid = x[start_idx]
    res_left = linregress(x[x < mid], y[x < mid])
    res_right = linregress(x[x >= mid], y[x >= mid])
    best = loss(res_left, res_right)

    for i in range(start_idx + 1, len(x) - 5):
        candidate = x[i]
        r_left = linregress(x[x < mid], y[x < mid])
        r_right = linregress(x[x >= mid], y[x >= mid])
        current = loss(r_left, r_right)
        if current <= best:
            best, mid, res_left, res_right = current, candidate, r_left, r_right

    return float(mid), res_left, res_right


# ---------------------------------------------------------------------------
# 2. fit_single_exp
# ---------------------------------------------------------------------------


def fit_single_exp(
    tau_ps: np.ndarray,
    *,
    rho_floor: float = DEFAULT_RHO_FLOOR_PP,
    trim_to_peak: bool = False,
) -> SingleExpFit:
    """Single-rate fit of ``log rho(tau) = -lambda * tau + b``.

    Default ``rho_floor`` matches the paper's polyanion-pair convention
    (``5e-5``). For an O-pair distribution -- CsHSO4's tau_O-O, which is
    single-exponential -- pass ``rho_floor=DEFAULT_RHO_FLOOR_OO``.

    ``trim_to_peak`` drops the leading rise up to the density peak. The paper
    applies it to O-pair distributions only; see :func:`_log_density`.
    """
    edges, log_rho = _log_density(tau_ps, rho_floor, trim_to_peak=trim_to_peak)
    result = linregress(edges, log_rho)
    return SingleExpFit(
        lambda_per_ps=float(-result.slope),
        intercept=float(result.intercept),
        r2=float(result.rvalue**2),
        stderr_lambda=float(result.stderr),
    )


# ---------------------------------------------------------------------------
# 3. fit_piecewise
# ---------------------------------------------------------------------------


def _piecewise_log_rho(
    tau: np.ndarray, k1: float, k2: float, tau_c: float, b1: float
) -> np.ndarray:
    return np.where(tau < tau_c, k1 * tau + b1, k2 * (tau - tau_c) + k1 * tau_c + b1)


def fit_piecewise(
    tau_ps: np.ndarray,
    *,
    rho_floor: float = DEFAULT_RHO_FLOOR_OO,
    trim_to_peak: bool = True,
    comparison: bool = False,
) -> PiecewiseFit:
    """Piecewise-linear fit of ``log rho(tau)`` with a kink at ``tau_c``.

    Used for CsH2PO4's ``tau_O-O``, whose log-density is bi-exponential: a fast
    branch ``lambda_1`` below the crossover ``tau_c`` and a slow branch
    ``lambda_2`` above it.

    ``comparison=True`` selects the SI S10 policy: retain the complete
    histogram tail and evaluate regressions at each candidate split. The
    default uses the main-text tail trimming and running-mid crossover scan
    described in :func:`_log_density` and :func:`_seed_tau_c`.

    The seed crossover, both slopes, and the intercept are refined together
    by a four-parameter ``curve_fit``.
    """
    if comparison:
        # SI S10 source keeps the complete histogram tail and scans the actual
        # candidate split; the main-text defaults are deliberately unchanged.
        tau = np.asarray(tau_ps)
        pdf, all_edges = np.histogram(tau[tau > 0], bins="fd", density=True)
        keep = pdf > rho_floor
        edges, log_rho = all_edges[1:][keep], np.log(pdf[keep])
        if trim_to_peak and len(log_rho):
            peak = np.argmax(log_rho)
            edges, log_rho = edges[peak:], log_rho[peak:]
    else:
        edges, log_rho = _log_density(tau_ps, rho_floor, trim_to_peak=trim_to_peak)
    if edges.size < (12 if comparison else 10):
        raise ValueError("Too few populated bins for a piecewise fit")
    if comparison:
        best = float("inf")
        for mid in edges[4:-5]:
            left = linregress(edges[edges < mid], log_rho[edges < mid])
            right = linregress(edges[edges >= mid], log_rho[edges >= mid])
            loss = -right.rvalue**2 + left.rvalue**2
            if loss <= best:
                best, seed_tau_c, res_left, res_right = loss, mid, left, right
    else:
        seed_tau_c, res_left, res_right = _seed_tau_c(edges, log_rho)
    p0 = [res_left.slope, res_right.slope, seed_tau_c, res_left.intercept]
    popt, pcov = curve_fit(
        _piecewise_log_rho, edges, log_rho, p0=p0, maxfev=20_000
    )
    k1, k2, tau_c, b1 = popt
    perr = np.sqrt(np.diag(pcov))

    return PiecewiseFit(
        lambda_1_per_ps=float(-k1),
        lambda_2_per_ps=float(-k2),
        tau_c_ps=float(tau_c),
        intercept=float(b1),
        stderr_lambda_1=float(perr[0]),
        stderr_lambda_2=float(perr[1]),
        stderr_tau_c=float(perr[2]),
    )


# ---------------------------------------------------------------------------
# 4. fit_biexp_survival
# ---------------------------------------------------------------------------


def _biexp_survival(
    tau: np.ndarray, amplitude: float, lambda_1: float, lambda_2: float
) -> np.ndarray:
    return amplitude * np.exp(-lambda_1 * tau) + (1.0 - amplitude) * np.exp(
        -lambda_2 * tau
    )


def fit_biexp_survival(tau_ps: np.ndarray, *, initial=None) -> BiExpFit:
    """Two-exponential mixture fit to the empirical survival function.

    The empirical survival is ``S(tau_i) = (n - i) / n`` on sorted ``tau``.
    The model has bounds ``amplitude in [0, 1]`` and ``lambda_j >= 0``.
    """
    tau = np.sort(np.asarray(tau_ps, dtype=float))
    n = tau.size
    if n < 8:
        raise ValueError("Need at least 8 residence times for a bi-exp fit.")
    survival = (n - np.arange(n)) / n
    median = float(np.median(tau))
    p0 = [0.5, 1.0 / max(median, 1e-6), 1.0 / max(10.0 * median, 1e-6)]
    if initial is not None:
        p0 = initial
    bounds = ([0.0, 0.0, 0.0], [1.0, np.inf, np.inf])
    popt, pcov = curve_fit(
        _biexp_survival, tau, survival, p0=p0, bounds=bounds, maxfev=30_000
    )
    amplitude, lam1, lam2 = popt
    # Canonicalize: lambda_1 is the larger (faster) rate.
    if lam2 > lam1:
        amplitude = 1.0 - amplitude
        lam1, lam2 = lam2, lam1
        # Swap the corresponding covariance entries.
        pcov = pcov[[0, 2, 1]][:, [0, 2, 1]]
    perr = np.sqrt(np.diag(pcov))
    return BiExpFit(
        amplitude=float(amplitude),
        lambda_1_per_ps=float(lam1),
        lambda_2_per_ps=float(lam2),
        stderr_amplitude=float(perr[0]),
        stderr_lambda_1=float(perr[1]),
        stderr_lambda_2=float(perr[2]),
    )


# ---------------------------------------------------------------------------
# 5. productive_probability
# ---------------------------------------------------------------------------


def productive_probability(
    events: RotationEvents, *, min_bin_count: int = 10, min_nonproductive: int = 2
) -> ProductiveProb:
    """Estimate the per-rotation productive probability ``p`` via geometric fit.

    For each proton, we walk its sorted O-pair switch frames and count the
    number of consecutive nonproductive events between productive ones. Under
    a Bernoulli model, that count follows ``P(N=n) = (1-p)^n * p``, so
    ``log P(N=n)`` is linear in ``n`` with slope ``log(1-p)``. The fit uses
    intervals with N >= 2 and bins occurring at least ten times by default,
    following the published selection. Set ``min_nonproductive=0`` to include
    every complete interval.
    This is a model-based estimate, not a test of event independence.
    """
    return fit_productive_probability(
        nonproductive_counts(events), min_bin_count=min_bin_count,
        min_nonproductive=min_nonproductive,
    )


def fit_productive_probability(
    counts, *, min_bin_count: int = 10, min_nonproductive: int = 2
) -> ProductiveProb:
    """Fit saved nonproductive counts using the published selection convention.

    The published selection excludes intervals with fewer than two NP events.
    Normalization over retained intervals changes the intercept, not the slope
    used for p. The count threshold is inclusive.
    Returned ``counts`` are the retained samples; :func:`nonproductive_counts`
    separately returns all complete intervals, including zero and one.
    """
    counts_array = np.asarray(counts)
    if (counts_array.ndim != 1 or not np.all(np.isfinite(counts_array))
            or np.any(counts_array < 0) or np.any(counts_array != np.floor(counts_array))):
        raise ValueError("Counts must be a one-dimensional array of nonnegative integers")
    if type(min_bin_count) is not int or min_bin_count < 1:
        raise ValueError("min_bin_count must be a positive integer")
    if type(min_nonproductive) is not int or min_nonproductive < 0:
        raise ValueError("min_nonproductive must be a nonnegative integer")
    counts_array = counts_array[counts_array >= min_nonproductive].astype(int)
    if not counts_array.size:
        raise ValueError("Not enough productive intervals to estimate p.")
    n_values, hist = np.unique(counts_array, return_counts=True)
    mask = hist >= min_bin_count
    if mask.sum() < 2:
        raise ValueError(
            f"Only {int(mask.sum())} bins with count >= {min_bin_count}; "
            "lower the threshold or supply more data."
        )
    n_values = n_values[mask]
    log_prob = np.log(hist[mask] / hist.sum())
    fit = linregress(n_values, log_prob)
    p = 1.0 - np.exp(fit.slope)
    if not 0 < p < 1:
        raise ValueError("Geometric fit does not give a probability in (0, 1)")
    # Propagate the slope stderr to p via dp/dslope = -exp(slope).
    stderr_p = float(np.exp(fit.slope) * fit.stderr)
    return ProductiveProb(p=float(p), stderr_p=stderr_p, counts=counts_array)


# ---------------------------------------------------------------------------
# 6. nonproductive_rate
# ---------------------------------------------------------------------------


def nonproductive_rate(lambda_P_per_ps: float, p: float) -> float:
    """Derive the nonproductive rotation rate ``lambda_NP = lambda_P (1 - p) / p``."""
    if not 0 < p < 1:
        raise ValueError(f"p must be in (0, 1); got {p}.")
    return float(lambda_P_per_ps * (1.0 - p) / p)


# ---------------------------------------------------------------------------
# 7. global_average_rate
# ---------------------------------------------------------------------------


def global_average_rate(events: RotationEvents) -> dict[str, float]:
    """Model-free rates: total events / (simulated time * number of protons).

    Returns a dict with ``overall``, ``productive``, ``nonproductive`` rates
    (all in ``1/ps``). Provides a sanity check against the model-derived
    rates from the residence-time fits.
    """
    t_sim_ps = float((events.n_frames - 1) * events.frame_to_ps)
    n_overall = sum(arr.size for arr in events.o_pair_switch_frames)
    n_productive = sum(arr.size for arr in events.p_pair_switch_frames)
    n_nonproductive = n_overall - n_productive
    norm = t_sim_ps * events.n_protons
    return {
        "overall": n_overall / norm,
        "productive": n_productive / norm,
        "nonproductive": n_nonproductive / norm,
    }


# ---------------------------------------------------------------------------
# 8. arrhenius_fit
# ---------------------------------------------------------------------------


def arrhenius_fit(
    temperatures_K: np.ndarray, rates_per_ps: np.ndarray
) -> ArrheniusFit:
    """Fit ``log(rate) = log A - E_a / (k_B T)`` and return the activation energy.

    Both inputs must be 1D arrays of the same length, sampled at multiple
    temperatures. ``E_a`` is reported in eV.
    """
    T = np.asarray(temperatures_K, dtype=float)
    rates = np.asarray(rates_per_ps, dtype=float)
    if T.shape != rates.shape or T.ndim != 1:
        raise ValueError("temperatures_K and rates_per_ps must be 1D and same length")
    if np.any(rates <= 0):
        raise ValueError("All rates must be positive.")
    inv_T = 1.0 / T
    log_rate = np.log(rates)
    result = linregress(inv_T, log_rate)
    E_a = float(-result.slope * K_B_EV_PER_K)
    A = float(np.exp(result.intercept))
    stderr_E_a = float(result.stderr * K_B_EV_PER_K)
    return ArrheniusFit(E_a_eV=E_a, A_per_ps=A, stderr_E_a_eV=stderr_E_a)


def residence_fit_comparison(tau_ps):
    """SI S10 fits, with density and survival on the published common grid.

    Reuses both fit functions. The survival from the piecewise density uses
    a finite-grid reverse cumulative sum, normalized to S(0)=1.
    Returns numeric arrays plus fitted scalar parameters for export/plotting.
    """
    tau = np.asarray(tau_ps, dtype=float)
    tau = tau[tau > 0]
    piece = fit_piecewise(tau, rho_floor=2e-4, comparison=True)
    bi = fit_biexp_survival(tau, initial=[0.5, 0.1, 0.01])
    grid = np.linspace(0, max(float(tau.max()), 1.0), 2000)
    pdf = np.exp(_piecewise_log_rho(grid, -piece.lambda_1_per_ps,
                 -piece.lambda_2_per_ps, piece.tau_c_ps, piece.intercept))
    survival = np.cumsum(pdf[::-1])[::-1]
    survival = np.clip(survival / survival[0], 1e-12, 1)
    bi_s = _biexp_survival(grid, bi.amplitude, bi.lambda_1_per_ps, bi.lambda_2_per_ps)
    bi_pdf = (bi.amplitude * bi.lambda_1_per_ps * np.exp(-bi.lambda_1_per_ps * grid)
              + (1-bi.amplitude) * bi.lambda_2_per_ps * np.exp(-bi.lambda_2_per_ps * grid))
    return dict(tau_ps=tau, grid_ps=grid, piecewise_pdf=pdf,
                piecewise_survival=survival, biexp_pdf=bi_pdf, biexp_survival=bi_s,
                piecewise_parameters=np.array([piece.lambda_1_per_ps,
                  piece.lambda_2_per_ps, piece.tau_c_ps, piece.intercept]),
                biexp_parameters=np.array([bi.amplitude, bi.lambda_1_per_ps, bi.lambda_2_per_ps]))


def rate_summary(events, *, overall_model):
    """Paper fits plus direct counts, with explicit insufficient-fit statuses.

    ``overall_model`` must be 'single' or 'piecewise'; chemistry is not guessed.
    A fit is a characteristic decay, not a direct-count mean rate. Probability
    uses the published N >= 2, bin count >= 10 selection and a Bernoulli model.
    """
    import pandas as pd
    if overall_model not in {'single','piecewise'}:
        raise ValueError('Choose single or piecewise explicitly')
    samples = collect_taus(events)
    rows = []
    def add(name,value,method,status='complete',detail=''):
        if method in {'fit_per_ps', 'model_per_ps'} and (not np.isfinite(value) or value <= 0):
            status, detail = 'insufficient_data', detail or 'Nonpositive or nonfinite fitted rate'
        rows.append(dict(quantity=name,value=value,method=method,status=status,detail=detail))
    for name,value in global_average_rate(events).items(): add(name,value,'direct_count_per_ps')
    productive = probability = None
    try:
        productive = fit_single_exp(samples.tau_PP_ps)
        add('productive',productive.lambda_per_ps,'fit_per_ps')
    except ValueError as exc: add('productive',np.nan,'fit_per_ps','insufficient_data',str(exc))
    try:
        if overall_model == 'piecewise':
            overall = fit_piecewise(samples.tau_OO_ps)
            add('overall_fast',overall.lambda_1_per_ps,'fit_per_ps')
            add('overall_slow',overall.lambda_2_per_ps,'fit_per_ps')
            add('transition',overall.tau_c_ps,'fit_ps')
        else:
            overall = fit_single_exp(samples.tau_OO_ps,rho_floor=DEFAULT_RHO_FLOOR_OO,trim_to_peak=True)
            add('overall',overall.lambda_per_ps,'fit_per_ps')
    except (ValueError, RuntimeError) as exc:
        add('overall',np.nan,'fit_per_ps','insufficient_data',str(exc))
    try:
        probability = productive_probability(events)
        add('productive_probability',probability.p,'geometric_fit')
        if productive is not None:
            add('nonproductive',nonproductive_rate(productive.lambda_per_ps,probability.p),'model_per_ps')
    except ValueError as exc:
        add('productive_probability',np.nan,'geometric_fit','insufficient_data',str(exc))
    return pd.DataFrame(rows)


def nonproductive_counts(events: RotationEvents) -> np.ndarray:
    """Count nonproductive switches between consecutive productive switches.

    Boundary-incomplete intervals are excluded; return raw counts even when
    their distribution cannot support the paper's geometric fit.
    """
    counts_per_n: list[int] = []
    for h in range(events.n_protons):
        o_sw = np.asarray(events.o_pair_switch_frames[h], dtype=int)
        p_sw = np.asarray(events.p_pair_switch_frames[h], dtype=int)
        if o_sw.size == 0 or p_sw.size == 0:
            continue
        if p_sw.size < 2:
            continue
        nonprod = np.setdiff1d(o_sw, p_sw)
        # O and X pair histories are segmented separately. Do not assume
        # every X-pair switch occurs in the O-pair switch array.
        gaps = (np.searchsorted(nonprod, p_sw[1:], side='left')
                - np.searchsorted(nonprod, p_sw[:-1], side='right'))
        counts_per_n.extend(gaps.tolist())

    return np.asarray(counts_per_n, dtype=int)
