"""Mean-square displacement and tracer diffusivity.

Reproduces the manuscript's transport figure: per-species MSD versus lag time
at one temperature, the tracer diffusion coefficient ``D`` extracted from the
diffusive regime of the proton MSD, and the Arrhenius activation energy across
temperatures.

The linear-regime search is credited to N. Molinari.

The published MSD curves were produced by an external Fortran tool.
:func:`species_msd` recomputes the same quantity directly from a LAMMPS dump
in Python. :func:`window_averaged_msd` computes the time-window average by
FFT in O(n log n), making the estimator practical for the 80k-frame production
trajectories. This is equivalent to an explicit O(n^2) average over each lag;
see its docstring for the identity.

Both functions require **unwrapped** coordinates (``xu yu zu`` in the dump
header), which is what the production runs wrote. Wrapped coordinates make the
MSD saturate at the box size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import linregress
from tqdm import tqdm

if TYPE_CHECKING:
    from ovito.pipeline import Pipeline

# A^2/ps -> cm^2/s
A2_PER_PS_TO_CM2_PER_S = (1e-8) ** 2 / 1e-12


@dataclass(frozen=True)
class SpeciesMSD:
    """Per-species mean-square displacement versus lag time."""

    time_ps: np.ndarray  # (n_lags,) lag time, excluding zero lag
    msd_A2: dict[str, np.ndarray]  # element symbol -> (n_lags,) MSD in A^2
    n_atoms: dict[str, int]  # element symbol -> atom count entering the average

    def as_columns(self, order: list[str] | None = None) -> dict[str, np.ndarray]:
        """Flatten to the reference-CSV column layout (``time_ps``, ``MSD_<el>_A2``)."""
        elements = order if order is not None else list(self.msd_A2)
        cols: dict[str, np.ndarray] = {"time_ps": self.time_ps}
        for el in elements:
            cols[f"MSD_{el}_A2"] = self.msd_A2[el]
        return cols


@dataclass(frozen=True)
class DiffusivityFit:
    """Tracer diffusivity from a linear fit to the diffusive regime of an MSD."""

    D_cm2_s: float
    D_err_cm2_s: float
    slope_A2_per_ps: float
    intercept_A2: float
    fit_start_ps: float
    window_ps: float
    r2: float
    loglog_slope: float
    """Slope of log(MSD) vs log(t) over the fitted window.

    Fickian diffusion gives 1. A value well above 1 means the window is still
    ballistic or sub-diffusive and ``D_cm2_s`` is not a diffusion coefficient:
    the 75 ps example trajectories give ~1.5 and a ``D`` four times the
    published value. Check this before quoting ``D``.
    """

    @property
    def is_diffusive(self) -> bool:
        """Whether the fitted window looks Fickian (log-log slope within 0.15 of 1)."""
        return abs(self.loglog_slope - 1.0) <= 0.15


def window_averaged_msd(
    pos: np.ndarray, *, frac: float = 0.7, atom_chunk: int = 64
) -> np.ndarray:
    """Time-window-averaged MSD of a set of tracers, computed by FFT.

    For unwrapped positions ``r_i(t)`` the estimator is

    .. code-block:: text

        MSD(D) = < |r_i(t + D) - r_i(t)|^2 >_{i, t}

    averaged over every atom ``i`` and every origin ``t`` with ``t + D`` inside
    the trajectory. Expanding the square splits it into a term that needs only
    the squared norms and a term that is an autocorrelation:

    .. code-block:: text

        MSD(D) = S1(D) - 2 * S2(D)
        S1(D)  = mean over valid t of [ |r(t)|^2 + |r(t+D)|^2 ]
        S2(D)  = mean over valid t of [ r(t) . r(t+D) ]

    ``S1`` follows from cumulative sums of ``|r|^2`` and ``S2`` is the
    autocorrelation of each coordinate, evaluated with a zero-padded FFT. The
    result is identical to an explicit average over each lag up to
    floating-point round-off, at O(n log n) instead of O(n^2).

    Parameters
    ----------
    pos
        ``(n_frames, n_atoms, 3)`` unwrapped positions.
    frac
        Report lags out to ``frac * n_frames``. The default is 0.7; the longest
        lags are averaged over few origins and are noisy. Pass ``frac=1.0`` to
        reproduce the published MSD CSVs, which report every lag.
    atom_chunk
        Number of atoms whose per-lag arrays are held in memory at once. The
        atom average is accumulated across chunks, so this trades speed for
        peak memory without changing the result.

    Returns
    -------
    np.ndarray
        ``(n_lags,)`` MSD in the units of ``pos`` squared, starting at lag 1
        (zero lag is trivially zero and is dropped).
    """
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"pos must be (n_frames, n_atoms, 3), got {pos.shape}")
    n_frames = pos.shape[0]
    n_lags = int(frac * n_frames)
    if n_lags < 2:
        raise ValueError(
            f"Trajectory too short: {n_frames} frames with frac={frac} gives "
            f"{n_lags} lags."
        )

    n_atoms = pos.shape[1]
    lags = np.arange(n_lags)
    counts = (n_frames - lags).astype(np.float64)  # valid origins per lag

    n_fft = 1
    while n_fft < 2 * n_frames:
        n_fft *= 2

    msd_sum = np.zeros(n_lags, dtype=np.float64)
    for lo in range(0, n_atoms, max(atom_chunk, 1)):
        block = pos[:, lo : lo + max(atom_chunk, 1), :]  # (n_frames, k, 3)
        sq = (block**2).sum(axis=2)  # (n_frames, k)

        # S1[D] = mean over valid t of |r(t)|^2 + |r(t+D)|^2. Both sums are a
        # head and a tail of the same cumulative sum over frames.
        csum = np.cumsum(sq, axis=0)
        total = csum[-1]
        head = csum[n_frames - 1 - lags]  # sum_{t=0}^{n-D-1}
        tail = total[None, :] - np.vstack(
            [np.zeros((1, sq.shape[1])), csum[: n_lags - 1]]
        )  # sum_{t=D}^{n-1}
        S1 = (head + tail) / counts[:, None]

        # S2[D] = mean over valid t of r(t) . r(t+D), by zero-padded FFT.
        S2 = np.zeros((n_lags, sq.shape[1]), dtype=np.float64)
        for dim in range(3):
            f = np.fft.rfft(block[:, :, dim], n=n_fft, axis=0)
            S2 += np.fft.irfft(f * np.conjugate(f), n=n_fft, axis=0)[:n_lags]
        S2 /= counts[:, None]

        msd_sum += (S1 - 2.0 * S2).sum(axis=1)

    return (msd_sum / n_atoms)[1:]


def species_msd(
    pipeline: Pipeline,
    *,
    material: str,
    start_frame: int = 0,
    end_frame: int | None = None,
    stride: int = 1,
    frame_to_ps: float = 0.05,
    frac: float = 0.7,
    show_progress: bool = True,
) -> SpeciesMSD:
    """Per-species MSD versus lag time, straight from a LAMMPS dump.

    Reads every frame once, then window-averages each species separately.
    Element labels come from :func:`superproton.io.species_map`, so the result
    is keyed by symbol (``"Cs"``, ``"H"``, ``"P"`` or ``"S"``, ``"O"``) rather
    than by the dump's integer type ids, whose ordering is not fixed across
    materials.

    Memory is ``8 * n_frames * n_atoms * 3`` bytes for the position block: a
    4 ns CDP run at 50 fs (80k frames, 1000 atoms) needs about 1.9 GB. Pass
    ``stride`` to thin, or ``end_frame`` to analyse a window.
    """
    from superproton.io import check_stable_particle_order, species_map

    species = species_map(pipeline, material, frame=start_frame)

    n_total = int(pipeline.source.num_frames)
    last = n_total if end_frame is None else min(end_frame, n_total)
    frame_indices = np.arange(start_frame, last, stride, dtype=int)
    if frame_indices.size == 0:
        raise ValueError(f"Empty frame range [{start_frame}, {last}) stride {stride}.")
    # Per-element indices below are resolved once and reused for every frame.
    check_stable_particle_order(pipeline, frame_indices)

    data0 = pipeline.compute(int(frame_indices[0]))
    types0 = np.array(data0.particles.particle_types).astype(int)
    idx_by_element = {
        el: np.where(types0 == tid)[0] for tid, el in species.items()
    }

    pos = np.zeros((frame_indices.size, types0.size, 3), dtype=np.float64)
    for t, fr in enumerate(
        tqdm(frame_indices, disable=not show_progress, desc="msd frames")
    ):
        pos[t] = np.array(pipeline.compute(int(fr)).particles.positions)

    dt = frame_to_ps * stride
    msd_by_element: dict[str, np.ndarray] = {}
    for el, idxs in idx_by_element.items():
        msd_by_element[el] = window_averaged_msd(pos[:, idxs, :], frac=frac)
    n_lags = len(next(iter(msd_by_element.values())))
    time_ps = np.arange(1, n_lags + 1, dtype=float) * dt

    return SpeciesMSD(
        time_ps=time_ps,
        msd_A2=msd_by_element,
        n_atoms={el: int(idxs.size) for el, idxs in idx_by_element.items()},
    )


def _as_fit(fit, *, dim: int, loglog_slope: float, fit_start_ps: float,
            window_ps: float) -> DiffusivityFit:
    """Wrap a ``linregress`` of MSD against time as a :class:`DiffusivityFit`."""
    return DiffusivityFit(
        D_cm2_s=float(fit.slope / (2 * dim) * A2_PER_PS_TO_CM2_PER_S),
        D_err_cm2_s=float(fit.stderr / (2 * dim) * A2_PER_PS_TO_CM2_PER_S),
        slope_A2_per_ps=float(fit.slope),
        intercept_A2=float(fit.intercept),
        fit_start_ps=fit_start_ps,
        window_ps=window_ps,
        r2=float(fit.rvalue**2),
        loglog_slope=loglog_slope,
    )


def _linear_regime_score(slope: float, rvalue: float, window_frac: float) -> float:
    """Score a candidate diffusive window in log-log space.

    Rewards a log-log slope near 1 (Fickian diffusion), a near-perfectly linear
    regression, and a longer window, in that order of weight.
    """
    return -abs(1.0 - slope) - 1e2 * abs(1.0 - rvalue) + 0.05 * window_frac


def diffusivity(
    time_ps: np.ndarray,
    msd_A2: np.ndarray,
    *,
    skip_ps: float = 0.0,
    window_ps: float = 1000.0,
    search_step: int = 2000,
    dim: int = 3,
    fit_range_ps: tuple[float, float] | None = None,
) -> DiffusivityFit:
    """Tracer diffusivity from the diffusive regime of an MSD curve.

    Scans candidate fit windows anchored at the end of the curve, scoring each
    by :func:`_linear_regime_score` in log-log space, then fits the winning
    window in linear space and converts the slope:

    .. code-block:: text

        D = (dMSD/dt) / (2 * dim)

    Accuracy
    --------
    ``D`` is sensitive to the fit window at the few-percent level,
    which is larger than the standard error of any single fit. On the CsH2PO4
    525 K production trajectory, fitting the proton MSD over fixed lag windows
    gives 1.83e-7 cm^2/s (100-1000 ps), 1.86e-7 (500-2000 ps) and 1.96e-7
    (10-500 ps); the automatic scan lands on 1.90e-7, and the published value
    is 1.848e-7. The spread is a property of the estimator, not a defect: the
    same numbers come out of the published MSD curve to within 0.1%. Quote
    ``D`` to two significant figures, keep the window fixed when comparing
    across temperatures, and prefer ``fit_range_ps`` when you need an exactly
    reproducible window.

    Parameters
    ----------
    time_ps, msd_A2
        Lag time in ps and MSD in A^2, as returned by :func:`species_msd`.
    skip_ps
        Drop this much of the short-time (ballistic / vibrational) head first.
    window_ps
        Reference window length used to normalise the window-length reward.
    search_step
        Stride in frames between candidate window starts.
    dim
        Spatial dimensionality of the diffusion. 3 for these bulk phases.
    fit_range_ps
        ``(lo, hi)`` lag window in ps. When given, the automatic scan is
        skipped and the fit is a plain linear regression over that window,
        which makes the result exactly reproducible across callers.
    """
    time_ps = np.asarray(time_ps, dtype=float)
    msd_A2 = np.asarray(msd_A2, dtype=float)
    if time_ps.shape != msd_A2.shape:
        raise ValueError(
            f"time_ps {time_ps.shape} and msd_A2 {msd_A2.shape} must have the same shape"
        )

    dt = float(time_ps[1] - time_ps[0]) if time_ps.size > 1 else float(time_ps[0])
    skip_idx = int(round(skip_ps / dt))
    t = time_ps[skip_idx:]
    y = msd_A2[skip_idx:]
    positive = (t > 0) & (y > 0)
    t, y = t[positive], y[positive]
    if t.size < 8:
        raise ValueError(f"Only {t.size} usable MSD points after skip_ps={skip_ps}.")

    if fit_range_ps is not None:
        lo, hi = float(fit_range_ps[0]), float(fit_range_ps[1])
        if not hi > lo:
            raise ValueError(f"fit_range_ps must be increasing, got {fit_range_ps}.")
        window = (t >= lo) & (t <= hi)
        if window.sum() < 3:
            raise ValueError(
                f"fit_range_ps={fit_range_ps} selects only {int(window.sum())} MSD "
                f"points (lags span {t[0]:.3g}-{t[-1]:.3g} ps)."
            )
        fit = linregress(t[window], y[window])
        loglog = linregress(np.log10(t[window]), np.log10(y[window]))
        return _as_fit(
            fit, dim=dim, loglog_slope=float(loglog.slope),
            fit_start_ps=float(t[window][0]),
            window_ps=float(t[window][-1] - t[window][0]),
        )

    log_t, log_y = np.log10(t), np.log10(y)
    window_frames = max(int(round(window_ps / dt)), 1)
    # The scan starts one quarter of a reference window in from the end. On a
    # trajectory shorter than that quarter there would be no candidate at all,
    # so clamp the offset to leave a usable window: the shipped 75 ps example
    # dumps are far shorter than the 1000 ps default reference window.
    initial = max(int(0.25 * window_frames), 1)
    min_points = 4
    if initial > t.size - min_points:
        initial = max(t.size - min_points, min(min_points, t.size - 1))

    best_score = -np.inf
    best: DiffusivityFit | None = None
    for i in range(t.size - initial, 0, -max(search_step, 1)):
        cand = linregress(log_t[i:], log_y[i:])
        window_frac = (t.size - i) / window_frames
        score = _linear_regime_score(cand.slope, cand.rvalue, window_frac)
        if score > best_score:
            fit = linregress(t[i:], y[i:])
            best_score = score
            best = _as_fit(
                fit, dim=dim, loglog_slope=float(cand.slope),
                fit_start_ps=float(t[i]),
                window_ps=float(window_frac * window_ps),
            )
    if best is None:
        raise ValueError("No candidate diffusive window found; check skip_ps/window_ps.")
    return best
