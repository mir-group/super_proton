"""Temporal overlap and spatial-temporal correlation of rotation events.

``temporal_overlap`` computes the NP-centred statistics of SI Tables S2-S3.
It counts intersecting rotation intervals on the same polyanion separately
for each NP role (exchanger, host) and P role (carrier, recipient, provider).
Overlap includes shared endpoints and complete containment:
``NP_start <= P_end and P_start <= NP_end``. Roles are never pooled.

Spatial-temporal event-pair correlation G_event(r, dt)
---------------------------------------------------

``g_event`` reproduces the analysis from SI Section "Spatial-temporal
correlation of productive rotations" (Figures S11-S12): for each pair of
productive rotation events ``(i, j)``, bin by polyanion-center separation
``r`` and time lag ``dt = t_j - t_i``, then normalize by a Poisson baseline
that assumes the same average event rate and shell density.

    G_event(r, dt) = H_obs(r, dt) / H_exp(r, dt)
    H_exp(r, dt)   = N_ev * nu_bar * dt_bin * n_shell(r) / N_polyanion

``G_event >> 1`` at short ``r`` and ``dt`` indicates a cascade; ``G_event ~ 1``
indicates statistical independence. The manuscript uses the recipient
polyanion as the spatial anchor of each event (the polyanion that just
gained a proton).

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from superproton.hopping import _roles_for_switches
from superproton.io import pairwise_min_image_distances
from superproton.rotation import RotationEvents

if TYPE_CHECKING:
    from ovito.pipeline import Pipeline


@dataclass(frozen=True)
class PolyanionCenters:
    """Polyanion (P or S) center positions and the simulation cell."""

    positions: np.ndarray  # (n_polyanions, 3) cartesian coordinates
    cell: np.ndarray  # (3, 3) row-vector lattice matrix
    polyanion_ids: np.ndarray  # (n_polyanions,) LAMMPS identifiers, sorted


@dataclass(frozen=True)
class GEventResult:
    """Output of :func:`g_event`."""

    shells_A: np.ndarray  # (n_shells,) mean polyanion-pair distance in each shell
    dt_centers_ps: np.ndarray  # (n_dt,) time-bin centers
    dt_edges_ps: np.ndarray  # (n_dt + 1,) time-bin edges
    G: np.ndarray  # (n_shells, n_dt) -- nan where H_exp underflows
    H_obs: np.ndarray  # (n_shells, n_dt)
    H_exp: np.ndarray  # (n_shells, n_dt)
    rate_per_ps: float


def polyanion_centers(
    pipeline: Pipeline, material: str, *, frame: int = 0
) -> PolyanionCenters:
    """Extract polyanion-center positions and cell from one trajectory frame.

    Uses ovito's pipeline.compute at ``frame``. The center element is P for
    CDP and S for CHS, determined via :func:`superproton.io.species_map`.
    """
    from superproton.io import polyanion_element, species_map

    species = species_map(pipeline, material, frame=frame)
    inv = {v: k for k, v in species.items()}
    center_type = inv[polyanion_element(material)]

    data = pipeline.compute(frame)
    types = np.array(data.particles.particle_types).astype(int)
    ids = np.array(data.particles.identifiers).astype(int)
    pos = np.array(data.particles.positions).astype(float)
    mask = types == center_type
    poly_pos = pos[mask]
    poly_ids = ids[mask]
    # Sort by id for deterministic ordering.
    order = np.argsort(poly_ids)
    poly_pos = poly_pos[order]
    poly_ids = poly_ids[order]
    cell_matrix = np.array(data.cell.matrix)[:, :3].T.copy()
    return PolyanionCenters(positions=poly_pos, cell=cell_matrix, polyanion_ids=poly_ids)


def pp_distance_matrix(centers: PolyanionCenters) -> np.ndarray:
    """Minimum-image distance matrix between every pair of polyanion centers."""
    return pairwise_min_image_distances(centers.positions, centers.cell)


def identify_shells(d_matrix: np.ndarray, *, tol_A: float = 0.5) -> np.ndarray:
    """Cluster the upper-triangular distance entries into discrete shells.

    A new shell begins when the next distance exceeds the first distance in
    the current shell by more than ``tol_A``. Returns the per-shell mean
    distance (Angstroms), sorted ascending.
    """
    upper = d_matrix[np.triu_indices_from(d_matrix, k=1)]
    if upper.size == 0:
        return np.zeros(0, dtype=float)
    upper = np.sort(upper)
    shells: list[float] = []
    current = [upper[0]]
    for d in upper[1:]:
        if d - current[0] > tol_A:
            shells.append(float(np.mean(current)))
            current = [d]
        else:
            current.append(d)
    shells.append(float(np.mean(current)))
    return np.asarray(shells, dtype=float)


def assign_shell_indices(
    d_matrix: np.ndarray, shells: np.ndarray, *, r_max_A: float
) -> np.ndarray:
    """Map every off-diagonal P-P distance to its nearest shell index.

    Self pairs and distances above ``r_max_A`` map to ``-1``.
    """
    n = d_matrix.shape[0]
    if shells.size == 0:
        return np.full((n, n), -1, dtype=np.int64)
    shell_idx = np.full((n, n), -1, dtype=np.int64)
    # Vectorize the nearest-shell lookup via outer subtraction.
    mask = (d_matrix <= r_max_A) & (d_matrix > 0)
    # For each masked cell, pick argmin of |d - shells|.
    if mask.any():
        diffs = np.abs(d_matrix[..., None] - shells[None, None, :])
        nearest = np.argmin(diffs, axis=-1)
        shell_idx[mask] = nearest[mask]
    return shell_idx


def collect_recipient_events(
    events: RotationEvents,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sorted ``(t_ps, recipient_polyanion_id)`` arrays from RotationEvents.

    For each productive rotation, the recipient polyanion is the one in the
    new P-pair that was not in the previous P-pair. Switches where the new
    pair has no member in common with the previous pair are skipped.
    """
    dt = float(events.frame_to_ps)
    times_ps: list[float] = []
    recipient_ids: list[int] = []

    for h in range(events.n_protons):
        p_sw = events.p_pair_switch_frames[h]
        pairs = events.p_pair_ids[h]
        if p_sw.size == 0 or pairs.shape[0] < 2:
            continue
        for k, sw in enumerate(p_sw.tolist()):
            prev_pair = pairs[k]
            new_pair = pairs[k + 1]
            common = np.intersect1d(prev_pair, new_pair)
            if common.size == 0:
                continue
            recipient_candidates = np.setdiff1d(new_pair, prev_pair)
            recipient_candidates = recipient_candidates[recipient_candidates != -1]
            if recipient_candidates.size == 0:
                continue
            recipient_ids.append(int(recipient_candidates[0]))
            times_ps.append(float(sw) * dt)

    if not times_ps:
        return np.empty(0, dtype=float), np.empty(0, dtype=np.int64)

    times_ps_arr = np.array(times_ps, dtype=float)
    recipient_ids_arr = np.array(recipient_ids, dtype=np.int64)
    order = np.argsort(times_ps_arr)
    return times_ps_arr[order], recipient_ids_arr[order]


def _compute_G_event(
    times_ps: np.ndarray,
    recipient_polyanion_ids: np.ndarray,
    polyanion_ids: np.ndarray,
    shell_idx: np.ndarray,
    shells: np.ndarray,
    *,
    dt_max_ps: float,
    dt_bin_ps: float,
    rate_per_ps: float | None,
    min_H_exp: float,
) -> GEventResult:
    """Core histogram routine; testable without ovito."""
    n_P = polyanion_ids.size
    id_to_idx = {int(p): i for i, p in enumerate(polyanion_ids)}
    site_idx = np.array(
        [id_to_idx[int(p)] for p in recipient_polyanion_ids], dtype=np.int64
    )
    N_ev = times_ps.size
    if N_ev < 2:
        n_shells = shells.size
        dt_edges = np.arange(0.0, dt_max_ps + dt_bin_ps, dt_bin_ps)
        n_dt = len(dt_edges) - 1
        return GEventResult(
            shells_A=shells,
            dt_centers_ps=0.5 * (dt_edges[:-1] + dt_edges[1:]),
            dt_edges_ps=dt_edges,
            G=np.full((n_shells, n_dt), np.nan),
            H_obs=np.zeros((n_shells, n_dt)),
            H_exp=np.zeros((n_shells, n_dt)),
            rate_per_ps=0.0,
        )

    T_total = float(times_ps[-1] - times_ps[0])
    if rate_per_ps is None:
        rate_per_ps = N_ev / T_total

    dt_edges = np.arange(0.0, dt_max_ps + dt_bin_ps, dt_bin_ps)
    n_dt = len(dt_edges) - 1
    dt_widths = dt_edges[1:] - dt_edges[:-1]
    n_shells = shells.size
    H_obs = np.zeros((n_shells, n_dt), dtype=float)

    for i in range(N_ev - 1):
        dt_all = times_ps[i + 1 :] - times_ps[i]
        in_window = dt_all <= dt_max_ps
        if not in_window.any():
            continue
        dt_sel = dt_all[in_window]
        sites_sel = site_idx[i + 1 : i + 1 + dt_sel.size]
        shells_sel = shell_idx[site_idx[i], sites_sel]
        dt_bin = np.searchsorted(dt_edges, dt_sel, side="right") - 1
        valid = (dt_bin >= 0) & (dt_bin < n_dt) & (shells_sel >= 0)
        for s, b in zip(shells_sel[valid], dt_bin[valid], strict=True):
            H_obs[s, b] += 1

    # Expected counts under Poisson + isotropic-shell assumption.
    n_neighbors = np.zeros(n_shells, dtype=float)
    for s in range(n_shells):
        n_neighbors[s] = float((shell_idx == s).sum()) / n_P
    H_exp = (
        N_ev * rate_per_ps * dt_widths[None, :] * n_neighbors[:, None] / n_P
    )

    G = np.full_like(H_obs, np.nan)
    sufficient = H_exp >= min_H_exp
    G[sufficient] = H_obs[sufficient] / H_exp[sufficient]
    return GEventResult(
        shells_A=shells,
        dt_centers_ps=0.5 * (dt_edges[:-1] + dt_edges[1:]),
        dt_edges_ps=dt_edges,
        G=G,
        H_obs=H_obs,
        H_exp=H_exp,
        rate_per_ps=float(rate_per_ps),
    )


def g_event(
    events: RotationEvents,
    pipeline: Pipeline,
    material: str,
    *,
    r_max_A: float = 10.0,
    dt_max_ps: float = 200.0,
    dt_bin_ps: float = 1.0,
    shell_tol_A: float = 0.5,
    rate_per_ps: float | None = None,
    min_H_exp: float = 0.0,
) -> GEventResult:
    """Top-level convenience: extract events + centers and call the core routine."""
    centers = polyanion_centers(pipeline, material)
    d_PP = pp_distance_matrix(centers)
    shells = identify_shells(d_PP, tol_A=shell_tol_A)
    shells = shells[shells <= r_max_A]
    shell_idx = assign_shell_indices(d_PP, shells, r_max_A=r_max_A)
    times_ps, recipient_ids = collect_recipient_events(events)
    return _compute_G_event(
        times_ps=times_ps,
        recipient_polyanion_ids=recipient_ids,
        polyanion_ids=centers.polyanion_ids,
        shell_idx=shell_idx,
        shells=shells,
        dt_max_ps=dt_max_ps,
        dt_bin_ps=dt_bin_ps,
        rate_per_ps=rate_per_ps,
        min_H_exp=min_H_exp,
    )


# NP-centred temporal overlap (SI Tables S2-S3).
NP_ROLES = ("exchanger", "host")
P_ROLES = ("carrier", "recipient", "provider")
OVERLAP_BINS = ("0", "1", "2", ">=3")


@dataclass(frozen=True)
class TemporalOverlap:
    """Six counts per valid NP event, with local frame and proton indices.

    ``counts[(NP_role, P_role)]`` and ``polyanion_ids[NP_role]`` align with
    ``proton_index``. Invalid roles (incomplete pairs or no common member) are
    excluded; skipped event totals are recorded explicitly. A missing oxygen
    in the supplied parent map raises an error instead of dropping the event.
    """

    counts: dict[tuple[str, str], np.ndarray]
    polyanion_ids: dict[str, np.ndarray]
    proton_index: np.ndarray
    t_s_frame: np.ndarray
    t_f_frame: np.ndarray
    n_productive: int
    n_skipped_productive: int
    n_skipped_nonproductive: int

    def summary(self) -> list[dict]:
        """Per-role frequencies for 0, 1, 2, and >=3 overlapping P events.

        Each role combination uses the number of valid NP events as its
        denominator. An empty NP sample has undefined (NaN) percentages.
        """
        rows = []
        n_np = self.proton_index.size
        for (np_role, p_role), counts in self.counts.items():
            bins = np.bincount(np.minimum(counts, 3), minlength=4)
            for label, n in zip(OVERLAP_BINS, bins, strict=True):
                rows.append({
                    "NP_role": np_role, "P_role": p_role, "overlap_count": label,
                    "n_events": int(n), "n_np_events": int(n_np),
                    "percent": float(100 * n / n_np) if n_np else float("nan"),
                })
        return rows


def count_interval_overlaps(
    query_start: np.ndarray,
    query_end: np.ndarray,
    reference_start: np.ndarray,
    reference_end: np.ndarray,
) -> np.ndarray:
    """Count inclusive reference-interval intersections for each query.

    Sorted endpoint searches include both directions of containment without
    allocating the quadratic query-by-reference boolean matrix.
    """
    qs, qe, rs, re = map(np.asarray, (query_start, query_end, reference_start, reference_end))
    for start, end in ((qs, qe), (rs, re)):
        if start.ndim != 1 or end.ndim != 1 or start.shape != end.shape:
            raise ValueError("Interval starts and ends must be matching 1D arrays.")
        if not np.all(np.isfinite(start) & np.isfinite(end) & (start <= end)):
            raise ValueError("Intervals must be finite with start <= end.")
    return (
        np.searchsorted(np.sort(rs), qe, side="right")
        - np.searchsorted(np.sort(re), qs, side="left")
    )


def temporal_overlap(
    events: RotationEvents,
    oxygen_to_polyanion: Mapping[int, int],
) -> TemporalOverlap:
    """Count NP-P temporal overlaps using the manuscript's role definitions.

    For an O-pair switch ``{a,b} -> {b,c}``, the parent of ``b`` is the NP
    host and the parent of ``c`` the exchanger. For a productive X-pair
    switch, the shared, gained, and lost X identifiers are the carrier,
    recipient, and provider. Obtain the O-to-X mapping with
    :func:`superproton.tracking.parent_polyanions`.

    All protons contribute to the productive reference intervals, including
    the proton of the NP event, matching the original analysis. The returned
    frequencies measure temporal coincidence, not statistical independence.
    """
    p_rows, np_rows = [], []
    skipped_p = skipped_np = 0
    for h in range(events.n_protons):
        p_end = np.asarray(events.p_pair_switch_frames[h])
        o_end = np.asarray(events.o_pair_switch_frames[h])
        p_start = np.asarray(events.productive_t_s_frames[h])
        np_start = np.asarray(events.nonproductive_t_s_frames[h])
        np_mask = ~events.productive_mask(h)
        if p_start.shape != p_end.shape or np_start.shape != o_end[np_mask].shape:
            raise ValueError(f"Rotation-period starts do not align with switches for proton {h}.")

        p_roles = np.column_stack(_roles_for_switches(events.p_pair_ids[h]))
        o_roles = np.column_stack(_roles_for_switches(events.o_pair_ids[h]))
        if len(p_roles) != len(p_end) or len(o_roles) != len(o_end):
            raise ValueError(f"Pair histories do not align with switches for proton {h}.")

        for start, end, roles in zip(p_start, p_end, p_roles, strict=True):
            if np.any(roles < 0):
                skipped_p += 1
                continue
            p_rows.append((start, end, *roles))
        for start, end, roles in zip(np_start, o_end[np_mask], o_roles[np_mask], strict=True):
            if np.any(roles < 0):
                skipped_np += 1
                continue
            host, exchanger, _ = roles
            np_rows.append((
                start, end, oxygen_to_polyanion[int(exchanger)],
                oxygen_to_polyanion[int(host)], h,
            ))

    p = np.asarray(p_rows, dtype=np.int64).reshape(-1, 5)
    nonp = np.asarray(np_rows, dtype=np.int64).reshape(-1, 5)
    # Validate all intervals, including ones with no matching polyanion.
    count_interval_overlaps(nonp[:, 0], nonp[:, 1], p[:, 0], p[:, 1])
    counts = {}
    for ni, np_role in enumerate(NP_ROLES, start=2):
        for pi, p_role in enumerate(P_ROLES, start=2):
            result = np.zeros(len(nonp), dtype=np.int64)
            for x_id in np.unique(nonp[:, ni]):
                nmask = nonp[:, ni] == x_id
                pmask = p[:, pi] == x_id
                result[nmask] = count_interval_overlaps(
                    nonp[nmask, 0], nonp[nmask, 1], p[pmask, 0], p[pmask, 1],
                )
            counts[np_role, p_role] = result
    return TemporalOverlap(
        counts=counts,
        polyanion_ids={role: nonp[:, i] for i, role in enumerate(NP_ROLES, start=2)},
        proton_index=nonp[:, 4], t_s_frame=nonp[:, 0], t_f_frame=nonp[:, 1],
        n_productive=len(p), n_skipped_productive=skipped_p,
        n_skipped_nonproductive=skipped_np,
    )
