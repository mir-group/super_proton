"""Bound-proton coordination ``N_p(t)`` and its joint density with ``tau_O-O``.

Reproduces the bound-proton analysis in Figures 4d and 7b: for each completed
O-pair residence interval, identify the polyanion that hosts the tracked proton
less frequently and average its covalently bonded proton population over the
whole interval. The same polyanion remains selected during proton rattling.

The manuscript counts mutual nearest-neighbor O-H bonds. Use
``mutual_bound_protons`` for that calculation; ``bound_protons_per_frame``
is a donor-count diagnostic that also counts shared-oxygen protons.
The original Fig. 4d/7b routine fixes the less frequently occupied polyanion
within each interval, rather than changing partners on each rattling step.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from superproton.rotation import RotationEvents
from superproton.tracking import ProtonTracks


@dataclass(frozen=True)
class CoordinationData:
    """Per-frame bound-proton counts on every polyanion."""

    n_frames: int
    n_polyanions: int
    polyanion_ids: np.ndarray  # (n_polyanions,) of LAMMPS X identifiers, sorted
    bound_per_frame: np.ndarray  # (n_frames, n_polyanions) int

    def index_of(self, polyanion_id: int) -> int:
        """Return the column index in ``bound_per_frame`` for a given X identifier."""
        i = int(np.searchsorted(self.polyanion_ids, polyanion_id))
        if i >= self.polyanion_ids.size or int(self.polyanion_ids[i]) != int(polyanion_id):
            raise KeyError(f"polyanion id {polyanion_id} not in CoordinationData")
        return i


def bound_protons_per_frame(tracks: ProtonTracks) -> CoordinationData:
    """Count, at each frame, how many protons have each polyanion as their donor X.

    Uses ``np.add.at`` to accumulate counts vectorized per frame, avoiding
    the (n_frames * n_protons) Python loop.
    """
    n_frames, _ = tracks.donor_X_id.shape
    poly_ids = np.unique(tracks.donor_X_id)
    n_poly = poly_ids.size

    # Map polyanion ids to dense 0..n_poly-1 column indices.
    id_to_col = np.full(int(poly_ids.max()) + 1, -1, dtype=np.int64)
    id_to_col[poly_ids] = np.arange(n_poly, dtype=np.int64)

    cols = id_to_col[tracks.donor_X_id]  # (n_frames, n_protons)
    counts = np.zeros((n_frames, n_poly), dtype=np.int64)
    for t in range(n_frames):
        np.add.at(counts[t], cols[t], 1)
    return CoordinationData(
        n_frames=n_frames,
        n_polyanions=n_poly,
        polyanion_ids=poly_ids,
        bound_per_frame=counts,
    )


def _interval_endpoints(
    sw_frames: np.ndarray, n_frames: int
) -> np.ndarray:
    """Return (n_intervals, 2) array of half-open [start, end) frame ranges.

    Intervals between consecutive switches only. Opening and closing intervals
    that touch the trajectory boundary are dropped, matching
    :func:`superproton.residence.collect_taus`.
    """
    if sw_frames.size < 2:
        return np.empty((0, 2), dtype=np.int64)
    starts = sw_frames[:-1]
    ends = sw_frames[1:]
    return np.column_stack([starts, ends])


def mean_Np_per_residence(
    tracks: ProtonTracks,
    events: RotationEvents,
    bound: CoordinationData,
    *, neighbor: str = "least_occupied",
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned (tau_OO_ps, mean_Np) arrays for completed intervals.

    ``least_occupied`` reproduces proton_dyn_test.py: select one of the
    proton's visited host polyanions by the lowest occupancy count over the
    whole interval (ties choose the first-encountered host), then average its
    bound-proton population. ``instantaneous`` retains the previous API's
    per-frame opposite-partner convention as an explicit alternative.
    """
    if neighbor == "least_occupied":
        values = [residence_coordination(events.o_pair_switch_frames[h],
                  tracks.donor_X_id[:, h], bound, frame_to_ps=events.frame_to_ps)
                  for h in range(events.n_protons)]
        if not values:
            return np.empty(0), np.empty(0)
        return tuple(np.concatenate([v[i] for v in values]) for i in range(2))
    if neighbor != "instantaneous":
        raise ValueError("neighbor must be least_occupied or instantaneous")
    dt = float(events.frame_to_ps)

    tau_chunks: list[np.ndarray] = []
    np_chunks: list[np.ndarray] = []

    parent_X = _per_O_parent_X(tracks)

    for h in range(events.n_protons):
        o_sw = events.o_pair_switch_frames[h]
        if o_sw.size < 2:
            continue
        intervals = _interval_endpoints(o_sw, events.n_frames)

        # o_pair_ids[h] has shape (n_intervals + 1, 2). The first entry is the
        # pair active BEFORE the first switch (skipped), the entry at index k
        # is the pair active over interval k-1 (between switches k-1 and k).
        pairs = events.o_pair_ids[h]
        if pairs.shape[0] < intervals.shape[0] + 1:
            # Skip protons whose pair-id list is incomplete (defensive).
            continue

        tau_h = np.empty(intervals.shape[0], dtype=float)
        np_h = np.empty(intervals.shape[0], dtype=float)
        mask = np.ones(intervals.shape[0], dtype=bool)

        for k, (t_start, t_end) in enumerate(intervals):
            o_a, o_b = pairs[k + 1]  # pair active during interval k
            if o_a == -1 or o_b == -1:
                mask[k] = False
                continue
            x_a = parent_X.get(int(o_a))
            x_b = parent_X.get(int(o_b))
            if x_a is None or x_b is None or x_a == x_b:
                mask[k] = False
                continue
            donor_segment = tracks.donor_O_id[t_start:t_end, h]
            # neighbor X[t] = x_b when donor is o_a, x_a when donor is o_b.
            neighbor_X = np.where(donor_segment == o_a, x_b, x_a)
            try:
                cols = np.array([bound.index_of(int(x)) for x in neighbor_X])
            except KeyError:
                mask[k] = False
                continue
            frame_range = np.arange(t_start, t_end)
            Np_t = bound.bound_per_frame[frame_range, cols]
            np_h[k] = float(Np_t.mean())
            tau_h[k] = float((t_end - t_start) * dt)

        if mask.any():
            tau_chunks.append(tau_h[mask])
            np_chunks.append(np_h[mask])

    if not tau_chunks:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    return np.concatenate(tau_chunks), np.concatenate(np_chunks)


def joint_density_Np_tau(
    tau_ps: np.ndarray,
    mean_Np: np.ndarray,
    *,
    tau_bins: int = 60,
    np_bins: int | np.ndarray = 30,
    tau_range: tuple[float, float] | None = None,
    np_range: tuple[float, float] = (0.5, 3.5),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D joint probability density ``rho(bar{N_p}, tau_O-O)`` for Figure 4d.

    Returns ``(density, tau_edges, np_edges)`` with ``density`` shape
    ``(tau_bins, np_bins)`` normalized so that the integral is one.
    """
    if tau_range is None:
        tau_range = (float(tau_ps.min()), float(tau_ps.max()))
    H, tau_edges, np_edges = np.histogram2d(
        tau_ps,
        mean_Np,
        bins=[tau_bins, np_bins],
        range=[tau_range, np_range],
        density=True,
    )
    return H, tau_edges, np_edges


def _per_O_parent_X(tracks: ProtonTracks) -> dict[int, int]:
    """Build ``{O_id: parent_X_id}`` from the donor arrays in tracks.

    We invert the frame-0 ``donor_O_id -> donor_X_id`` mapping. Because each
    O has a fixed parent X across the trajectory, any frame in which a given
    O is observed as a donor reveals its parent X. We take the first
    occurrence per O to keep the dict deterministic.
    """
    parent: dict[int, int] = {}
    flat_O = tracks.donor_O_id.ravel()
    flat_X = tracks.donor_X_id.ravel()
    for o, x in zip(flat_O.tolist(), flat_X.tolist(), strict=True):
        parent.setdefault(int(o), int(x))
    return parent


def residence_coordination(switch_frames, host_ids, bound: CoordinationData, *,
                            frame_to_ps=0.05):
    """Fixed-neighbor coordination for one proton, shared by raw/cache inputs.

    Frames are local, intervals [start,end); incomplete trajectory boundaries
    are excluded. Host IDs and bound.polyanion_ids must use the same labels.
    The legacy cache uses per-proton labels 1..N; raw trajectories use atom IDs.
    """
    tau, means = [], []
    unique, first = np.unique(host_ids, return_index=True)
    first_visit = dict(zip(unique, first))
    for start, end in zip(switch_frames[:-1], switch_frames[1:]):
        if not 0 <= start < end <= len(host_ids):
            raise ValueError("Invalid residence interval")
        ids, counts = np.unique(host_ids[start:end], return_counts=True)
        tied = ids[counts == counts.min()]
        neighbor = min(tied, key=first_visit.__getitem__)
        column = bound.index_of(int(neighbor))
        tau.append((end - start) * frame_to_ps)
        means.append(bound.bound_per_frame[start:end, column].mean())
    return np.asarray(tau), np.asarray(means)


def mutual_bound_protons(pipeline, material, frame_indices) -> CoordinationData:
    """Count strictly mutual nearest-neighbor O-H bonds, with periodic images.

    Like the original get_nearby_Hs, the nearest neighbor is taken among all
    atom species. An O counts only if its nearest atom is H and that H's
    nearest atom is the same O. No arbitrary distance cutoff is introduced.
    """
    from superproton.io import species_map
    from superproton.tracking import parent_polyanions

    frames = np.asarray(frame_indices, dtype=int)
    if not frames.size:
        raise ValueError("No frames supplied")
    parents = parent_polyanions(pipeline, material, frame=int(frames[0]))
    species = {v: k for k, v in species_map(pipeline, material).items()}
    x_ids = np.unique(list(parents.values()))
    counts = np.zeros((len(frames), len(x_ids)), dtype=int)
    for row, frame in enumerate(frames):
        data = pipeline.compute(int(frame))
        counts[row] = _mutual_bound_counts(
            data, parents, x_ids, hydrogen_type=species["H"], oxygen_type=species["O"]
        )
    return CoordinationData(len(frames), len(x_ids), x_ids, counts)


def _mutual_bound_counts(data, parents, polyanion_ids, *, hydrogen_type, oxygen_type):
    """Strict mutual bond counts from an already loaded frame (shared reader)."""
    from ovito.data import NearestNeighborFinder

    ids = np.asarray(data.particles.identifiers)
    types = np.asarray(data.particles.particle_types)
    oxygen = np.flatnonzero(types == oxygen_type)
    nearest, _ = NearestNeighborFinder(1, data).find_all()
    nearest = nearest[:, 0]
    mutual = (types[nearest[oxygen]] == hydrogen_type) & (nearest[nearest[oxygen]] == oxygen)
    hosts = [parents[int(o)] for o in ids[oxygen[mutual]]]
    return np.bincount(np.searchsorted(polyanion_ids, hosts), minlength=len(polyanion_ids))
