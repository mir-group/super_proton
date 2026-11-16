"""Productive vs nonproductive polyanion-rotation classification.

``classify_events`` turns :class:`superproton.tracking.ProtonTracks` into the
:class:`RotationEvents` contract consumed by :mod:`superproton.residence`,
:mod:`superproton.coordination` and :mod:`superproton.correlation`.

Definitions match the manuscript (Figure 3 and Section
"Two types of rotational motions"):

- A **productive** rotation transfers a proton across a polyanion pair, i.e.
  the proton's nearest-P (or nearest-S) neighbor changes. The carrier
  polyanion rotates while keeping the covalent O-H bond.
- A **nonproductive** rotation changes only the nearest-O within the same
  polyanion pair. The exchanger rotates without the proton committing to a
  new pair.
- A **rotation period** ``[t_s, t_f]`` is the sub-interval at the end of an
  O-pair residence time ``tau_O-O`` during which proton rattling has stopped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RotationEvents:
    """All rotation events extracted from a trajectory.

    Each ``list[np.ndarray]`` is length ``n_protons``; the inner array holds
    integer frame indices for that proton's events. Frame indices are sorted
    ascending. Frame zero and ``n_frames - 1`` are never included.

    O-pair and polyanion-pair histories are segmented independently. A
    productive switch changes the donor O, but need not reset the O pair
    at that same frame; do not assume the switch arrays are nested.

    Attributes
    ----------
    n_protons
        Number of protons tracked.
    n_frames
        Total number of MD frames in the source trajectory. Used downstream
        to drop residence intervals that touch the trajectory boundaries.
    frame_to_ps
        Time per frame in picoseconds. ``0.05`` ps (50 fs) for the example
        dumps shipped with this repo.
    o_pair_switch_frames
        Per-proton frames at which the unordered donor O pair changed.
    p_pair_switch_frames
        Per-proton frames at which the unordered donor polyanion pair changed
        (productive rotations).
    productive_t_s_frames
        Per-proton rotation-period start frames ``t_s`` for productive
        events, aligned 1-to-1 with ``p_pair_switch_frames[h]``.
    nonproductive_t_s_frames
        Per-proton ``t_s`` frames for nonproductive rotations, aligned 1-to-1
        with the nonproductive entries of ``o_pair_switch_frames[h]``
        (i.e. members that are not in ``p_pair_switch_frames[h]``).
    """

    n_protons: int
    n_frames: int
    frame_to_ps: float
    o_pair_switch_frames: list[np.ndarray]
    p_pair_switch_frames: list[np.ndarray]
    productive_t_s_frames: list[np.ndarray]
    nonproductive_t_s_frames: list[np.ndarray]
    # Per-proton list of (n_intervals + 1, 2) arrays giving the unordered O id
    # pair and X id pair active over each residence interval (and after the
    # final switch). The trailing `-1` in the last entry's second column
    # marks an interval that ended at the trajectory boundary without
    # encountering a second distinct id.
    o_pair_ids: list[np.ndarray]
    p_pair_ids: list[np.ndarray]

    def __post_init__(self) -> None:
        n = self.n_protons
        for name in (
            "o_pair_switch_frames",
            "p_pair_switch_frames",
            "productive_t_s_frames",
            "nonproductive_t_s_frames",
            "o_pair_ids",
            "p_pair_ids",
        ):
            lst = getattr(self, name)
            if len(lst) != n:
                raise ValueError(
                    f"RotationEvents.{name} has length {len(lst)}, expected {n}"
                )

    def productive_mask(self, h: int) -> np.ndarray:
        """Boolean mask over ``o_pair_switch_frames[h]`` marking productive events."""
        o = self.o_pair_switch_frames[h]
        p = self.p_pair_switch_frames[h]
        return np.isin(o, p)


def _unordered_pair_switches(
    seq: np.ndarray,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Detect frames at which the unordered pair of recent identifiers changes.

    Maintain a current unordered pair ``{a, b}`` of the two most recently seen
    distinct identifiers. At each frame:

    - if the new identifier is in ``{a, b}``, no switch (proton rattles within
      the pair),
    - if the new identifier introduces a third id, record a switch at this
      frame and start a fresh pair with the new id alone.

    Parameters
    ----------
    seq
        1D integer sequence of nearest-neighbor identifiers (one entry per
        recorded frame).

    Returns
    -------
    switch_frames
        Local frame indices (0-based positions in ``seq``) where the pair
        changed.
    pair_ids
        Length ``len(switch_frames) + 1`` list of ``(a, b)`` tuples giving the
        unordered pair active before each switch (and after the final one).
        The first entry's ``b`` is ``-1`` if the proton never met a second id
        before the (eventual) trajectory end.
    """
    n = seq.size
    if n == 0:
        return [], []
    a = int(seq[0])
    b: int | None = None
    pair_ids: list[tuple[int, int]] = []
    switch_frames: list[int] = []

    for t in range(1, n):
        x = int(seq[t])
        if b is None:
            if x != a:
                b = x
                pair_ids.append((min(a, b), max(a, b)))
        else:
            if x != a and x != b:
                switch_frames.append(t)
                a, b = x, None
    if not pair_ids:
        # The proton stayed bonded to a single donor for the whole window.
        pair_ids.append((a, -1))
    elif b is None:
        # A switch was recorded but no second id was ever observed afterwards.
        pair_ids.append((a, -1))
    return switch_frames, pair_ids


def _reorientation_start_frames(
    seq: np.ndarray, switch_frames: list[int]
) -> np.ndarray:
    """Backtrack each switch frame to the rotation-period start ``t_s``.

    For a switch at frame ``sw``, ``t_s`` is the first frame, walking back
    from ``sw - 1`` while the identifier matches ``seq[sw - 1]``, at which
    the identifier last differed. This is the moment the proton committed
    to the host immediately preceding the switch (the carrier in the
    productive case, or the exchanger-O in the nonproductive case).
    """
    out = np.empty(len(switch_frames), dtype=np.int64)
    start_pos = 0
    for k, sw in enumerate(switch_frames):
        curr_id = int(seq[sw - 1])
        sub = seq[start_pos:sw]
        diff_positions = np.where(sub != curr_id)[0]
        if diff_positions.size == 0:
            out[k] = start_pos
        else:
            last_diff = int(diff_positions[-1])
            out[k] = start_pos + last_diff + 1
        start_pos = sw
    return out


def classify_events(tracks) -> RotationEvents:
    """Build a :class:`RotationEvents` from per-frame :class:`tracking.ProtonTracks`.

    For every proton we run the unordered-pair-switch detector twice:

    - on the donor-O identifier sequence, producing the O-pair switch frames;
    - on the donor-polyanion identifier sequence, producing the P-pair switch
      frames (productive events).

    Nonproductive O-pair switches are then the set difference between the two.
    The rotation-period start frames ``t_s`` are computed by backtracking
    along the appropriate identifier sequence: ``donor_X_id`` for productive
    events, ``donor_O_id`` for nonproductive events.
    """
    # Local import to avoid a circular dependency at module load time.
    from superproton.tracking import ProtonTracks

    if not isinstance(tracks, ProtonTracks):
        raise TypeError(
            f"classify_events expects a ProtonTracks, got {type(tracks).__name__}"
        )

    o_sw_frames: list[np.ndarray] = []
    p_sw_frames: list[np.ndarray] = []
    prod_ts: list[np.ndarray] = []
    nonprod_ts: list[np.ndarray] = []
    o_pair_id_arrays: list[np.ndarray] = []
    p_pair_id_arrays: list[np.ndarray] = []

    for h in range(tracks.n_protons):
        o_seq = tracks.donor_O_id[:, h]
        x_seq = tracks.donor_X_id[:, h]

        o_sw, o_pairs = _unordered_pair_switches(o_seq)
        p_sw, p_pairs = _unordered_pair_switches(x_seq)

        prod_ts_h = _reorientation_start_frames(x_seq, p_sw)
        p_sw_set = set(p_sw)
        nonprod_sw = [f for f in o_sw if f not in p_sw_set]
        nonprod_ts_h = _reorientation_start_frames(o_seq, nonprod_sw)

        o_sw_frames.append(np.asarray(o_sw, dtype=np.int64))
        p_sw_frames.append(np.asarray(p_sw, dtype=np.int64))
        prod_ts.append(prod_ts_h)
        nonprod_ts.append(nonprod_ts_h)
        o_pair_id_arrays.append(np.asarray(o_pairs, dtype=np.int64))
        p_pair_id_arrays.append(np.asarray(p_pairs, dtype=np.int64))

    return RotationEvents(
        n_protons=tracks.n_protons,
        n_frames=tracks.n_frames,
        frame_to_ps=tracks.frame_to_ps,
        o_pair_switch_frames=o_sw_frames,
        p_pair_switch_frames=p_sw_frames,
        productive_t_s_frames=prod_ts,
        nonproductive_t_s_frames=nonprod_ts,
        o_pair_ids=o_pair_id_arrays,
        p_pair_ids=p_pair_id_arrays,
    )


def rotation_period_histograms(tracks, events):
    """SI S7 binned O-pair and polyanion-pair rotation periods.

    Reuse the existing start-frame detector separately for all O switches;
    productive O starts need not equal productive polyanion starts.
    """
    series = {'O-pair': [], 'X-pair': []}
    for h in range(events.n_protons):
        o = events.o_pair_switch_frames[h]
        starts = _reorientation_start_frames(tracks.donor_O_id[:,h],o)
        series['O-pair'].extend((o-starts)*events.frame_to_ps)
        series['X-pair'].extend((events.p_pair_switch_frames[h]
                              -events.productive_t_s_frames[h])*events.frame_to_ps)
    rows = []
    for name, values in series.items():
        if not values: continue
        density, edges = np.histogram(values,bins='auto',density=True)
        rows.extend(dict(switch_type=name,bin_left_ps=l,bin_right_ps=r,
                         bin_center_ps=(l+r)/2,prob_density=y)
                    for l,r,y in zip(edges[:-1],edges[1:],density))
    return rows


def proton_motion_table(tracks, events):
    """Export donor changes, distinguishing local rattling and pair switches.

    A third oxygen changes the O pair; a third polyanion changes the host
    pair. Neither a donor change alone nor one productive event proves
    irreversible diffusion. Frame indices are source indices, not atom IDs.
    """
    import pandas as pd
    rows = []
    for h in range(tracks.n_protons):
        oo, pp = set(events.o_pair_switch_frames[h]), set(events.p_pair_switch_frames[h])
        changes = np.flatnonzero(tracks.donor_O_id[1:,h] != tracks.donor_O_id[:-1,h])+1
        for f in changes:
            kind = 'productive' if f in pp else 'nonproductive' if f in oo else 'rattling'
            rows.append((int(tracks.proton_ids[h]),int(tracks.frame_indices[f]),
                         int(f),kind,int(tracks.donor_O_id[f-1,h]),
                         int(tracks.donor_O_id[f,h]),int(tracks.donor_X_id[f-1,h]),
                         int(tracks.donor_X_id[f,h])))
    return pd.DataFrame(rows,columns=['proton_id','source_frame','local_frame','kind',
                       'old_oxygen_id','new_oxygen_id','old_center_id','new_center_id'])
