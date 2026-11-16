"""Hydrogen-bond pointing attempts and outcome classification.

Implements the SI Section "Commitment of hydrogen-bond pointing to proton
transfer" (manuscript Figures S15-S17). For each proton we identify the
hydrogen-bond acceptor with a distance hysteresis (form when |H...O| < d_form,
break when > d_break), record fresh pointing events (the acceptor moves onto
a polyanion outside the proton's current pair), and classify each attempt's
outcome as **committed** (the proton transfers its covalent bond to the
pointed-at polyanion), **reverted** (the polyanion pair returns to its pre-
pointing state first), or **superseded** (the next pointing event toward a
different polyanion begins first).

No dwell-time condition is applied, consistent with the manuscript's
published pointing figures.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from superproton.io import min_image

from superproton.reorientation import HOPositions
from superproton.tracking import ProtonTracks


DEFAULT_D_FORM_ANG = 1.8
DEFAULT_D_BREAK_ANG = 2.4


@dataclass(frozen=True)
class HBondDistances:
    """Per-frame H-to-donor-O and H-to-acceptor-O distances in Angstroms."""

    donor_O_dist: np.ndarray  # (n_frames, n_protons)
    acceptor_O_dist: np.ndarray  # (n_frames, n_protons)
    frame_indices: np.ndarray


def hbond_distances(
    positions: HOPositions, tracks: ProtonTracks, *, cell: np.ndarray
) -> HBondDistances:
    """Compute per-proton, per-frame |H - O| distances for donor and acceptor.

    Both ``positions`` and ``tracks`` must share the same frame range. The
    function looks up donor and acceptor O positions from ``positions.O_pos``
    using the identifier maps in ``HOPositions`` and the per-frame O ids in
    ``ProtonTracks``.

    ``cell`` is the 3x3 row-vector simulation cell, from
    :func:`superproton.io.cell`. Distances are folded to the minimum image,
    which is **required**, not optional: the production dumps store unwrapped
    coordinates while the donor and acceptor identities come from ovito's
    periodic nearest-neighbour search. Whenever a proton and its own bonded
    oxygen sit in different periodic images the raw separation is a lattice
    vector -- covalent O-H "bonds" of up to 35 A, in about 13% of entries on a
    CsH2PO4 production run. Those spurious distances all exceed ``d_break``,
    so the hysteresis in :func:`_committed_acceptor_with_hysteresis` breaks
    and re-forms continuously and the pointing statistics collapse.
    """
    if not np.array_equal(positions.frame_indices, tracks.frame_indices):
        raise ValueError("positions and tracks frame_indices must match")

    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError(f"cell must be a 3x3 matrix, got {cell.shape}")
    inv_cell = np.linalg.inv(cell)

    def _min_image(diff: np.ndarray) -> np.ndarray:
        return min_image(diff, cell, inv_cell)

    n_frames, n_protons = tracks.donor_O_id.shape
    o_id_to_idx = {int(p): i for i, p in enumerate(positions.O_ids)}
    h_id_to_idx = {int(p): i for i, p in enumerate(positions.H_ids)}
    proton_cols = np.array([h_id_to_idx[int(p)] for p in tracks.proton_ids])

    donor_dist = np.zeros((n_frames, n_protons), dtype=float)
    acc_dist = np.zeros((n_frames, n_protons), dtype=float)
    for t in range(n_frames):
        H = positions.H_pos[t, proton_cols, :]
        donor_O_cols = np.array(
            [o_id_to_idx[int(o)] for o in tracks.donor_O_id[t]]
        )
        acc_O_cols = np.array(
            [o_id_to_idx[int(o)] for o in tracks.acceptor_O_id[t]]
        )
        donor_dist[t] = np.linalg.norm(
            _min_image(positions.O_pos[t, donor_O_cols, :] - H), axis=1
        )
        acc_dist[t] = np.linalg.norm(
            _min_image(positions.O_pos[t, acc_O_cols, :] - H), axis=1
        )
    return HBondDistances(
        donor_O_dist=donor_dist,
        acceptor_O_dist=acc_dist,
        frame_indices=tracks.frame_indices.copy(),
    )


@dataclass(frozen=True)
class PointingAttempt:
    """A single fresh pointing event and how it resolved."""

    proton_index: int
    fresh_frame: int
    polyanion_pair_at_fresh: tuple[int, int]
    """The proton's polyanion pair just before the pointing event.

    Sorted by identifier, so this is an **unordered** pair: neither entry is
    the donor by position. Use :attr:`donor_X_at_fresh` when you need to know
    which side the proton was covalently bound to.
    """

    donor_X_at_fresh: int
    """The polyanion the proton was covalently bound to at ``fresh_frame``.

    This is the polyanion a committed transfer vacates, and therefore the one a
    subsequent pointing event must not treat as fresh (that would be a return
    event, which the analysis excludes).
    """

    pointed_at: int  # X the new acceptor sits on
    outcome: str  # "committed", "reverted", or "superseded"
    outcome_frame: int


@dataclass(frozen=True)
class PointingData:
    """Container for all fresh pointing attempts in a trajectory."""

    attempts: list[PointingAttempt]

    def by_outcome(self) -> dict[str, list[PointingAttempt]]:
        out: dict[str, list[PointingAttempt]] = {
            "committed": [],
            "reverted": [],
            "superseded": [],
        }
        for a in self.attempts:
            out[a.outcome].append(a)
        return out


def _committed_acceptor_with_hysteresis(
    distances: HBondDistances,
    tracks: ProtonTracks,
    *,
    d_form: float,
    d_break: float,
) -> np.ndarray:
    """Return ``(n_frames, n_protons)`` int array: committed acceptor X each frame.

    Apply the H-bond distance hysteresis. A second-nearest O becomes the
    committed acceptor when its H-distance falls below ``d_form`` and remains
    so until the distance exceeds ``d_break``.

    While no acceptor is committed the **last committed** polyanion is carried
    forward, so a momentary loss of the hydrogen bond is a gap inside one
    pointing episode rather than the end of it. Substituting the proton's own
    ``donor_X`` would collapse the pair to ``{A, A}`` during these gaps,
    creating spurious reverted attempts in :func:`detect_attempts`. Frames
    before the first commitment have no pair yet and are marked ``-1``.
    """
    n_frames, n_protons = tracks.acceptor_X_id.shape
    out = np.zeros((n_frames, n_protons), dtype=np.int64)
    # -1 until this proton has committed to an acceptor for the first time.
    current = np.full(n_protons, -1, dtype=np.int64)
    committed = np.zeros(n_protons, dtype=bool)
    for t in range(n_frames):
        d = distances.acceptor_O_dist[t]
        cand_X = tracks.acceptor_X_id[t]
        # Re-form events: candidate within d_form
        new_commit = (d < d_form) & ~committed
        current[new_commit] = cand_X[new_commit]
        committed |= new_commit
        # Break events: candidate beyond d_break -> drop commitment
        break_mask = (d > d_break) & committed
        committed[break_mask] = False
        # `current` is deliberately left untouched where not committed: the
        # last committed acceptor persists across the gap.
        out[t] = current
    return out


def detect_attempts(
    tracks: ProtonTracks,
    distances: HBondDistances,
    *,
    d_form: float = DEFAULT_D_FORM_ANG,
    d_break: float = DEFAULT_D_BREAK_ANG,
) -> PointingData:
    """Identify fresh pointing events and classify each outcome.

    A pointing event begins when the committed acceptor moves onto a polyanion
    outside the current ``{donor_X, acceptor_X}`` pair. A pointing event is
    **fresh** when the pointed-at polyanion differs from the polyanion the
    proton last left (excludes return events).

    Outcome per fresh event:

    - "committed": the proton's donor_X subsequently becomes the pointed-at
      polyanion before the next pointing event or trajectory end.
    - "reverted": the polyanion pair returns to its pre-pointing value
      before a covalent transfer.
    - "superseded": a new pointing event toward a different polyanion
      begins before either of the above.

    Attempts left undecided at the end of the trajectory are excluded.
    """
    committed = _committed_acceptor_with_hysteresis(
        distances, tracks, d_form=d_form, d_break=d_break
    )
    n_frames, n_protons = tracks.donor_X_id.shape

    attempts: list[PointingAttempt] = []
    for h in range(n_protons):
        donor_X_h = tracks.donor_X_id[:, h]
        acc_X_h = committed[:, h]

        # Initial pair = {donor_X[0], committed_acceptor[0]}, or a sentinel
        # while no acceptor has been committed yet.
        a = int(donor_X_h[0])
        b = int(acc_X_h[0])
        pair = (min(a, b), max(a, b)) if b >= 0 else (-1, -1)
        last_left: int | None = None

        pending: PointingAttempt | None = None

        for t in range(1, n_frames):
            d = int(donor_X_h[t])
            a_new = int(acc_X_h[t])
            if a_new < 0:
                # No acceptor has been committed yet for this proton.
                continue
            # NOTE: a_new == d is left in place deliberately. Both of the
            # proton's nearest oxygens can belong to the SAME polyanion -- an
            # XO4 is tetrahedral -- giving a momentary {A, A} pair. That is a
            # real intermediate state of the state machine, not noise: treating
            # those frames as gaps and carrying the previous pair forward was
            # measured against the origin script and made the agreement much
            # worse (221 attempts at 12.7% committed, against 691 at 89.6%),
            # because a genuine later transition then gets misattributed.
            new_pair = (min(d, a_new), max(d, a_new))
            if new_pair == pair:
                continue
            if pair == (-1, -1):
                # This proton's first commitment. There is no earlier pair for
                # it to have pointed away from, so this is where its history
                # starts, not a pointing attempt.
                pair = new_pair
                continue

            # The pair changed: resolve the pending attempt before opening a new one.
            if pending is not None:
                if d == pending.pointed_at:
                    outcome = "committed"
                elif new_pair == pending.polyanion_pair_at_fresh:
                    outcome = "reverted"
                else:
                    # The hydrogen bond redirected to a third polyanion.
                    outcome = "superseded"
                pending = replace(pending, outcome=outcome, outcome_frame=t)
                attempts.append(pending)
                if outcome == "committed":
                    # The proton has just moved onto `pointed_at`, vacating the
                    # polyanion it was bound to when the attempt opened.
                    last_left = pending.donor_X_at_fresh
                pending = None

            # Detect a new pointing event if the acceptor moved to a polyanion
            # outside the previous pair (i.e. donor stays, acceptor is a new X).
            if a_new not in pair and a_new != last_left:
                pending = PointingAttempt(
                    proton_index=h,
                    fresh_frame=t,
                    polyanion_pair_at_fresh=pair,
                    donor_X_at_fresh=d,
                    pointed_at=a_new,
                    outcome="pending",
                    outcome_frame=-1,
                )
            pair = new_pair

    return PointingData(attempts=attempts)


def outcome_fractions(data: PointingData) -> dict[str, float]:
    """Fraction of attempts in each of committed / reverted / superseded."""
    by_kind = data.by_outcome()
    total = sum(len(v) for v in by_kind.values())
    if total == 0:
        return {"committed": 0.0, "reverted": 0.0, "superseded": 0.0}
    return {k: len(v) / total for k, v in by_kind.items()}


def crossing_lookahead(
    data: PointingData,
    tracks: ProtonTracks,
    lookahead_ps_grid: np.ndarray,
) -> np.ndarray:
    """Fraction of attempts whose pointed-at polyanion becomes covalent within each lookahead.

    This diagnostic measures from attempt opening. For the manuscript S20
    post-episode definition, use episode_crossing_lookahead instead. Returns
    a 1D array aligned with
    ``lookahead_ps_grid``.
    """
    dt = float(tracks.frame_to_ps)
    frames_grid = np.ceil(np.asarray(lookahead_ps_grid, dtype=float) / dt).astype(int)
    n_attempts = len(data.attempts)
    if n_attempts == 0:
        return np.zeros(frames_grid.size, dtype=float)
    crossed_within = np.zeros((n_attempts, frames_grid.size), dtype=bool)

    for i, a in enumerate(data.attempts):
        donor_X_h = tracks.donor_X_id[:, a.proton_index]
        # Earliest frame after fresh_frame where donor_X equals pointed_at.
        candidates = np.where(donor_X_h[a.fresh_frame :] == a.pointed_at)[0]
        if candidates.size == 0:
            continue
        first_cross = int(candidates[0])
        for j, look in enumerate(frames_grid):
            if first_cross <= look:
                crossed_within[i, j] = True

    return crossed_within.mean(axis=0)


@dataclass(frozen=True)
class LookaheadResult:
    """SI S20 crossings after fresh pointing episodes (including censored episodes)."""
    lookahead_ps: np.ndarray
    fraction: np.ndarray
    n_fresh: int
    n_crossed: np.ndarray


def read_pointing_hosts(pipeline, material, *, start_frame=200, end_frame=None,
                        stride=1, d_form=1.8, d_break=2.4,
                        n_neighbors=16, n_oxygen=6):
    """Track the committed oxygen identity with six-O hysteresis.

    Unlike the two-neighbor diagnostic above, a committed oxygen is followed
    even after it ceases to be second nearest. By default, scan 16 mixed-species
    neighbors and retain six oxygens. Distances use float32 precision.
    Returns donor and committed-acceptor polyanion IDs; -1 means no acceptor.
    Memory scales with two host arrays, not the full six-O distance history.
    """
    from ovito.data import NearestNeighborFinder
    from superproton.io import check_stable_particle_order, species_map
    from superproton.tracking import parent_polyanions

    if stride <= 0 or not 0 < d_form < d_break:
        raise ValueError("Require positive stride and 0 < d_form < d_break")
    if not 2 <= n_oxygen <= n_neighbors <= 64:
        raise ValueError("Require 2 <= n_oxygen <= n_neighbors <= 64")
    stop = int(pipeline.source.num_frames) if end_frame is None else end_frame
    frames = np.arange(start_frame, stop, stride, dtype=int)
    if not frames.size or start_frame < 0 or stop > pipeline.source.num_frames:
        raise ValueError("Empty or out-of-bounds frame range")
    check_stable_particle_order(pipeline, frames)
    parents = parent_polyanions(pipeline, material, frame=0)
    species = {v:k for k,v in species_map(pipeline, material).items()}
    data = pipeline.compute(start_frame)
    h_idx = np.flatnonzero(np.asarray(data.particles.particle_types) == species['H'])
    # Dense particle-row lookup: independent of the magnitude of atom IDs.
    # The extra row is the missing-oxygen sentinel, with no parent (-1).
    ids = np.asarray(data.particles.identifiers)
    missing = len(ids)
    parent_by_row = np.full(missing + 1, -1, dtype=np.int64)
    oxygen = np.flatnonzero(np.asarray(data.particles.particle_types) == species['O'])
    parent_by_row[oxygen] = [parents[int(ids[o])] for o in oxygen]
    donors = np.empty((len(frames),len(h_idx)),dtype=int)
    acceptors = np.full_like(donors,-1)
    committed = np.full(len(h_idx), missing, dtype=int)
    for row, frame in enumerate(frames):
        data = pipeline.compute(int(frame))
        types = np.asarray(data.particles.particle_types)
        neighbors, vec = NearestNeighborFinder(n_neighbors, data).find_all(h_idx)
        is_o = types[neighbors] == species['O']
        order = np.argsort(~is_o,axis=1,kind='stable')[:,:n_oxygen]
        oxy = np.take_along_axis(neighbors,order,axis=1)
        valid = np.take_along_axis(is_o,order,axis=1)
        oxy = np.where(valid, oxy, missing)
        distance = np.linalg.norm(np.take_along_axis(vec, order[:,:,None], axis=1), axis=2).astype(np.float32)
        distance[~valid] = np.inf
        if np.any(oxy[:,0] == missing):
            raise ValueError("No donor oxygen among 16 neighbors")
        matches = (oxy == committed[:,None]) & (committed[:,None] != missing)
        current_distance = np.min(np.where(matches,distance,np.inf),axis=1)
        broken = (current_distance >= d_break) | (committed == oxy[:,0])
        committed[broken] = missing
        candidate = (distance[:,1:] <= d_form) & (oxy[:,1:] != oxy[:,:1])
        take = (committed == missing) & candidate.any(axis=1)
        first = np.argmax(candidate,axis=1)+1
        committed[take] = oxy[np.flatnonzero(take), first[take]]
        donors[row] = parent_by_row[oxy[:,0]]
        acceptors[row] = parent_by_row[committed]
    return donors, acceptors


@dataclass(frozen=True)
class _PairEpisodes:
    """Run-length-encoded pair history for one proton, including gap sentinels."""

    starts: np.ndarray
    ends: np.ndarray
    pairs: np.ndarray
    previous_donor: np.ndarray


def _pair_episodes(donor, acceptor) -> _PairEpisodes:
    """Build pair runs without per-frame Python tuples.

    A gap carries the entire most recent pair: the donor may change while
    there is no committed acceptor. Same-host pairs remain real runs, and
    frames before the first acceptor use the distinct ``(-1, -1)`` sentinel.
    Only the compact run arrays survive this function; temporary per-frame
    arrays are restricted to one proton at a time.
    """
    nf = len(donor)
    if not nf:
        empty = np.empty(0, dtype=np.int64)
        return _PairEpisodes(empty, empty.copy(), np.empty((0, 2), dtype=np.int64), empty.copy())
    frame = np.arange(nf)
    last_valid = np.maximum.accumulate(np.where(acceptor >= 0, frame, -1))
    source = np.maximum(last_valid, 0)
    prior_d = donor[source]
    prior_a = acceptor[source]
    pairs = np.column_stack((np.minimum(prior_d, prior_a), np.maximum(prior_d, prior_a)))
    pairs[last_valid < 0] = -1
    starts = np.r_[0, np.flatnonzero(np.any(pairs[1:] != pairs[:-1], axis=1)) + 1]
    ends = np.r_[starts[1:], nf]

    # The previous DISTINCT donor excludes return pointing episodes. Its
    # change can occur inside a pair run (rattling), so derive it separately.
    changed = np.r_[False, donor[1:] != donor[:-1]]
    last_change = np.maximum.accumulate(np.where(changed, frame, 0))[starts]
    previous = np.where(last_change > 0, donor[np.maximum(last_change - 1, 0)], -1)
    return _PairEpisodes(starts, ends, pairs[starts], previous)


def episode_crossing_lookahead(
    donor_hosts, acceptor_hosts, lookahead_ps_grid, *, frame_to_ps=0.05
) -> LookaheadResult:
    """Published SI S20 definition: crossing during or after a fresh episode.

    A pair episode is a maximal constant *unordered*, inter-polyanion pair;
    carry the last pair across gaps without a committed acceptor. Exclude
    returns to the donor's previous distinct host. Measure added lookahead
    from the episode END, and allow the search across later episodes. All
    fresh episodes enter the denominator, including those censored by the
    trajectory end, matching the saved P_cross_lk*_fresh columns.
    """
    donor, acceptor = np.asarray(donor_hosts), np.asarray(acceptor_hosts)
    grid = np.asarray(lookahead_ps_grid, dtype=float)
    if donor.shape != acceptor.shape or donor.ndim != 2:
        raise ValueError("Host arrays must have matching (frames,protons) shapes")
    if grid.ndim != 1 or not grid.size or np.any(grid < 0) or frame_to_ps <= 0:
        raise ValueError("Require nonnegative lookaheads and a positive frame interval")
    limits = np.rint(grid / frame_to_ps).astype(int)
    counts = np.zeros(len(grid), dtype=int)
    n_fresh = 0
    n_frames = len(donor)
    max_lookahead = int(limits.max())
    for h in range(donor.shape[1]):
        d = donor[:, h]
        runs = _pair_episodes(d, acceptor[:, h])
        for run, (start, end) in enumerate(zip(runs.starts, runs.ends)):
            a, b = runs.pairs[run]
            if a < 0 or a == b:
                continue
            target = a if b == d[start] else b
            if d[start] < 0 or target == d[start] or target == runs.previous_donor[run]:
                continue
            n_fresh += 1
            stop = min(n_frames, end + max_lookahead)
            cross = np.flatnonzero(d[start:stop] == target)
            if cross.size:
                needed = start + int(cross[0]) - end
                counts += needed <= limits
    return LookaheadResult(
        grid, counts / n_fresh if n_fresh else np.zeros(len(grid)), n_fresh, counts
    )


def pointing_outcomes(
    donor_hosts, acceptor_hosts, *, frame_to_ps=0.05, proton_ids=None, source_frames=None
):
    """Event-bounded outcomes from identity-preserving committed host histories.

    Uses the SI S18/S19 definition. An episode is a maximal constant unordered
    inter-polyanion pair; no-acceptor gaps carry the last pair. Return episodes
    are retained with ``fresh=False``. The last undecided attempt is censored,
    not failed. Published fractions use fresh, decided attempts only.
    Completion delay starts at episode onset;
    this is distinct from the S20 post-episode lookahead clock.

    Returns a pandas DataFrame, including persistent IDs, original frame
    indices, outcome time and observed completion delay (NaN if unobserved).
    Source-frame times assume uniform saved-frame spacing and frame 0 at t=0.
    """
    import pandas as pd

    d, a = np.asarray(donor_hosts), np.asarray(acceptor_hosts)
    if d.ndim != 2 or d.shape != a.shape or frame_to_ps <= 0:
        raise ValueError("Matching (frames, protons) arrays and positive dt required")
    nf, nh = d.shape
    ids = np.arange(nh) if proton_ids is None else np.asarray(proton_ids)
    frames = np.arange(nf) if source_frames is None else np.asarray(source_frames)
    if len(ids) != nh or len(frames) != nf:
        raise ValueError("ID/frame metadata does not match host arrays")
    step = frames[1] - frames[0] if nf > 1 else 1
    if step <= 0 or (nf > 1 and not np.all(np.diff(frames) == step)):
        raise ValueError("Source frames must be uniformly increasing")
    raw_dt = frame_to_ps / step
    columns = [
        "attempt_id",
        "proton_id",
        "proton_index",
        "donor_id",
        "target_id",
        "start_local",
        "end_local",
        "start_frame",
        "end_frame",
        "start_ps",
        "outcome",
        "outcome_local",
        "outcome_frame",
        "outcome_ps",
        "delay_ps",
        "fresh",
    ]
    rows = []
    for h in range(nh):
        runs = _pair_episodes(d[:, h], a[:, h])
        episodes = np.flatnonzero(runs.pairs[:, 0] != runs.pairs[:, 1])
        for ei, run in enumerate(episodes):
            start, end = runs.starts[run], runs.ends[run]
            pair = runs.pairs[run]
            donor = int(d[start, h])
            target = pair[0] if pair[1] == donor else pair[1]
            if donor < 0 or target < 0 or donor == target:
                continue
            before = runs.pairs[run - 1] if run else np.array([-1, -1])
            if np.all(before == -1):
                before = np.array([donor, donor])
            nxt = int(episodes[ei + 1]) if ei + 1 < len(episodes) else None
            limit = runs.starts[nxt] if nxt is not None else nf
            cross = np.flatnonzero(d[start:limit, h] == target)
            cross = start + int(cross[0]) if cross.size else None
            # Search run boundaries, not every intervening frame. The first
            # matching run begins at the exact earliest reversion frame.
            stop_run = nxt if nxt is not None else len(runs.starts)
            reverted = np.flatnonzero(np.all(runs.pairs[run:stop_run] == before, axis=1))
            revert = runs.starts[run + reverted[0]] if reverted.size else None
            if cross is not None and (revert is None or cross < revert):
                outcome, at = "committed", cross
            elif revert is not None:
                outcome, at = "reverted", revert
            elif nxt is not None:
                outcome, at = (
                    ("reverted" if np.array_equal(runs.pairs[nxt], before) else "superseded"),
                    runs.starts[nxt],
                )
            else:
                outcome, at = "censored", None
            rows.append(
                [
                    len(rows),
                    int(ids[h]),
                    h,
                    donor,
                    target,
                    start,
                    end,
                    int(frames[start]),
                    int(frames[end - 1] + step),
                    frames[start] * raw_dt,
                    outcome,
                    at,
                    None if at is None else int(frames[at]),
                    np.nan if at is None else frames[at] * raw_dt,
                    (at - start) * frame_to_ps if outcome == "committed" else np.nan,
                    target != runs.previous_donor[run],
                ]
            )
    return pd.DataFrame(rows, columns=columns)


def pointing_summary(attempts):
    """Fresh-event outcome counts and fractions; censored rows are explicit."""
    import pandas as pd
    fresh = attempts.loc[attempts.fresh.astype(bool)]
    decided = fresh.loc[fresh.outcome != 'censored']
    counts = fresh.outcome.value_counts()
    names = ['committed','reverted','superseded','censored']
    return pd.DataFrame({'outcome': names, 'count': [int(counts.get(n,0)) for n in names],
        'fraction_decided': [counts.get(n,0)/len(decided)
            if len(decided) and n != 'censored' else np.nan for n in names]})
