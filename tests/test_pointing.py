"""Unit tests for superproton.pointing.

Covers the well-specified low-level pieces: distance computation, the
acceptor-hysteresis state machine, and the outcome-fraction / lookahead
post-processing. The full detect_attempts pipeline is exercised by a smoke
test only; quantitative validation against the manuscript's pointing
figures needs a production trajectory and lives in ``test_reference.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from superproton.pointing import (
    DEFAULT_D_BREAK_ANG,
    DEFAULT_D_FORM_ANG,
    HBondDistances,
    PointingData,
    _committed_acceptor_with_hysteresis,
    crossing_lookahead,
    detect_attempts,
    hbond_distances,
    outcome_fractions,
)
from superproton.reorientation import HOPositions

from _helpers import make_tracks




# ---------------------------------------------------------------------------
# hbond_distances
# ---------------------------------------------------------------------------


def test_hbond_distances_basic():
    # Single proton, 2 frames, donor and acceptor at known distances.
    donor_O = np.array([[101], [101]])
    donor_X = np.array([[1], [1]])
    acc_O = np.array([[102], [102]])
    acc_X = np.array([[2], [2]])
    tracks = make_tracks(donor_O, donor_X, acc_O, acc_X)

    H_pos = np.array([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    O_pos = np.array(
        [
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],  # f0: O101 1 A away, O102 3 A away
            [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],  # f1: O101 1 A away, O102 3 A away
        ]
    )
    positions = HOPositions(
        H_pos=H_pos,
        O_pos=O_pos,
        H_ids=np.array([1]),
        O_ids=np.array([101, 102]),
        frame_indices=np.arange(2, dtype=int),
    )
    # 100 A cell: far larger than these 1-3 A separations, so the
    # minimum-image folding is a no-op and the expected values are unchanged.
    d = hbond_distances(positions, tracks, cell=100.0 * np.eye(3))
    assert d.donor_O_dist[0, 0] == pytest.approx(1.0)
    assert d.acceptor_O_dist[0, 0] == pytest.approx(3.0)
    assert d.donor_O_dist[1, 0] == pytest.approx(1.0)
    assert d.acceptor_O_dist[1, 0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# _committed_acceptor_with_hysteresis
# ---------------------------------------------------------------------------


def test_hysteresis_forms_holds_and_carries_forward():
    """Commit below d_form, hold through the band, and carry the last pair on release.

    Seven frames for one proton. The acceptor polyanion is 7 for the first
    five frames and 8 for the last two, with the H-bond distance scripted to
    walk through every state:

        f0  3.0  too far, and nothing committed yet          -> -1
        f1  1.5  below d_form, commits                       ->  7
        f2  2.0  inside the hysteresis band, holds           ->  7
        f3  2.6  above d_break, releases                     ->  7 (carried)
        f4  3.0  still released                              ->  7 (carried)
        f5  3.0  new candidate 8, still too far              ->  7 (carried)
        f6  1.5  below d_form, commits to the new one        ->  8

    The two carry-forward frames are the point: an earlier version substituted
    the proton's own donor_X there, collapsing the polyanion pair to {A, A}.
    detect_attempts read every such collapse and re-form as a complete pointing
    attempt that instantly reverted, inflating the attempt count roughly
    fivefold and driving the reverted fraction toward 1. f6 checks the carry is
    not sticky: a genuine new commitment still takes effect.
    """
    donor_O = np.full((7, 1), 101)
    donor_X = np.full((7, 1), 1)
    acc_O = np.array([[102]] * 5 + [[103]] * 2)
    acc_X = np.array([[7]] * 5 + [[8]] * 2)
    tracks = make_tracks(donor_O, donor_X, acc_O, acc_X)

    distances = HBondDistances(
        donor_O_dist=np.ones_like(donor_O, dtype=float),
        acceptor_O_dist=np.array([[3.0], [1.5], [2.0], [2.6], [3.0], [3.0], [1.5]]),
        frame_indices=np.arange(7, dtype=int),
    )
    committed = _committed_acceptor_with_hysteresis(
        distances, tracks, d_form=DEFAULT_D_FORM_ANG, d_break=DEFAULT_D_BREAK_ANG
    )
    assert committed[:, 0].tolist() == [-1, 7, 7, 7, 7, 7, 8]
    # Never the donor's own polyanion, which would be a degenerate pair.
    assert 1 not in committed[:, 0].tolist()


# ---------------------------------------------------------------------------
# detect_attempts smoke test
# ---------------------------------------------------------------------------


def test_detect_attempts_runs_on_minimal_trajectory():
    """Just exercise the full state machine on a short trajectory; assert
    we get a valid PointingData with consistent fields."""
    n_frames = 12
    donor_O = np.full((n_frames, 1), 101)
    donor_X = np.full((n_frames, 1), 1)
    # Acceptor flips between two polyanions across frames.
    acc_O = np.array([[102]] * 4 + [[103]] * 4 + [[102]] * 4)
    acc_X = np.array([[2]] * 4 + [[3]] * 4 + [[2]] * 4)
    tracks = make_tracks(donor_O, donor_X, acc_O, acc_X)
    distances = HBondDistances(
        donor_O_dist=np.ones_like(donor_O, dtype=float),
        acceptor_O_dist=np.full((n_frames, 1), 1.5),  # always committed
        frame_indices=np.arange(n_frames, dtype=int),
    )
    data = detect_attempts(tracks, distances)
    assert isinstance(data, PointingData)
    for a in data.attempts:
        assert a.outcome in {"committed", "reverted", "superseded"}
        assert 0 <= a.fresh_frame < n_frames
        assert 0 <= a.outcome_frame < n_frames


# ---------------------------------------------------------------------------
# outcome_fractions
# ---------------------------------------------------------------------------


def test_outcome_fractions_sum_to_one():
    from superproton.pointing import PointingAttempt

    # Keyword form: PointingAttempt gains fields over time, and only the
    # outcome labels matter to outcome_fractions.
    def attempt(proton, fresh, pair, pointed_at, outcome, outcome_frame):
        return PointingAttempt(
            proton_index=proton,
            fresh_frame=fresh,
            polyanion_pair_at_fresh=pair,
            donor_X_at_fresh=pair[0],
            pointed_at=pointed_at,
            outcome=outcome,
            outcome_frame=outcome_frame,
        )

    attempts = [
        attempt(0, 1, (1, 2), 3, "committed", 5),
        attempt(0, 6, (1, 3), 4, "committed", 9),
        attempt(0, 10, (1, 4), 5, "reverted", 12),
        attempt(1, 2, (5, 6), 7, "superseded", 4),
    ]
    fracs = outcome_fractions(PointingData(attempts=attempts))
    assert fracs["committed"] == pytest.approx(0.5)
    assert fracs["reverted"] == pytest.approx(0.25)
    assert fracs["superseded"] == pytest.approx(0.25)
    assert sum(fracs.values()) == pytest.approx(1.0)


def test_outcome_fractions_handle_empty():
    fracs = outcome_fractions(PointingData(attempts=[]))
    assert fracs == {"committed": 0.0, "reverted": 0.0, "superseded": 0.0}


# ---------------------------------------------------------------------------
# crossing_lookahead
# ---------------------------------------------------------------------------


def test_crossing_lookahead_monotonic_and_bounded():
    """Synthetic: 100 attempts each crossing at a different fixed frame.
    The cumulative fraction across lookahead must be monotonic non-decreasing
    and bounded in [0, 1]."""
    from superproton.pointing import PointingAttempt

    n_frames = 200
    # All attempts share fresh_frame=0, but donor_X switches to pointed_at at
    # different times for each attempt. Simulate this by giving each attempt
    # its own proton in donor_X_id with the pointed_at appearing at frame
    # k for the k-th attempt.
    donor_X = np.zeros((n_frames, 100), dtype=np.int64)
    for h in range(100):
        donor_X[h:, h] = 99  # pointed_at = 99 from frame h onward
    donor_O = np.full_like(donor_X, 1)
    acc_O = np.full_like(donor_X, 2)
    acc_X = np.full_like(donor_X, 99)
    tracks = make_tracks(donor_O, donor_X, acc_O, acc_X)
    attempts = [
        PointingAttempt(
            proton_index=h,
            fresh_frame=0,
            polyanion_pair_at_fresh=(0, 0),
            donor_X_at_fresh=0,
            pointed_at=99,
            outcome="committed",
            outcome_frame=h,
        )
        for h in range(100)
    ]
    data = PointingData(attempts=attempts)
    grid_ps = np.array([0.5, 1.0, 2.5, 5.0, 10.0])
    frac = crossing_lookahead(data, tracks, grid_ps)
    assert frac.shape == grid_ps.shape
    assert np.all(np.diff(frac) >= -1e-12)  # non-decreasing
    assert frac[-1] == pytest.approx(1.0)
    assert (frac >= 0.0).all() and (frac <= 1.0).all()


def test_episode_lookahead_counts_after_end_and_across_later_episodes():
    from superproton.pointing import episode_crossing_lookahead
    # Pair {1,2} lasts frames [0,5); proton finally reaches 2 at f7, after
    # a different pair {1,3} has intervened. Only 2 ps of *added* time needed.
    donor=np.array([1,1,1,1,1,1,1,2,2,2])[:,None]
    acceptor=np.array([2,2,2,2,2,3,3,3,3,3])[:,None]
    r=episode_crossing_lookahead(donor,acceptor,[1,2,4],frame_to_ps=1)
    assert r.n_fresh == 3
    np.testing.assert_array_equal(r.n_crossed,[0,1,1])


def test_episode_lookahead_excludes_returns_and_carries_pair_across_gaps():
    from superproton.pointing import episode_crossing_lookahead
    donor=np.array([1,1,2,2,2,2])[:,None]
    acceptor=np.array([2,-1,1,-1,3,3])[:,None]
    r=episode_crossing_lookahead(donor,acceptor,[1,2],frame_to_ps=1)
    # First unordered {1,2} episode survives the donor swap and acceptor gap;
    # the final fresh {2,3} episode is censored but remains in the denominator.
    assert r.n_fresh == 2
    np.testing.assert_allclose(r.fraction,[.5,.5])
    returns=episode_crossing_lookahead(np.array([1,2,2,2])[:,None],
                                      np.array([-1,1,1,1])[:,None],[1],frame_to_ps=1)
    assert returns.n_fresh == 0
