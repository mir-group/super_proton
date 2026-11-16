"""Unit tests for superproton.coordination.

Drives :func:`bound_protons_per_frame` and :func:`mean_Np_per_residence`
against hand-built ProtonTracks where the expected bound counts and neighbor
polyanion assignments are obvious by inspection.
"""

from __future__ import annotations

import numpy as np
import pytest

from superproton.coordination import (
    bound_protons_per_frame,
    joint_density_Np_tau,
    mean_Np_per_residence,
)
from superproton.rotation import classify_events

from _helpers import make_tracks




# ---------------------------------------------------------------------------
# bound_protons_per_frame
# ---------------------------------------------------------------------------


def test_bound_protons_per_frame_simple():
    # 3 frames, 4 protons. Polyanions 10 and 20.
    # frame 0: 3 protons on 10, 1 on 20
    # frame 1: 2 on 10, 2 on 20
    # frame 2: 4 on 10, 0 on 20
    donor_X = np.array(
        [
            [10, 10, 10, 20],
            [10, 10, 20, 20],
            [10, 10, 10, 10],
        ],
        dtype=np.int64,
    )
    donor_O = donor_X * 10  # dummy
    bound = bound_protons_per_frame(make_tracks(donor_O, donor_X))
    assert bound.polyanion_ids.tolist() == [10, 20]
    assert bound.bound_per_frame.tolist() == [[3, 1], [2, 2], [4, 0]]
    assert bound.index_of(10) == 0
    assert bound.index_of(20) == 1
    with pytest.raises(KeyError):
        bound.index_of(99)


def test_bound_protons_per_frame_conservation():
    # Total bound protons per frame must equal the number of protons.
    rng = np.random.default_rng(0)
    n_frames, n_protons, n_poly = 50, 30, 6
    poly_ids = np.arange(100, 100 + n_poly, dtype=np.int64)
    donor_X = poly_ids[rng.integers(0, n_poly, size=(n_frames, n_protons))]
    donor_O = donor_X * 10
    bound = bound_protons_per_frame(make_tracks(donor_O, donor_X))
    assert np.all(bound.bound_per_frame.sum(axis=1) == n_protons)


# ---------------------------------------------------------------------------
# mean_Np_per_residence
# ---------------------------------------------------------------------------


def _make_two_proton_tracks_with_one_residence_interval():
    """Hand-built trajectory: proton 0 has exactly one tau_OO interval,
    bounded by two O-pair switches, with a clear neighbor polyanion."""
    # Proton 0 trajectory:
    # f0-1:   donor_O=11 (P1, X=1)        -> pair {11, ?} forming
    # f2-3:   donor_O=12 (P1, X=1)        -> O-pair {11,12} forms, both on P1 (same X!)
    #                                       This violates "two-X" assumption; redesign.
    #
    # Use Os on TWO different polyanions instead:
    # f0-1:  donor_O=11 (X=1)
    # f2-3:  donor_O=21 (X=2)             -> O-pair {11,21} forms across X={1,2}
    # f4-7:  donor_O alternates between 11 and 21  (rattling within {11,21})
    # f8:    donor_O=31 (X=3)             -> O-pair switch (third id)
    # f9-12: donor_O=31 (X=3)
    # f13:   donor_O=41 (X=4)             -> new pair {31,41}, NO switch (just forming)
    # f14:   donor_O=51 (X=5)             -> O-pair switch (third id again)
    #
    # So o_pair_switch_frames[0] = [8, 14], one interval [8, 14).
    # The O-pair active over that interval is the second entry (index 2) of o_pair_ids,
    # but wait -- canonical algorithm: pairs[0] is the pair active BEFORE first switch
    # (here {11, 21}). pairs[1] is active over interval bounded by switches 0 and 1
    # ({31, 41}). pairs[2] is the "tail" pair {51, -1}.
    #
    # So for the interval [8, 14), the active pair is pairs[1] = {31, 41} on Xs 3 and 4.
    #
    # During the interval, donor_O is 31 (f8-12) then 41 (f13). So donor_X is 3 then 4.
    # neighbor_X is 4 then 3. Mean N_p over the interval is average of
    # bound[t, neighbor_X[t]] for t in [8, 14).
    donor_O = np.array([
        [11, 11, 21, 21, 11, 21, 11, 21, 31, 31, 31, 31, 31, 41, 51],
        # Proton 1: contributes a steady bound count on every polyanion it visits.
        [21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21],
    ]).T
    donor_X = np.array([
        [1, 1, 2, 2, 1, 2, 1, 2, 3, 3, 3, 3, 3, 4, 5],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ]).T
    return make_tracks(donor_O, donor_X)


def test_mean_Np_per_residence_returns_aligned_tau_and_Np():
    tracks = _make_two_proton_tracks_with_one_residence_interval()
    events = classify_events(tracks)
    bound = bound_protons_per_frame(tracks)
    tau, np_arr = mean_Np_per_residence(tracks, events, bound)
    # Proton 0 has exactly one interval; proton 1 has none.
    assert tau.shape == np_arr.shape
    assert tau.size == 1
    # Interval is frames 8..13 (inclusive) i.e. length 6 frames -> 6 * 0.05 = 0.3 ps.
    assert tau[0] == pytest.approx(6 * 0.05)
    # X4 is visited once, X3 five times. The fixed neighbor is X4 for
    # all six frames, so its population is [0,0,0,0,0,1].
    assert np_arr[0] == pytest.approx(1/6)
    # Retain the previous instantaneous-partner diagnostic explicitly.
    _, instantaneous = mean_Np_per_residence(tracks, events, bound, neighbor="instantaneous")
    assert instantaneous[0] == pytest.approx(0)



def test_mean_Np_per_residence_handles_no_intervals():
    # Single-frame trajectory: no intervals at all.
    donor_O = np.array([[11], [11], [11]])
    donor_X = np.array([[1], [1], [1]])
    tracks = make_tracks(donor_O, donor_X)
    events = classify_events(tracks)
    bound = bound_protons_per_frame(tracks)
    tau, np_arr = mean_Np_per_residence(tracks, events, bound)
    assert tau.size == 0
    assert np_arr.size == 0


# ---------------------------------------------------------------------------
# joint_density_Np_tau
# ---------------------------------------------------------------------------


def test_joint_density_normalizes_to_one():
    rng = np.random.default_rng(7)
    tau = rng.exponential(50.0, size=2000)
    np_vals = 1.8 + 0.2 * rng.standard_normal(2000)
    H, tau_edges, np_edges = joint_density_Np_tau(
        tau, np_vals, tau_bins=40, np_bins=20, tau_range=(0, 300)
    )
    dtau = tau_edges[1] - tau_edges[0]
    dnp = np_edges[1] - np_edges[0]
    integral = H.sum() * dtau * dnp
    # All samples fall inside the binned range, so integral should be ~ 1.
    assert integral == pytest.approx(1.0, abs=0.05)


def test_coordination_tie_uses_first_encounter_not_atom_identifier():
    from superproton.coordination import CoordinationData, residence_coordination
    hosts = np.array([90,90,10,10,90])
    bound = CoordinationData(5,2,np.array([10,90]),np.tile([3,1],(5,1)))
    # Equal occupancy within [1,5), but 90 was seen first in the trajectory.
    tau, number = residence_coordination(np.array([1,5]),hosts,bound,frame_to_ps=1)
    np.testing.assert_array_equal(tau,[4])
    np.testing.assert_array_equal(number,[1])
