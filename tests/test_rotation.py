"""Unit tests for the pair-switch detection and rotation-period backtracking.

Drives :func:`superproton.rotation._unordered_pair_switches`,
:func:`superproton.rotation._reorientation_start_frames`, and
:func:`superproton.rotation.classify_events` against hand-built integer
sequences and synthetic :class:`superproton.tracking.ProtonTracks` instances.
No ovito or LAMMPS dump is required to run.
"""

from __future__ import annotations

import numpy as np
import pytest

from superproton.rotation import (
    RotationEvents,
    _reorientation_start_frames,
    _unordered_pair_switches,
    classify_events,
)

from _helpers import make_tracks


# ---------------------------------------------------------------------------
# _unordered_pair_switches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq, expected_switches, expected_pairs",
    [
        # One host all the way through: no second id, so the pair stays open.
        ([1, 1, 1, 1], [], [(1, -1)]),
        # Two hosts: the pair {1,2} forms and the proton rattles inside it.
        # Forming a pair is not a switch.
        ([1, 1, 2, 1, 2, 2, 1], [], [(1, 2)]),
        # A third id at frame 6 leaves the pair {1,2}: that is the switch.
        ([1, 1, 2, 2, 2, 2, 3], [6], [(1, 2), (3, -1)]),
        # Pair {1,2} forms at f2; f4 introduces 3 -> switch, and {3,1} forms at
        # f6; f8 introduces 4, outside {3,1} -> switch.
        ([1, 1, 2, 2, 3, 3, 1, 1, 4, 4], [4, 8], [(1, 2), (1, 3), (4, -1)]),
        # No frames at all.
        ([], [], []),
    ],
    ids=["single-host", "pair-forms", "one-switch", "two-switches", "empty"],
)
def test_unordered_pair_switches(seq, expected_switches, expected_pairs):
    """The switch detector over the canonical sequences from ``set_pairs``."""
    sw, pairs = _unordered_pair_switches(np.array(seq, dtype=int))
    assert sw == expected_switches
    assert pairs == expected_pairs


# ---------------------------------------------------------------------------
# _reorientation_start_frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq, switches, expected_t_s",
    [
        # The worked example in PO_indx_Hdist.py's docstring.
        ([1, 1, 2, 2, 2, 2, 3], [6], [2]),
        # At switch 4 the carrier is 2, last seen different at f1 -> t_s = 2.
        # At switch 8 the carrier is 1, last seen different at f5 -> t_s = 6.
        ([1, 1, 2, 2, 3, 3, 1, 1, 4, 4], [4, 8], [2, 6]),
        # Carrier already present at the window start, with no earlier
        # different id, so the period starts at the window start.
        ([2, 2, 2, 2, 3], [4], [0]),
    ],
    ids=["docstring-example", "two-switches", "carrier-at-window-start"],
)
def test_reorientation_start_frames(seq, switches, expected_t_s):
    """Backtracking a switch frame to the start of its rotation period."""
    ts = _reorientation_start_frames(np.array(seq, dtype=int), switches)
    assert ts.tolist() == expected_t_s


# ---------------------------------------------------------------------------
# classify_events: hand-built tracks
# ---------------------------------------------------------------------------




def test_classify_events_distinguishes_productive_from_nonproductive():
    # Canonical ``set_pairs`` treats first pair formation as NOT a switch;
    # the third distinct id is the first recorded switch. So we need a
    # trajectory long enough to have already formed both the O-pair and the
    # X-pair before the switches we assert.
    #
    # frame:  0  1  2  3  4  5  6  7  8
    # O id:  11 11 12 12 21 21 22 22 31
    # X id:   1  1  1  1  2  2  2  2  3
    #
    # O-pair {11,12} forms at f2. At f4 id 21 is outside that pair -> O-switch
    # but X-pair {1,2} is just forming there -> no P-switch -> nonproductive.
    # New O-pair {21,22} forms at f6. At f8 id 31 is outside {21,22} -> O-switch
    # and X id 3 is outside {1,2} -> P-switch -> productive.
    donor_O = np.array([[11, 11, 12, 12, 21, 21, 22, 22, 31]]).T
    donor_X = np.array([[ 1,  1,  1,  1,  2,  2,  2,  2,  3]]).T

    events = classify_events(make_tracks(donor_O, donor_X))
    assert events.o_pair_switch_frames[0].tolist() == [4, 8]
    assert events.p_pair_switch_frames[0].tolist() == [8]
    assert events.productive_t_s_frames[0].size == 1
    assert events.nonproductive_t_s_frames[0].size == 1


def test_classify_events_nonproductive_within_same_polyanion():
    # Trajectory restricted to a single polyanion: every O-pair switch is
    # nonproductive and there can be no P-pair switch.
    #
    # frame:  0  1  2  3  4  5  6  7  8  9 10 11 12
    # O id:  10 10 11 11 12 12 13 13 14 14 15 15 16
    # X id:   1  1  1  1  1  1  1  1  1  1  1  1  1
    #
    # O-pair {10,11} forms at f2. At f4 id 12 is outside -> SWITCH(f4),
    # new O-pair {12,13} forms at f6. At f8 id 14 is outside -> SWITCH(f8),
    # new pair {14,15} forms at f10. At f12 id 16 is outside -> SWITCH(f12).
    donor_O = np.array([[10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16]]).T
    donor_X = np.ones_like(donor_O)

    events = classify_events(make_tracks(donor_O, donor_X))
    assert events.o_pair_switch_frames[0].tolist() == [4, 8, 12]
    assert events.p_pair_switch_frames[0].tolist() == []
    assert events.productive_t_s_frames[0].size == 0
    assert events.nonproductive_t_s_frames[0].size == 3


def test_classify_events_multiple_protons():
    # frames 0-8 with 2 protons.
    # Proton 0: same pattern as ``distinguishes_productive_from_nonproductive``.
    # Proton 1: stays put on a single O the whole time -> no events.
    donor_O = np.array(
        [
            [11, 11, 12, 12, 21, 21, 22, 22, 31],
            [99, 99, 99, 99, 99, 99, 99, 99, 99],
        ]
    ).T
    donor_X = np.array(
        [
            [1, 1, 1, 1, 2, 2, 2, 2, 3],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
        ]
    ).T

    events = classify_events(make_tracks(donor_O, donor_X))
    assert events.n_protons == 2
    assert events.n_frames == 9
    assert events.o_pair_switch_frames[0].tolist() == [4, 8]
    assert events.p_pair_switch_frames[0].tolist() == [8]
    assert events.o_pair_switch_frames[1].size == 0
    assert events.p_pair_switch_frames[1].size == 0


def test_classify_events_rejects_non_tracks_input():
    with pytest.raises(TypeError):
        classify_events(np.zeros((4, 2)))


# ---------------------------------------------------------------------------
# Sanity: classify_events output plugs straight into residence.py
# ---------------------------------------------------------------------------


def test_classify_events_output_is_acceptable_to_residence():
    from superproton.residence import collect_taus, global_average_rate

    donor_O = np.array([[10, 10, 11, 11, 12, 12, 13, 13, 14, 14]]).T
    donor_X = np.array([[ 1,  1,  1,  1,  2,  2,  3,  3,  4,  4]]).T
    events = classify_events(make_tracks(donor_O, donor_X))
    # Should produce a usable RotationEvents -- just exercise the downstream.
    assert isinstance(events, RotationEvents)
    tau = collect_taus(events)
    # donor_X visits 1,1,2,3,4: the pair detector sees one switch (at frame 6),
    # and a single switch bounds no interval, so there are no P-pair residence
    # times. donor_O visits five ids, giving two switches and one O-pair interval.
    assert tau.tau_PP_ps.size == 0
    assert tau.tau_OO_ps.size == 1
    g = global_average_rate(events)
    assert "overall" in g
