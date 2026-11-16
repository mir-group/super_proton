"""Unit tests for superproton.osharing.

Builds hand-crafted ProtonTracks (with real acceptor data, not -1 sentinels)
and verifies the sharing mask and the per-event fraction-vs-time-to-
completion curve.
"""

from __future__ import annotations

import numpy as np
import pytest

from superproton.osharing import (
    OSharingData,
    o_sharing_per_frame,
    sharing_fraction_vs_t_to_completion,
)
from superproton.rotation import classify_events

from _helpers import make_tracks




def test_o_sharing_per_frame_basic_detection():
    # 3 frames, 2 protons.
    # Proton 0 has donor 101, acceptor 102. Proton 1 has donor 102, acceptor 103.
    # Oxygen 102 is claimed by both protons, so under the symmetric definition
    # BOTH are O-sharing: proton 0 holds it as its acceptor and proton 1 as its
    # donor. (The earlier one-sided predicate flagged only proton 0; that is the
    # manuscript's prose, not the definition behind Figure 6. See the module
    # docstring.)
    donor_O = np.array([[101, 102], [101, 102], [101, 102]])
    donor_X = np.array([[1, 2], [1, 2], [1, 2]])
    acceptor_O = np.array([[102, 103], [102, 103], [102, 103]])
    acceptor_X = np.array([[2, 3], [2, 3], [2, 3]])
    tracks = make_tracks(donor_O, donor_X, acceptor_O, acceptor_X)
    sharing = o_sharing_per_frame(tracks)
    assert sharing.n_frames == 3 and sharing.n_protons == 2
    np.testing.assert_array_equal(sharing.is_sharing[:, 0], [True, True, True])
    np.testing.assert_array_equal(sharing.is_sharing[:, 1], [True, True, True])
    assert sharing.mean_fraction == pytest.approx(1.0)


def test_o_sharing_per_frame_no_sharing():
    # Every proton's acceptor is a unique O that no one is covalently bonded to.
    donor_O = np.array([[101, 201]])
    donor_X = np.array([[1, 2]])
    acceptor_O = np.array([[301, 401]])
    acceptor_X = np.array([[3, 4]])
    sharing = o_sharing_per_frame(make_tracks(donor_O, donor_X, acceptor_O, acceptor_X))
    assert not sharing.is_sharing.any()
    assert sharing.mean_fraction == 0.0


def test_o_sharing_rejects_sentinel_acceptor():
    donor_O = np.array([[101, 102]])
    donor_X = np.array([[1, 2]])
    sentinel = np.full_like(donor_O, -1)
    with pytest.raises(ValueError):
        o_sharing_per_frame(
            make_tracks(donor_O, donor_X, sentinel, sentinel)
        )


def test_o_sharing_skips_self_match():
    # One proton whose two slots hold the same oxygen. That cannot come out of
    # proton_neighbors (nearest and second-nearest are distinct atoms), but a
    # single proton must never count as sharing an oxygen with itself.
    donor_O = np.array([[101]])
    donor_X = np.array([[1]])
    acceptor_O = np.array([[101]])
    acceptor_X = np.array([[1]])
    sharing = o_sharing_per_frame(make_tracks(donor_O, donor_X, acceptor_O, acceptor_X))
    assert not sharing.is_sharing.any()


def test_sharing_fraction_vs_t_to_completion_shapes():
    """The fraction curves should span [-window_ps, 0] with correct length."""
    # Build a trajectory that has at least one nonproductive event for proton 0.
    # 11 frames, single polyanion -> all O-pair switches are nonproductive.
    donor_O = np.array([[101, 101, 102, 102, 102, 102, 102, 102, 103, 103, 103]]).T
    donor_X = np.ones_like(donor_O)
    # Acceptor is a fixed O that doesn't match any donor -> sharing is always False.
    acceptor_O = np.full_like(donor_O, 999)
    acceptor_X = np.full_like(donor_O, 9)
    tracks = make_tracks(donor_O, donor_X, acceptor_O, acceptor_X)
    events = classify_events(tracks)
    # But we still want sharing-fraction to have the right shape. Construct
    # a manual sharing mask (so the test doesn't depend on the dummy
    # acceptor data triggering the "all -1" sentinel guard).
    sharing = OSharingData(
        n_frames=donor_O.shape[0],
        n_protons=1,
        is_sharing=np.zeros_like(donor_O, dtype=bool),
    )

    curves = sharing_fraction_vs_t_to_completion(sharing, events, window_ps=0.5)
    for kind in ("productive", "nonproductive"):
        offsets, frac = curves[kind]
        assert offsets[-1] == 0.0
        assert offsets[0] == pytest.approx(-0.5)
        assert offsets.shape == frac.shape
        # At least the 0-offset point exists (window=0.5 ps -> 11 offsets at 50 fs stride).
        assert offsets.size == 11


def test_sharing_fraction_recovers_known_pattern():
    """Plant active sharing in the 3 frames preceding one nonproductive event
    and verify the fraction curve sees 1.0 at those offsets and 0 elsewhere."""
    donor_O = np.array([[101, 101, 102, 102, 102, 102, 102, 102, 103, 103, 103]]).T
    donor_X = np.ones_like(donor_O)
    # Switch is at frame 8. Plant True at frames 5, 6, 7.
    is_sharing = np.zeros_like(donor_O, dtype=bool)
    is_sharing[5:8, 0] = True
    # Counter threshold is 5 in the implementation, so use 1 proton with 1
    # event -> the counter never reaches 5, fractions will all be NaN.
    # Bring the threshold down by stacking the same event many times via
    # multiple protons sharing the same trajectory.
    n_protons = 10
    donor_O_multi = np.tile(donor_O, (1, n_protons))
    donor_X_multi = np.tile(donor_X, (1, n_protons))
    acc_multi = np.full_like(donor_O_multi, 999)
    accx_multi = np.full_like(donor_O_multi, 9)
    is_sharing_multi = np.tile(is_sharing, (1, n_protons))
    tracks_multi = make_tracks(donor_O_multi, donor_X_multi, acc_multi, accx_multi)
    events_multi = classify_events(tracks_multi)
    sharing_multi = OSharingData(
        n_frames=donor_O_multi.shape[0], n_protons=n_protons, is_sharing=is_sharing_multi
    )

    curves = sharing_fraction_vs_t_to_completion(sharing_multi, events_multi, window_ps=0.5)
    offsets, frac = curves["nonproductive"]
    # Offsets at -0.05, -0.1, -0.15 ps correspond to frames 7, 6, 5 before
    # completion at frame 8 (dt = 0.05 ps).
    # Build a frame-offset lookup: offset_idx in curve == window_frames + 1 -
    # (sw - actual_frame) -> just check the last-N entries manually.
    # The is_sharing entries that were True line up with frame offsets
    # of -3, -2, -1 (relative to the switch at frame 8 -> frames 5, 6, 7).
    # In the returned offsets array these are positions -4, -3, -2 (negative
    # indexing) given offsets ends at 0.
    assert frac[-4] == pytest.approx(1.0)
    assert frac[-3] == pytest.approx(1.0)
    assert frac[-2] == pytest.approx(1.0)
    # Frame at offset 0 = the switch frame itself: is_sharing[8] is False.
    assert frac[-1] == pytest.approx(0.0)
