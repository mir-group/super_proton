"""Unit tests for superproton.reorientation.

Builds synthetic H and O position trajectories with known displacement
ratios and verifies that :func:`compute_reorientation` recovers the
expected baseline and outlier metrics, plus the cos(dH, dO) directional
diagnostic. No ovito or LAMMPS dump is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from superproton.reorientation import (
    HOPositions,
    compute_reorientation,
)
from superproton.rotation import classify_events
from superproton.tracking import ProtonTracks


def _build_tracks_and_positions(
    *,
    donor_O_id: np.ndarray,
    donor_X_id: np.ndarray,
    H_id: int = 1,
    O_ids: tuple[int, ...] = (101, 102),
    H_disp_per_frame: float = 0.2,
    O_disp_per_frame: float = 0.1,
    spike_frame: int | None = None,
    spike_H_disp: float = 2.0,
):
    n_frames, n_protons = donor_O_id.shape
    assert n_protons == 1, "one-proton helper"
    sentinel = np.full_like(donor_O_id, -1)
    tracks = ProtonTracks(
        n_protons=1,
        n_frames=n_frames,
        frame_to_ps=0.05,
        frame_indices=np.arange(n_frames, dtype=int),
        proton_ids=np.array([H_id], dtype=np.int64),
        donor_O_id=donor_O_id,
        donor_X_id=donor_X_id,
        acceptor_O_id=sentinel,
        acceptor_X_id=sentinel,
    )
    # Build H and O positions: H moves +x by H_disp_per_frame each step.
    H_pos = np.zeros((n_frames, 1, 3), dtype=np.float64)
    H_pos[:, 0, 0] = np.arange(n_frames) * H_disp_per_frame
    if spike_frame is not None:
        # Make the H jump in frame ``spike_frame`` instead of the usual step.
        H_pos[spike_frame, 0, 0] = H_pos[spike_frame - 1, 0, 0] + spike_H_disp
        # Subsequent frames continue with the regular step from the new base.
        for fr in range(spike_frame + 1, n_frames):
            H_pos[fr, 0, 0] = H_pos[fr - 1, 0, 0] + H_disp_per_frame
    O_pos = np.zeros((n_frames, len(O_ids), 3), dtype=np.float64)
    # Each O moves +x by O_disp_per_frame each step.
    for o in range(len(O_ids)):
        O_pos[:, o, 0] = np.arange(n_frames) * O_disp_per_frame
    positions = HOPositions(
        H_pos=H_pos,
        O_pos=O_pos,
        H_ids=np.array([H_id], dtype=np.int64),
        O_ids=np.array(O_ids, dtype=np.int64),
        frame_indices=np.arange(n_frames, dtype=int),
    )
    return tracks, positions


def test_compute_reorientation_baseline_matches_lever_ratio():
    """Build a single nonproductive event whose H/O step ratio is exactly 2.0
    everywhere within the rotation period; baseline must come out at 2.0
    and the outlier metric should be NaN (no spikes)."""
    # Period length matters: with only 2 instantaneous ratios the Z-score
    # statistics are degenerate. Use 11 frames so the [t_s, t_f] interval
    # carries 6 ratios.
    donor_O = np.array([[101, 101, 102, 102, 102, 102, 102, 102, 103, 103, 103]]).T
    donor_X = np.ones_like(donor_O)
    tracks, positions = _build_tracks_and_positions(
        donor_O_id=donor_O,
        donor_X_id=donor_X,
        H_id=1,
        O_ids=(101, 102, 103),
        H_disp_per_frame=0.2,
        O_disp_per_frame=0.1,
    )
    events = classify_events(tracks)
    # One nonproductive switch at frame 8 (id 103 outside {101,102}).
    assert events.o_pair_switch_frames[0].tolist() == [8]
    assert events.p_pair_switch_frames[0].tolist() == []

    result = compute_reorientation(positions, tracks, events)
    assert result.kind.tolist() == ["nonproductive"]
    # carrier O at t_f = donor_O[8, 0] = 103.
    # t_s = 2 (last "different" donor in window [0:8] is at f1 with id 101).
    assert int(result.t_s[0]) == 2
    assert int(result.t_f[0]) == 8
    # Every per-frame ratio is exactly 0.2 / 0.1 = 2.0 -> baseline = 2.0.
    assert result.baseline[0] == pytest.approx(2.0)
    # No spikes -> outlier should be NaN.
    assert np.isnan(result.outlier[0])
    # cos(dH, dO): both displacements are along +x, so cos = +1.
    assert result.cos_HO[0] == pytest.approx(1.0)


def test_compute_reorientation_outlier_captures_spike():
    """Inject one big H displacement; the outlier metric must record the spike."""
    donor_O = np.array([[101, 101, 102, 102, 102, 102, 102, 102, 103, 103, 103]]).T
    donor_X = np.ones_like(donor_O)
    tracks, positions = _build_tracks_and_positions(
        donor_O_id=donor_O,
        donor_X_id=donor_X,
        H_id=1,
        O_ids=(101, 102, 103),
        H_disp_per_frame=0.2,
        O_disp_per_frame=0.1,
        spike_frame=4,
        spike_H_disp=2.0,  # H jumps 2.0 at frame 4 instead of 0.2
    )
    events = classify_events(tracks)
    result = compute_reorientation(positions, tracks, events)
    # Period frames 2..8, 6 ratios: [2.0, 2.0, 20.0, 2.0, 2.0, 2.0] (spike at f4).
    # mean = 30/6 = 5.0; std = sqrt(((2-5)^2 * 5 + (20-5)^2)/6) = sqrt(45) ~ 6.71.
    # z=2 threshold = 5 + 2*6.71 = 18.43 -> only the 20.0 ratio exceeds it.
    assert result.baseline[0] == pytest.approx(5.0)
    assert result.outlier[0] == pytest.approx(20.0)


def test_compute_reorientation_no_outliers_at_high_threshold():
    """Same data with z=10 threshold: nothing crosses, outlier should be NaN."""
    donor_O = np.array([[101, 101, 102, 102, 102, 102, 102, 102, 103, 103, 103]]).T
    donor_X = np.ones_like(donor_O)
    tracks, positions = _build_tracks_and_positions(
        donor_O_id=donor_O,
        donor_X_id=donor_X,
        H_id=1,
        O_ids=(101, 102, 103),
        H_disp_per_frame=0.2,
        O_disp_per_frame=0.1,
        spike_frame=4,
        spike_H_disp=2.0,
    )
    events = classify_events(tracks)
    result = compute_reorientation(positions, tracks, events, z_threshold=10.0)
    assert np.isnan(result.outlier[0])


def test_compute_reorientation_no_events():
    donor_O = np.array([[101, 101, 101, 101]]).T
    donor_X = np.array([[1, 1, 1, 1]]).T
    tracks, positions = _build_tracks_and_positions(
        donor_O_id=donor_O, donor_X_id=donor_X, O_ids=(101,)
    )
    events = classify_events(tracks)
    result = compute_reorientation(positions, tracks, events)
    assert result.proton_index.size == 0
    assert result.baseline.size == 0


def test_compute_reorientation_frame_indices_mismatch_rejected():
    donor_O = np.array([[101, 102, 103]]).T
    donor_X = np.array([[1, 1, 1]]).T
    tracks, positions = _build_tracks_and_positions(
        donor_O_id=donor_O, donor_X_id=donor_X, O_ids=(101, 102, 103)
    )
    # Build a positions object with mismatched frame_indices.
    bad_positions = HOPositions(
        H_pos=positions.H_pos,
        O_pos=positions.O_pos,
        H_ids=positions.H_ids,
        O_ids=positions.O_ids,
        frame_indices=positions.frame_indices + 100,
    )
    events = classify_events(tracks)
    with pytest.raises(ValueError):
        compute_reorientation(bad_positions, tracks, events)
