"""Unit tests for superproton.correlation.

Tests the pure-function pieces (pair distances, shell identification, the
core G_event histogram, and NP-P temporal overlap) without requiring ovito.
Additional production-trajectory checks are in ``tests/test_reference.py``.
"""

from __future__ import annotations

import csv
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from superproton.correlation import (
    GEventResult,
    PolyanionCenters,
    _compute_G_event,
    assign_shell_indices,
    collect_recipient_events,
    count_interval_overlaps,
    identify_shells,
    pp_distance_matrix,
    temporal_overlap,
)
from superproton.rotation import RotationEvents


# ---------------------------------------------------------------------------
# pp_distance_matrix
# ---------------------------------------------------------------------------


def test_pp_distance_matrix_orthorhombic():
    # Two atoms in a cubic 10 A box; distance = 3 A directly, or 7 A across PBC.
    centers = PolyanionCenters(
        positions=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        cell=10.0 * np.eye(3),
        polyanion_ids=np.array([1, 2], dtype=np.int64),
    )
    d = pp_distance_matrix(centers)
    assert d.shape == (2, 2)
    assert d[0, 0] == 0
    assert d[1, 1] == 0
    assert d[0, 1] == pytest.approx(3.0)
    assert d[1, 0] == pytest.approx(3.0)


def test_pp_distance_matrix_min_image_wraps():
    # Two atoms at x=0 and x=9 in a 10 A box -> the min image distance is 1 A.
    centers = PolyanionCenters(
        positions=np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]),
        cell=10.0 * np.eye(3),
        polyanion_ids=np.array([1, 2], dtype=np.int64),
    )
    d = pp_distance_matrix(centers)
    assert d[0, 1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# identify_shells
# ---------------------------------------------------------------------------


def test_identify_shells_basic():
    # 4 atoms in a square arrangement: distances ~ 1, 1, sqrt(2) ~ 1.41, 1, 1, sqrt(2).
    centers = PolyanionCenters(
        positions=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        ),
        cell=100.0 * np.eye(3),  # large enough that PBC doesn't matter
        polyanion_ids=np.array([1, 2, 3, 4], dtype=np.int64),
    )
    d = pp_distance_matrix(centers)
    shells = identify_shells(d, tol_A=0.3)
    # Expect two shells: one near 1.0, one near sqrt(2) ~ 1.41.
    assert shells.size == 2
    assert shells[0] == pytest.approx(1.0)
    assert shells[1] == pytest.approx(np.sqrt(2.0), rel=0.05)


def test_assign_shell_indices_excludes_self_and_far():
    centers = PolyanionCenters(
        positions=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [10.0, 0.0, 0.0]]
        ),
        cell=100.0 * np.eye(3),
        polyanion_ids=np.array([1, 2, 3, 4], dtype=np.int64),
    )
    d = pp_distance_matrix(centers)
    shells = np.array([1.0])
    idx = assign_shell_indices(d, shells, r_max_A=2.0)
    # Diagonal -> -1; pair (0, 3) at distance 10 -> -1; pair (1, 2) at sqrt(2) ~ 1.41 -> 0 if within tol of 1.0 but here r_max=2.0 so still assigned.
    assert np.all(np.diag(idx) == -1)
    assert idx[0, 3] == -1
    assert idx[3, 0] == -1
    # Close pairs map to shell 0.
    assert idx[0, 1] == 0
    assert idx[1, 0] == 0


# ---------------------------------------------------------------------------
# collect_recipient_events
# ---------------------------------------------------------------------------


def test_collect_recipient_events_handles_consecutive_switches():
    # One proton, two productive switches.
    # pair[0] = (1, 2), pair[1] = (2, 3), pair[2] = (3, 4)
    # switch 0 (at frame 5): pair (1,2) -> (2,3). recipient = 3.
    # switch 1 (at frame 12): pair (2,3) -> (3,4). recipient = 4.
    events = RotationEvents(
        n_protons=1,
        n_frames=20,
        frame_to_ps=0.05,
        o_pair_switch_frames=[np.array([5, 12], dtype=np.int64)],
        p_pair_switch_frames=[np.array([5, 12], dtype=np.int64)],
        productive_t_s_frames=[np.array([4, 11], dtype=np.int64)],
        nonproductive_t_s_frames=[np.empty(0, dtype=np.int64)],
        o_pair_ids=[np.array([[1, 2], [2, 3], [3, 4]], dtype=np.int64)],
        p_pair_ids=[np.array([[1, 2], [2, 3], [3, 4]], dtype=np.int64)],
    )
    times_ps, recipients = collect_recipient_events(events)
    assert times_ps.tolist() == pytest.approx([5 * 0.05, 12 * 0.05])
    assert recipients.tolist() == [3, 4]


# ---------------------------------------------------------------------------
# _compute_G_event
# ---------------------------------------------------------------------------


def test_compute_G_event_uniform_random_gives_G_near_one():
    """For a random Poisson process scattered uniformly across sites, G should
    average near 1.0 (independent events)."""
    rng = np.random.default_rng(0)
    n_P = 20
    polyanion_ids = np.arange(1, n_P + 1, dtype=np.int64)
    # Random spatial positions on a 10 A cube for the distance matrix.
    positions = rng.uniform(0, 10, size=(n_P, 3))
    centers = PolyanionCenters(
        positions=positions, cell=20.0 * np.eye(3), polyanion_ids=polyanion_ids
    )
    d_PP = pp_distance_matrix(centers)
    shells = identify_shells(d_PP, tol_A=0.5)
    shells = shells[shells <= 8.0]
    shell_idx = assign_shell_indices(d_PP, shells, r_max_A=8.0)
    # Random event stream: 1000 events uniformly random in time AND site.
    n_events = 1000
    times_ps = np.sort(rng.uniform(0, 1000, size=n_events))
    recipient_ids = polyanion_ids[rng.integers(0, n_P, size=n_events)]
    result = _compute_G_event(
        times_ps=times_ps,
        recipient_polyanion_ids=recipient_ids,
        polyanion_ids=polyanion_ids,
        shell_idx=shell_idx,
        shells=shells,
        dt_max_ps=50.0,
        dt_bin_ps=1.0,
        rate_per_ps=None,
        min_H_exp=1.0,
    )
    assert result.G.shape == (shells.size, 50)
    g_finite = result.G[np.isfinite(result.G)]
    # Independent events -> H_obs / H_exp ~ 1 on average, with noise.
    assert g_finite.mean() == pytest.approx(1.0, abs=0.1)


def test_compute_G_event_returns_empty_for_too_few_events():
    polyanion_ids = np.arange(1, 5, dtype=np.int64)
    centers = PolyanionCenters(
        positions=np.eye(3)[:3].repeat(2, axis=0)[:4],
        cell=10.0 * np.eye(3),
        polyanion_ids=polyanion_ids,
    )
    d = pp_distance_matrix(centers)
    shells = identify_shells(d)
    shell_idx = assign_shell_indices(d, shells, r_max_A=5.0)
    result = _compute_G_event(
        times_ps=np.array([1.0]),
        recipient_polyanion_ids=np.array([1]),
        polyanion_ids=polyanion_ids,
        shell_idx=shell_idx,
        shells=shells,
        dt_max_ps=10.0,
        dt_bin_ps=1.0,
        rate_per_ps=None,
        min_H_exp=0.0,
    )
    assert isinstance(result, GEventResult)
    assert np.all(np.isnan(result.G))


# NP-centred temporal overlap and CLI output.


def test_all_interval_relations():
    # NP contains P, P contains NP, identical, endpoint contact, and disjoint.
    starts = np.array([0, 4, 3, 0, 7, 0, 8])
    ends = np.array([10, 5, 7, 3, 9, 2, 9])
    np.testing.assert_array_equal(
        count_interval_overlaps(starts, ends, np.array([3]), np.array([7])),
        [1, 1, 1, 1, 1, 0, 0],
    )
    np.testing.assert_array_equal(count_interval_overlaps(starts, ends, [], []), 0)
    assert count_interval_overlaps([], [], [3], [7]).size == 0


def test_endpoint_search_matches_brute_force():
    rng = np.random.default_rng(18)
    query = np.sort(rng.integers(0, 100, (80, 2)), axis=1)
    reference = np.sort(rng.integers(0, 100, (70, 2)), axis=1)
    expected = ((query[:, 0, None] <= reference[:, 1])
                & (reference[:, 0] <= query[:, 1, None])).sum(axis=1)
    np.testing.assert_array_equal(
        count_interval_overlaps(*query.T, *reference.T), expected,
    )


@pytest.mark.parametrize("starts,ends", [([3], [2]), ([1], []), ([[1]], [[2]]),
                                         ([float("nan")], [2])])
def test_invalid_intervals(starts, ends):
    with pytest.raises(ValueError):
        count_interval_overlaps(starts, ends, [], [])


def _overlap_events():
    # Proton 0: NP on exchanger X20 / host X10, interval [2, 8].
    # Protons 1-4: four contained P events with different roles; the last is
    # on unrelated polyanions and must not count despite overlapping in time.
    o_pairs = [[[101, 201], [101, 202]],
               [[201, 301], [201, 401]], [[301, 401], [201, 401]],
               [[201, 301], [301, 401]], [[501, 601], [501, 701]]]
    p_pairs = [[[10, 20]], [[20, 30], [20, 40]], [[30, 40], [20, 40]],
               [[20, 30], [30, 40]], [[50, 60], [50, 70]]]
    return RotationEvents(
        n_protons=5, n_frames=12, frame_to_ps=0.05,
        o_pair_switch_frames=[np.array([8])] + [np.array([6]) for _ in range(4)],
        p_pair_switch_frames=[np.array([], dtype=int)] + [np.array([6]) for _ in range(4)],
        productive_t_s_frames=[np.array([], dtype=int)] + [np.array([4]) for _ in range(4)],
        nonproductive_t_s_frames=[np.array([2])] + [np.array([], dtype=int) for _ in range(4)],
        o_pair_ids=list(map(np.array, o_pairs)), p_pair_ids=list(map(np.array, p_pairs)),
    )


def test_roles_are_separate_and_require_same_polyanion():
    result = temporal_overlap(_overlap_events(), {101: 10, 201: 20, 202: 20})
    assert result.n_productive == 4
    assert result.proton_index.tolist() == [0]
    assert result.polyanion_ids["exchanger"].tolist() == [20]
    assert result.polyanion_ids["host"].tolist() == [10]
    for (np_role, _), counts in result.counts.items():
        assert counts.tolist() == [int(np_role == "exchanger")]
    for row in result.summary():
        expected = "1" if row["NP_role"] == "exchanger" else "0"
        assert row["percent"] == (100 if row["overlap_count"] == expected else 0)


def test_summary_collects_entire_three_or_more_tail():
    result = temporal_overlap(_overlap_events(), {101: 10, 201: 20, 202: 20})
    result = replace(result, counts={("host", "carrier"): np.array([0, 1, 2, 3, 4, 9])},
                     proton_index=np.arange(6))
    rows = result.summary()
    assert [row["n_events"] for row in rows] == [1, 1, 1, 3]
    assert sum(row["percent"] for row in rows) == pytest.approx(100)


def test_invalid_tail_is_skipped_and_missing_mapping_is_an_error():
    events = _overlap_events()
    with pytest.raises(KeyError):
        temporal_overlap(events, {101: 10})
    # Incomplete NP tail and a productive transition with no shared X.
    events.o_pair_ids[0][-1] = [101, -1]
    events.p_pair_ids[1][-1] = [50, 60]
    result = temporal_overlap(events, {})
    assert result.n_skipped_nonproductive == 1
    assert result.n_skipped_productive == 1
    assert result.n_productive == 3
    assert all(row["n_events"] == 0 and np.isnan(row["percent"])
               for row in result.summary())


def test_cli_writes_summary_and_per_event_counts(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_analysis.py"
    spec = importlib.util.spec_from_file_location("run_analysis_overlap_test", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "_load_tracks_and_events", lambda *_: (None, _overlap_events()))
    monkeypatch.setattr(cli, "parent_polyanions", lambda *a, **k: {101: 10, 201: 20, 202: 20})
    args = SimpleNamespace(material="CDP", temperature=525, start_frame=0, out=tmp_path)
    got = cli.run_overlap(None, args)
    with (tmp_path / "CDP_525K_overlap.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 24
    assert rows[0]["n_np_events"] == "1"
    with (tmp_path / "CDP_525K_overlap_events.csv").open() as fh:
        event_rows = list(csv.DictReader(fh))
    assert len(event_rows) == 1
    assert event_rows[0]["exchanger_carrier"] == "1"
    assert got["n_nonproductive"] == 1
    assert cli.RUNNERS["overlap"] is cli.run_overlap
    assert cli._published("CHS", 450)["overlap"]["exchanger_carrier_no_overlap_pct"] == 80.14
