"""Shared frame reads must retain strict coordination and interval semantics."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superproton import coordination, io, tracking
from superproton.workflow import Analysis


@pytest.mark.parametrize("material", ["CDP", "CHS"])
def test_shared_tracks_match_separate_counts_and_residence(material, tmp_path, monkeypatch):
    filename = "cdp_525K_dump_1.trj" if material == "CDP" else "chs_450K_dump_1.trj"
    source = Path(__file__).parents[1] / "data" / filename
    if not source.exists():
        pytest.skip("Bundled trajectory unavailable")
    kwargs = {"material": material, "frame_to_ps": .05, "start_frame": 7,
              "end_frame": 1007, "stride": 2}
    separate = Analysis(source, **kwargs)
    expected = separate.coordination
    counts = coordination.mutual_bound_protons(
        separate.pipeline, material, separate.tracks.frame_indices
    )
    shared = Analysis(source, cache_dir=tmp_path / "cache", collect_coordination=True, **kwargs)
    actual_tracks = shared.tracks
    for name in ("donor_O_id", "donor_X_id", "acceptor_O_id", "acceptor_X_id",
                 "frame_indices", "proton_ids"):
        np.testing.assert_array_equal(getattr(actual_tracks, name), getattr(separate.tracks, name))
    np.testing.assert_array_equal(actual_tracks.bound_polyanion_ids, counts.polyanion_ids)
    np.testing.assert_array_equal(actual_tracks.mutual_bound_counts, counts.bound_per_frame)
    assert actual_tracks.mutual_bound_counts.dtype == np.uint8
    np.testing.assert_array_equal(actual_tracks.frame_indices, np.arange(7, 1007, 2))
    assert actual_tracks.frame_to_ps == .1
    assert len(expected) > 0  # Exercise actual completed intervals, not just empty tables.

    def forbidden(*args, **kwargs):
        raise AssertionError("Coordination should reuse persisted per-frame counts")

    # Only tracks were cached; constructing coordination must work without a pipeline.
    monkeypatch.setattr(io, "validated_pipeline", forbidden)
    monkeypatch.setattr(coordination, "mutual_bound_protons", forbidden)
    restored = Analysis(source, cache_dir=tmp_path / "cache", collect_coordination=True, **kwargs)
    pd.testing.assert_frame_equal(restored.coordination, expected, check_exact=True)
    assert "pipeline" not in restored.__dict__
    pd.testing.assert_frame_equal(shared.coordination, expected, check_exact=True)


def test_strict_mutual_counts_do_not_count_nearest_oxygen_alone():
    from ovito.data import DataCollection, ParticleType

    data = DataCollection()
    particles = data.create_particles(count=8)
    particles.create_property("Particle Identifier", data=[11, 21, 22, 12, 31, 23, 13, 24])
    types = particles.create_property("Particle Type", data=[4, 2, 2, 4, 1, 2, 4, 2])
    for number in (1, 2, 4):
        types.types.append(ParticleType(id=number))
    particles.create_property("Position", data=[
        [0, 0, 0], [.9, 0, 0], [1.1, 0, 0],
        [3, 0, 0], [3.2, 0, 0], [3.7, 0, 0],
        [7, 0, 0], [7.9, 0, 0],
    ])
    data.create_cell(np.column_stack([np.eye(3) * 20, np.zeros(3)]), pbc=(True, True, True))
    # O11's closest H prefers another H; O12's closest atom is a spectator.
    # Only O13-H24 is mutual, even though all H atoms have a nearest oxygen.
    actual = coordination._mutual_bound_counts(
        data, {11: 101, 12: 101, 13: 102}, np.array([101, 102]),
        hydrogen_type=2, oxygen_type=4,
    )
    np.testing.assert_array_equal(actual, [0, 1])


def test_selected_proton_still_collects_whole_cell_counts():
    source = Path(__file__).parents[1] / "data/cdp_525K_dump_1.trj"
    if not source.exists():
        pytest.skip("Bundled trajectory unavailable")
    pipeline = io.validated_pipeline(source)
    full = tracking.proton_neighbors(pipeline, material="CDP", end_frame=8,
                                     collect_coordination=True, show_progress=False)
    selected = tracking.proton_neighbors(
        pipeline, material="CDP", end_frame=8, select_proton_ids=[268],
        collect_coordination=True, show_progress=False,
    )
    assert selected.n_protons == 1
    np.testing.assert_array_equal(selected.bound_polyanion_ids, full.bound_polyanion_ids)
    np.testing.assert_array_equal(selected.mutual_bound_counts, full.mutual_bound_counts)


@pytest.mark.parametrize("material", ["CDP", "CHS"])
def test_prefix_reuses_frames_and_recensors_intervals(material, monkeypatch):
    filename = "cdp_525K_dump_1.trj" if material == "CDP" else "chs_450K_dump_1.trj"
    source = Path(__file__).parents[1] / "data" / filename
    if not source.exists():
        pytest.skip("Bundled trajectory unavailable")
    kwargs = {"material": material, "frame_to_ps": .05, "start_frame": 7, "stride": 2,
              "collect_coordination": True}
    parent = Analysis(source, end_frame=1007, **kwargs)
    _ = parent.residence  # Both cached events and residence must be recomputed for a prefix.
    independent = Analysis(source, end_frame=512, **kwargs)
    expected_coordination = independent.coordination
    expected_residence = independent.residence

    def forbidden(*args, **kwargs):
        raise AssertionError("Creating a prefix must not open or read the trajectory")

    monkeypatch.setattr(io, "validated_pipeline", forbidden)
    monkeypatch.setattr(coordination, "mutual_bound_protons", forbidden)
    child = parent.prefix(512)
    assert child is not parent and child.end_frame == 512
    assert child.events is not parent.events
    assert "pipeline" not in child.__dict__
    assert np.shares_memory(child.tracks.donor_O_id, parent.tracks.donor_O_id)
    assert np.shares_memory(child.tracks.mutual_bound_counts, parent.tracks.mutual_bound_counts)
    np.testing.assert_array_equal(child.tracks.frame_indices, np.arange(7, 512, 2))
    for name in vars(expected_residence):
        np.testing.assert_array_equal(getattr(child.residence, name), getattr(expected_residence, name))
    pd.testing.assert_frame_equal(child.coordination, expected_coordination, check_exact=True)
    assert parent.prefix(1007) is parent
    with pytest.raises(ValueError, match="cannot extend"):
        child.prefix(513)
    for invalid in (7, -1, 12.5):
        with pytest.raises(ValueError, match="Prefix end"):
            parent.prefix(invalid)
    changed = parent.prefix(512)
    changed.end_frame = 511
    with pytest.raises(RuntimeError, match="Analysis settings changed"):
        _ = changed.coordination


def test_unbounded_parent_rejects_prefix_beyond_loaded_frames(monkeypatch):
    source = Path(__file__).parents[1] / "data/cdp_525K_dump_1.trj"
    if not source.exists():
        pytest.skip("Bundled trajectory unavailable")
    parent = Analysis(source, material="CDP", frame_to_ps=.05, start_frame=7, stride=5)
    tracks = parent.tracks

    class NoMoreReads:
        def compute(self, *args, **kwargs):
            raise AssertionError("Prefix validation must use loaded frame indices")

    monkeypatch.setitem(parent.__dict__, "pipeline", NoMoreReads())
    with pytest.raises(ValueError, match="available sampled frames"):
        parent.prefix(int(tracks.frame_indices[-1]) + parent.stride + 1)
    with pytest.raises(ValueError, match="available sampled frames"):
        parent.prefix(10**9)
