"""Caches must avoid trajectory work without changing the scientific outputs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superproton import io, pointing
from superproton.workflow import Analysis


@pytest.fixture
def trajectory(tmp_path):
    source = Path(__file__).parents[1] / "data/cdp_525K_dump_1.trj"
    if not source.exists():
        pytest.skip("Bundled trajectory unavailable")
    lines = []
    with source.open() as stream:
        for _ in range(40):
            header = [next(stream) for _ in range(9)]
            lines.extend(header)
            lines.extend(next(stream) for _ in range(int(header[3])))
    path = tmp_path / "short.trj"
    path.write_text("".join(lines))
    return path


def test_pointing_metadata_without_tracking_and_optional_lookahead(trajectory, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("S18 should not compute tracks or S20")

    monkeypatch.setattr(Analysis, "tracks", property(forbidden))
    analysis = Analysis(trajectory, material="CDP", frame_to_ps=.05,
                        start_frame=3, end_frame=38, stride=2)
    legacy_lookahead = pointing.episode_crossing_lookahead
    monkeypatch.setattr(pointing, "episode_crossing_lookahead", forbidden)
    tables = analysis.pointing(lookahead_ps=None)
    assert set(tables) == {"attempts", "summary"}
    # The bundled CDP dump uses persistent H IDs 251..500, not columns 0..249.
    assert 268 in set(tables["attempts"].proton_id)
    assert set(tables["attempts"].proton_id) <= set(range(251, 501))
    assert ((tables["attempts"].start_frame - 3) % 2 == 0).all()
    np.testing.assert_allclose(tables["attempts"].start_ps,
                               tables["attempts"].start_frame * .05)
    monkeypatch.setattr(pointing, "read_pointing_hosts", forbidden)
    again = analysis.pointing(lookahead_ps=None)
    pd.testing.assert_frame_equal(tables["attempts"], again["attempts"])
    tables["summary"].loc[:, "count"] = -999
    assert (analysis.pointing(lookahead_ps=None)["summary"]["count"] >= 0).all()
    # NumPy scalar options remain accepted and use the same memo entry.
    pd.testing.assert_frame_equal(again["attempts"], analysis.pointing(
        d_form=np.float64(1.8), d_break=np.float64(2.4), lookahead_ps=None
    )["attempts"])
    monkeypatch.setattr(pointing, "episode_crossing_lookahead", legacy_lookahead)


def test_cache_survives_new_analysis_and_lookahead_choice(trajectory, tmp_path, monkeypatch):
    kwargs = {"material": "CDP", "frame_to_ps": .05, "cache_dir": tmp_path / "cache"}
    original = Analysis(trajectory, **kwargs)
    outputs = original.pointing()
    tracks = original.tracks
    coordination = original.coordination

    def forbidden(*args, **kwargs):
        raise AssertionError("A valid disk cache should not open the trajectory")

    monkeypatch.setattr(io, "validated_pipeline", forbidden)
    restored = Analysis(trajectory, **kwargs)
    for key, table in outputs.items():
        pd.testing.assert_frame_equal(table, restored.pointing()[key])
    pd.testing.assert_frame_equal(outputs["attempts"],
                                  restored.pointing(lookahead_ps=None)["attempts"])
    pd.testing.assert_frame_equal(coordination, restored.coordination)
    for name, value in vars(tracks).items():
        np.testing.assert_equal(value, getattr(restored.tracks, name))
    with pytest.raises(AssertionError, match="open the trajectory"):
        restored.pointing(d_form=1.7)
    with pytest.raises(AssertionError, match="open the trajectory"):
        Analysis(trajectory, **kwargs, start_frame=1).pointing()


@pytest.mark.parametrize("change", ["source", "window"])
def test_changed_inputs_cannot_cache_stale_dependencies(trajectory, tmp_path, change):
    analysis = Analysis(trajectory, material="CDP", frame_to_ps=.05, cache_dir=tmp_path / "cache")
    _ = analysis.tracks
    before = set(analysis.cache_dir.glob("*.npz"))
    if change == "source":
        trajectory.touch()
    else:
        analysis.start_frame = 1
    with pytest.raises(RuntimeError, match="create a new Analysis"):
        _ = analysis.coordination
    assert set(analysis.cache_dir.glob("*.npz")) == before


def test_numpy_float_thresholds(trajectory):
    analysis = Analysis(trajectory, material="CDP", frame_to_ps=.05)
    result = analysis.pointing(d_form=np.float32(1.8), d_break=np.float32(2.4), lookahead_ps=None)
    assert set(result) == {"attempts", "summary"}
