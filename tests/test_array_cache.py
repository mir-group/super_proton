"""Cache hits preserve exact arrays and never reuse stale or damaged data."""

import json
import os

import numpy as np
import pytest

from superproton.io import cached_arrays


def test_cache_hit_preserves_arrays_and_canonical_options(tmp_path):
    source = tmp_path / "source.dump"
    source.write_text("trajectory")
    directory = tmp_path / "cache"
    expected = {
        "ids": np.array([2**60, -1], dtype=np.int64),
        "values": np.array([[np.nan, np.inf, -0.0, np.nextafter(1.0, 2.0)]]),
        "empty": np.empty((0, 3), dtype=np.float32),
        "flags": np.array([True, False]),
    }
    calls = []

    def compute():
        calls.append(True)
        return expected

    first = cached_arrays(directory, source=source, parameters={"version": 1, "stride": 1},
                          compute=compute)
    second = cached_arrays(directory, source=source, parameters={"stride": 1, "version": 1},
                           compute=compute)
    assert len(calls) == 1
    for key, value in expected.items():
        assert first[key].dtype == second[key].dtype == value.dtype
        assert first[key].shape == second[key].shape == value.shape
        assert first[key].tobytes() == second[key].tobytes() == value.tobytes()
    with np.load(next(directory.glob("*.npz")), allow_pickle=False) as archive:
        metadata = json.loads(archive["__metadata__"].item())
        assert metadata["source"]["path"] == str(source.resolve())
        assert metadata["source"]["ctime_ns"] == source.stat().st_ctime_ns


def test_cache_invalidates_for_options_source_path_and_source_changes(tmp_path):
    source = tmp_path / "source.dump"
    source.write_text("one")
    other = tmp_path / "other.dump"
    other.write_text("one")
    calls = []

    def compute():
        calls.append(True)
        return {"count": np.array(len(calls))}

    def cached(path=source, version=1):
        return cached_arrays(tmp_path / "cache", source=path, parameters={"version": version},
                             compute=compute)["count"].item()

    assert cached() == cached() == 1
    assert cached(version=2) == 2
    assert cached(other) == 3
    old_stat = source.stat()
    source.write_text("two")  # Same byte count; timestamp must still invalidate.
    os.utime(source, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 10_000_000))
    assert cached() == 4
    source.write_text("longer")
    assert cached() == 5


@pytest.mark.parametrize("damage", ["corrupt", "metadata", "object"])
def test_invalid_cache_warns_and_recovers(tmp_path, damage):
    source = tmp_path / "source"
    source.touch()
    directory = tmp_path / "cache"
    calls = []

    def compute():
        calls.append(True)
        return {"x": np.array([1, 2, 3])}

    def cached():
        return cached_arrays(directory, source=source, parameters={}, compute=compute)

    cached()
    target = next(directory.glob("*.npz"))
    with np.load(target, allow_pickle=False) as archive:
        metadata = archive["__metadata__"].copy()
    if damage == "corrupt":
        target.write_bytes(b"not an NPZ archive")
    elif damage == "metadata":
        np.savez_compressed(target, __metadata__=np.array("{}"), x=np.array([999]))
    else:
        np.savez_compressed(target, __metadata__=metadata, x=np.array([{}], dtype=object))
    with pytest.warns(RuntimeWarning, match="Ignoring invalid array cache"):
        np.testing.assert_array_equal(cached()["x"], [1, 2, 3])
    cached()
    assert len(calls) == 2


def test_source_changes_during_compute_are_not_cached(tmp_path):
    source = tmp_path / "source"
    source.write_text("before")

    def compute():
        source.write_text("changed during read")
        return {"x": np.array([1])}

    with pytest.raises(RuntimeError, match="Source changed during calculation"):
        cached_arrays(tmp_path / "cache", source=source, parameters={}, compute=compute)
    assert not list((tmp_path / "cache").iterdir())


@pytest.mark.parametrize("value", [[1, 2], np.array([{}], dtype=object)])
def test_computed_values_must_be_arrays_without_pickle(tmp_path, value):
    source = tmp_path / "source"
    source.touch()
    with pytest.raises(ValueError, match="NumPy arrays without object dtype"):
        cached_arrays(tmp_path / "cache", source=source, parameters={},
                      compute=lambda: {"x": value})
    assert not list((tmp_path / "cache").iterdir())


def test_disabled_cache_calls_compute_without_source_access(tmp_path):
    expected = {"x": np.array([1])}
    assert cached_arrays(None, source=tmp_path / "missing", parameters={},
                         compute=lambda: expected) is expected


def test_permission_error_is_not_swallowed(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.touch()
    compute = lambda: {"x": np.array([1])}
    cached_arrays(tmp_path / "cache", source=source, parameters={}, compute=compute)

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(np, "load", denied)
    with pytest.raises(PermissionError, match="denied"):
        cached_arrays(tmp_path / "cache", source=source, parameters={}, compute=compute)
