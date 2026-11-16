"""Hand-specified cases for the reusable notebook interfaces."""

import numpy as np
import pytest
from _helpers import make_tracks

from superproton.io import AtomMapping
from superproton.pointing import pointing_outcomes, pointing_summary
from superproton.reorientation import _increment_metrics
from superproton.rotation import classify_events, proton_motion_table


def test_pair_history_distinguishes_rattle_and_third_identity():
    # A/B donor swaps are rattling; C is a new oxygen on the same host B.
    # D belongs to a third host, so only that last transition is productive.
    oxy = np.array([11, 12, 11, 13, 12, 14, 13])[:, None]
    host = np.array([101, 102, 101, 102, 102, 103, 102])[:, None]
    tracks = make_tracks(oxy, host)
    table = proton_motion_table(tracks, classify_events(tracks))
    assert table.kind.tolist() == [
        "rattling",
        "rattling",
        "nonproductive",
        "rattling",
        "productive",
        "rattling",
    ]


def test_pointing_delayed_completion_reversion_redirection_and_censoring():
    donor = np.array([1, 1, 1, 1, 3, 3, 3, 3, 3, 3])[:, None]
    acceptor = np.array([2, 2, 3, 3, 1, 4, 4, 1, 5, 5])[:, None]
    # {1,2} redirects to {1,3}; that attempt completes at f4.
    # {3,4} reverts to {1,3} at f7; {3,5} is unresolved at the end.
    out = pointing_outcomes(
        donor, acceptor, frame_to_ps=0.5, proton_ids=[77], source_frames=np.arange(100, 120, 2)
    )
    row = out.loc[out.start_local == 2].iloc[0]
    assert row.outcome == "committed" and row.delay_ps == 1
    assert row.proton_id == 77 and row.start_frame == 104 and row.outcome_frame == 108
    assert out.loc[out.start_local == 0, "outcome"].item() == "superseded"
    assert out.loc[out.start_local == 5, "outcome"].item() == "reverted"
    assert not out.loc[out.start_local == 7, "fresh"].item()
    assert out.iloc[-1].outcome == "censored" and np.isnan(out.iloc[-1].delay_ps)
    summary = pointing_summary(out)
    assert summary.fraction_decided.sum() == pytest.approx(1)


def test_rattling_swap_does_not_start_pointing_episode():
    donor = np.array([1, 2, 1, 2, 1])[:, None]
    acceptor = np.array([2, 1, 2, 1, 2])[:, None]
    out = pointing_outcomes(donor, acceptor)
    assert len(out) == 1
    assert out.iloc[0].outcome_local == 1


def test_pointing_no_acceptor_gap_preserves_episode():
    donor = np.ones((4, 1), dtype=int)
    acceptor = np.array([2, -1, -1, 2])[:, None]
    out = pointing_outcomes(donor, acceptor)
    assert len(out) == 1 and out.iloc[0].end_local == 4


def test_increment_geometry_has_known_ratio_and_direction():
    # H moves twice as far, first along +x then along -x; O always +x.
    h = np.array([[0.0, 0, 0], [2, 0, 0], [0, 0, 0], [0, 0, 0]])
    o = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 0]])
    ratio, cosine = _increment_metrics(h, o)
    np.testing.assert_allclose(ratio[:2], [2, 2])
    np.testing.assert_allclose(cosine[:2], [1, -1])
    assert np.isnan(ratio[2]) and np.isnan(cosine[2])


def test_mapping_rejects_overlapping_selections():
    with pytest.raises(ValueError):
        AtomMapping(2, 2, 3)


def test_custom_mapping_and_shuffled_particle_rows(tmp_path):
    from pathlib import Path

    from superproton.tracking import parent_polyanions
    from superproton.workflow import Analysis

    source = Path(__file__).parents[1] / "data/cdp_525K_dump_1.trj"
    if not source.exists():
        pytest.skip("Bundled trajectory unavailable")
    frames = []
    with source.open() as stream:
        for _ in range(12):
            header = [next(stream) for _ in range(9)]
            count = int(header[3])
            rows = [next(stream).split() for _ in range(count)]
            frames.append((header, rows))
    short = tmp_path / "short.trj"
    short.write_text(
        "".join("".join(h) + "".join(" ".join(r) + "\n" for r in rows) for h, rows in frames)
    )
    base = Analysis(short, material="CDP", frame_to_ps=0.05)
    parents = parent_polyanions(base.pipeline, "CDP")
    convert = lambda v: int(v) * 1000000 + 7
    type_map = {1: 5, 2: 17, 3: 101, 4: 29}
    transformed = []
    for header, rows in frames:
        changed = [[str(convert(r[0])), str(type_map[int(r[1])]), *r[2:]] for r in reversed(rows)]
        transformed.append("".join(header) + "".join(" ".join(r) + "\n" for r in changed))
    other = tmp_path / "remapped.trj"
    other.write_text("".join(transformed))
    mapping = AtomMapping(17, 29, 101, {convert(o): convert(x) for o, x in parents.items()})
    custom = Analysis(other, material=mapping, frame_to_ps=0.05)
    np.testing.assert_array_equal(custom.tracks.donor_O_id, base.tracks.donor_O_id * 1000000 + 7)
    np.testing.assert_array_equal(custom.tracks.donor_X_id, base.tracks.donor_X_id * 1000000 + 7)
    from superproton.pointing import read_pointing_hosts

    d, a = read_pointing_hosts(custom.pipeline, mapping, start_frame=0)
    expected_d, expected_a = read_pointing_hosts(base.pipeline, "CDP", start_frame=0)
    np.testing.assert_array_equal(d, expected_d * 1000000 + 7)
    np.testing.assert_array_equal(a, np.where(expected_a < 0, -1, expected_a * 1000000 + 7))
    from superproton.geometry import read_xo_unit_vectors

    v, _ = read_xo_unit_vectors(base.pipeline, "CDP")
    w, _ = read_xo_unit_vectors(custom.pipeline, mapping)
    np.testing.assert_allclose(v, w, atol=1e-12)

    from superproton.tracking import candidate_trace, proton_neighbors

    # A selected ID must retrieve the same history after row/type/ID remapping.
    chosen = int(base.tracks.proton_ids[3])
    subset = proton_neighbors(
        custom.pipeline,
        material=mapping,
        select_proton_ids=[convert(chosen)],
        show_progress=False,
    )
    assert subset.proton_ids.tolist() == [convert(chosen)]
    np.testing.assert_array_equal(subset.donor_O_id[:, 0], custom.tracks.donor_O_id[:, 3])
    for invalid in ([], [convert(chosen)] * 2, [999999999]):
        with pytest.raises(ValueError, match="unique IDs"):
            proton_neighbors(custom.pipeline, material=mapping, select_proton_ids=invalid)
    original_modifiers = list(custom.pipeline.modifiers)
    a = candidate_trace(base.pipeline, material="CDP", proton_id=chosen)
    b = candidate_trace(custom.pipeline, material=mapping, proton_id=convert(chosen))
    np.testing.assert_array_equal(a.track.O_nearest_neighbor, b.track.O_nearest_neighbor)
    np.testing.assert_array_equal(a.track.X_nearest_neighbor, b.track.X_nearest_neighbor)
    np.testing.assert_allclose(a.track.H_displacement_A, b.track.H_displacement_A, atol=1e-12)
    np.testing.assert_array_equal(a.labels.atom_id * 1000000 + 7, b.labels.atom_id)
    assert list(custom.pipeline.modifiers) == original_modifiers


def test_published_direction_window_excludes_transfer_increment():
    from superproton.reorientation import HOPositions, pretransfer_direction_samples

    # The donor is oxygen 12 from f50 through f59; f60 introduces host C.
    oxygen = np.array(([11, 12] * 25) + [12] * 10 + [13, 12])[:, None]
    host = np.where(oxygen == 11, 101, np.where(oxygen == 12, 102, 103))
    tracks = make_tracks(oxygen, host, frame_to_ps=0.05)
    nf = len(oxygen)
    hp = np.zeros((nf, 1, 3))
    op = np.zeros((nf, 3, 3))
    hp[:, 0, 0] = np.arange(nf) * 2.0
    op[:, :, 0] = np.arange(nf)[:, None]
    hp[60:, 0, 0] += 1000  # Transfer jump must never enter the direction samples.
    positions = HOPositions(hp, op, np.array([1]), np.array([11, 12, 13]), np.arange(nf))
    samples = pretransfer_direction_samples(positions, tracks, classify_events(tracks))
    assert samples.source_frame.tolist() == list(range(59, 48, -1))
    np.testing.assert_allclose(samples.ratio, 2)
    np.testing.assert_allclose(samples.cos_HO, 1)


@pytest.mark.parametrize("problem", ["changed_id", "changing_cell", "nonuniform_time"])
def test_input_contract_checks_each_read_frame(tmp_path, problem):
    from superproton.io import validated_pipeline

    frames = []
    for f in range(4):
        step = 100 * f + (1 if problem == "nonuniform_time" and f == 2 else 0)
        upper = 11 if problem == "changing_cell" and f == 2 else 10
        atom = 99 if problem == "changed_id" and f == 2 else 1
        frames.append(
            f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n2\n"
            f"ITEM: BOX BOUNDS pp pp pp\n0 {upper}\n0 10\n0 10\n"
            f"ITEM: ATOMS id type xu yu zu\n{atom} 2 1 1 1\n2 4 2 1 1\n"
        )
    path = tmp_path / "invalid.trj"
    path.write_text("".join(frames))
    pipe = validated_pipeline(path)
    # A transient violation must be caught even if later frames recover.
    with pytest.raises(ValueError, match="changed|Nonuniform"):
        pipe.compute(2)


def test_nonintegral_sampling_is_rejected_before_reading():
    from superproton.workflow import Analysis

    with pytest.raises(ValueError, match="integers"):
        Analysis("unused.dump", material="CDP", frame_to_ps=0.05, stride=1.5)
    with pytest.raises(ValueError, match="Positive"):
        Analysis("unused.dump", material="CDP", frame_to_ps=float("nan"))


def test_nonproductive_counts_do_not_require_nested_pair_switches():
    from types import SimpleNamespace

    from superproton.residence import nonproductive_counts

    events = SimpleNamespace(
        n_protons=1,
        o_pair_switch_frames=[np.array([2, 3, 6, 7, 8, 10])],
        p_pair_switch_frames=[np.array([2, 5, 10])],
    )
    # The productive reset at 5 is absent from O-pair resets.
    np.testing.assert_array_equal(nonproductive_counts(events), [1, 3])


def test_geometric_fit_uses_original_selection_and_inclusive_threshold():
    from superproton.residence import fit_productive_probability

    counts = [0] * 40 + [1] * 30 + [2] * 20 + [3] * 10
    fit = fit_productive_probability(counts)
    assert fit.p == pytest.approx(0.5)
    assert len(fit.counts) == 30 and fit.counts.min() == 2


@pytest.mark.parametrize("material", ["CDP", "CHS"])
def test_geometric_fit_matches_archived_paper_distributions(material):
    from pathlib import Path

    import pandas as pd

    from superproton.residence import fit_productive_probability

    root = Path(__file__).parents[1] / "data/reference/nonproductive"
    counts = pd.read_csv(root / f"{material}_nonproductive_counts.csv")
    fits = pd.read_csv(root / f"{material}_geometric_fit.csv")
    for row in fits.itertuples():
        selected = counts.loc[counts.temperature_K == row.temperature_K, "N_nonproductive"]
        result = fit_productive_probability(selected.to_numpy())
        assert result.p == pytest.approx(row.p, abs=5.1e-7)
        assert result.counts.size == row.n_events
