"""Exact episode boundaries, including cases hidden by per-frame gaps."""

import numpy as np
import pytest

from superproton.pointing import episode_crossing_lookahead, pointing_outcomes


def test_no_acceptor_gap_carries_both_pair_members_when_donor_changes():
    donor = np.array([1, 1, 3, 3, 3, 3])[:, None]
    acceptor = np.array([2, -1, -1, 1, 1, 1])[:, None]
    attempts = pointing_outcomes(donor, acceptor)
    # At f2 the donor is 3, but the last valid pair remains {1,2} until f3.
    assert attempts.start_local.tolist() == [0, 3]
    assert attempts.end_local.tolist() == [3, 6]
    assert attempts.outcome.tolist() == ["superseded", "censored"]
    assert attempts.fresh.tolist() == [True, False]
    result = episode_crossing_lookahead(donor, acceptor, [0, 1])
    assert result.n_fresh == 1
    np.testing.assert_array_equal(result.n_crossed, [0, 0])


def test_same_host_intermediate_can_revert_before_next_interpolyanion_episode():
    donor = np.ones((6, 1), dtype=int)
    acceptor = np.array([1, 2, 2, 1, 3, 3])[:, None]
    attempts = pointing_outcomes(donor, acceptor)
    assert attempts.start_local.tolist() == [1, 4]
    first = attempts.iloc[0]
    assert first.outcome == "reverted" and first.outcome_local == 3
    assert attempts.iloc[-1].outcome == "censored"
    result = episode_crossing_lookahead(donor, acceptor, [0, 1])
    assert result.n_fresh == 2


def test_reversion_at_next_episode_boundary_precedes_simultaneous_crossing():
    donor = np.array([3, 1, 1, 2, 2])[:, None]
    acceptor = np.array([2, 2, 2, 3, 3])[:, None]
    attempts = pointing_outcomes(donor, acceptor, frame_to_ps=0.5)
    # The {1,2} attempt restores its previous {2,3} pair exactly when the
    # donor becomes 2. S18's next-episode boundary makes this a reversion.
    attempt = attempts.loc[attempts.start_local == 1].iloc[0]
    assert attempt.outcome == "reverted" and attempt.outcome_local == 3
    assert np.isnan(attempt.delay_ps)
    assert attempts.iloc[-1].outcome == "censored"
    # S20 intentionally permits crossings across later episode boundaries.
    result = episode_crossing_lookahead(donor, acceptor, [0, 0.5, 1], frame_to_ps=0.5)
    np.testing.assert_array_equal(result.n_crossed, [1, 1, 1])


def test_initial_gap_is_not_an_episode_and_rattling_stays_in_one_pair():
    donor = np.array([1, 1, 1, 2, 1])[:, None]
    acceptor = np.array([-1, -1, 2, 1, 2])[:, None]
    attempts = pointing_outcomes(donor, acceptor)
    assert len(attempts) == 1
    assert attempts.iloc[0].start_local == 2
    assert attempts.iloc[0].end_local == 5
    assert attempts.iloc[0].outcome == "committed"
    assert attempts.iloc[0].outcome_local == 3


@pytest.mark.parametrize("shape", [(0, 3), (5, 0), (0, 0)])
def test_empty_histories_keep_schema_and_zero_lookahead_counts(shape):
    hosts = np.empty(shape, dtype=np.int64)
    attempts = pointing_outcomes(hosts, hosts)
    assert attempts.empty and "outcome_frame" in attempts.columns
    result = episode_crossing_lookahead(hosts, hosts, [0, 5])
    assert result.n_fresh == 0
    np.testing.assert_array_equal(result.n_crossed, [0, 0])
    np.testing.assert_array_equal(result.fraction, [0, 0])


def test_large_ids_and_strided_source_frames_keep_metadata_and_delay():
    offset = 2**40
    donor = (offset + np.array([1, 2, 1, 3, 3]))[:, None]
    acceptor = (offset + np.array([2, 1, 2, 2, 2]))[:, None]
    attempts = pointing_outcomes(
        donor,
        acceptor,
        frame_to_ps=0.15,
        proton_ids=[offset + 7],
        source_frames=100 + 3 * np.arange(5),
    )
    first = attempts.iloc[0]
    assert first.proton_id == offset + 7
    assert first.donor_id == offset + 1 and first.target_id == offset + 2
    assert first.start_frame == 100 and first.end_frame == 109
    assert first.outcome_frame == 103
    assert first.start_ps == pytest.approx(5)
    assert first.outcome_ps == pytest.approx(5.15)
    assert first.delay_ps == pytest.approx(0.15)
