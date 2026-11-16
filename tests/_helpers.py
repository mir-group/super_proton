"""Builders shared by the synthetic unit tests.

``ProtonTracks`` is the input to most of the package, so nearly every test
module needs to fabricate one. Four modules had their own near-identical copy
of this builder, in two variants differing only in whether the acceptor arrays
were supplied; this is that builder, with the acceptor arrays optional.
"""

from __future__ import annotations

import numpy as np

from superproton.tracking import ProtonTracks


def make_tracks(
    donor_O_id: np.ndarray,
    donor_X_id: np.ndarray,
    acceptor_O_id: np.ndarray | None = None,
    acceptor_X_id: np.ndarray | None = None,
    *,
    frame_to_ps: float = 0.05,
) -> ProtonTracks:
    """A ``ProtonTracks`` over ``(n_frames, n_protons)`` identifier arrays.

    The acceptor arrays default to ``-1`` sentinels, which is what the
    analyses that only read the donor expect. Supply them for the pointing and
    O-sharing tests, which read both slots.
    """
    n_frames, n_protons = donor_O_id.shape
    sentinel = np.full_like(donor_O_id, -1)
    return ProtonTracks(
        n_protons=n_protons,
        n_frames=n_frames,
        frame_to_ps=frame_to_ps,
        frame_indices=np.arange(n_frames, dtype=int),
        proton_ids=np.arange(1, n_protons + 1, dtype=int),
        donor_O_id=donor_O_id,
        donor_X_id=donor_X_id,
        acceptor_O_id=sentinel if acceptor_O_id is None else acceptor_O_id,
        acceptor_X_id=sentinel if acceptor_X_id is None else acceptor_X_id,
    )
