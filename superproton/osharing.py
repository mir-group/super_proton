"""Oxygen-sharing analysis (CsH2PO4 only).

Reproduces Figure 6: an oxygen is "shared" when two different protons both
count it among their two nearest oxygens, and a proton is O-sharing when
either of its own two nearest oxygens is such a shared O. The configuration
is exclusive to CsH2PO4 (in CsHSO4 the lower proton density makes it
vanishingly rare).

Note the definition is **symmetric and two-sided**, covering donor-donor,
donor-acceptor and acceptor-acceptor coincidences alike. The manuscript's
prose describes the narrower one-sided case ("a second nearest oxygen
neighbor that is the first nearest neighbor of a different proton"), but the
published 12.8% and Figure 6 use the symmetric definition. Applying the prose
definition captures only about a fifth of the sharing (2.5% rather than
13.3% on the 525 K production run) and inverts the shape of the Figure 6
curves near the switch. This module uses the symmetric definition to
reproduce the published analysis.

Two analyses are exposed:

- :func:`o_sharing_per_frame` : per-frame, per-proton boolean of whether
  either of the proton's two nearest oxygens is shared with another proton.
- :func:`sharing_fraction_vs_t_to_completion` : for each rotation event,
  align time-to-completion at 0 and report the fraction of events with
  active O-sharing at each time offset. Separated by productive vs
  nonproductive.

The mean per-frame sharing fraction (the manuscript's "12.8% at 525 K") is
also exposed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from superproton.rotation import RotationEvents
from superproton.tracking import ProtonTracks


@dataclass(frozen=True)
class OSharingData:
    """Per-frame O-sharing mask."""

    n_frames: int
    n_protons: int
    is_sharing: np.ndarray  # (n_frames, n_protons) bool

    @property
    def mean_fraction(self) -> float:
        """Average fraction of protons in an O-sharing configuration."""
        return float(self.is_sharing.mean())


def o_sharing_per_frame(tracks: ProtonTracks) -> OSharingData:
    """For each (frame, proton) return whether the proton is O-sharing.

    At each frame, pool both nearest oxygens of every proton (donor and
    acceptor). An oxygen associated with two or more distinct protons in
    that pool is shared; a proton is O-sharing if either of its own two
    oxygens is one of those.

    This is symmetric: donor-donor, donor-acceptor and acceptor-acceptor
    coincidences all count, as in the published Figure 6 analysis. See the
    module docstring for how this differs from the manuscript's prose
    description.
    """
    if np.all(tracks.acceptor_O_id == -1):
        raise ValueError(
            "tracks.acceptor_O_id is all -1; this looks like a test sentinel. "
            "o_sharing_per_frame requires real acceptor data from "
            "proton_neighbors()."
        )

    n_frames, n_H = tracks.donor_O_id.shape
    is_sharing = np.zeros((n_frames, n_H), dtype=bool)
    donor = tracks.donor_O_id
    acceptor = tracks.acceptor_O_id
    protons = np.arange(n_H)
    for t in range(n_frames):
        # Pool both slots of every proton, then find the oxygens claimed by two
        # or more DISTINCT protons. Counting distinct protons rather than raw
        # slot occurrences matters only for the degenerate case where one
        # proton's two slots hold the same oxygen -- impossible from
        # proton_neighbors, which returns the nearest and second-nearest -- but
        # it keeps a hand-built or malformed input from self-reporting as shared.
        pool_o = np.concatenate((donor[t], acceptor[t]))
        pool_h = np.concatenate((protons, protons))
        keep = pool_o >= 0
        pairs = np.unique(np.stack((pool_o[keep], pool_h[keep]), axis=1), axis=0)
        ids, counts = np.unique(pairs[:, 0], return_counts=True)
        shared = ids[counts > 1]
        if shared.size == 0:
            continue
        is_sharing[t] = np.isin(donor[t], shared) | np.isin(acceptor[t], shared)
    return OSharingData(n_frames=n_frames, n_protons=n_H, is_sharing=is_sharing)


def sharing_fraction_vs_t_to_completion(
    sharing: OSharingData,
    events: RotationEvents,
    *,
    window_ps: float = 50.0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fraction of rotation events with active O-sharing vs time before completion.

    For each event (productive or nonproductive) with switch frame ``t_f``,
    walk back over the preceding ``window_ps`` and at each offset record
    whether the proton was in an O-sharing configuration. Stack all events
    of each kind, average across them, return ``time_offset_ps -> fraction``.

    Returns
    -------
    dict
        ``{"productive": (offsets_ps, fraction), "nonproductive": (offsets_ps, fraction)}``,
        where ``offsets_ps`` is monotonically increasing from
        ``-window_ps`` to ``0`` (event completion) and ``fraction`` is
        averaged over events of that kind. Offsets where fewer than 5 events
        contribute are clipped to NaN to keep the curve interpretable.
    """
    dt = float(events.frame_to_ps)
    window_frames = int(np.ceil(window_ps / dt))
    offsets_ps = np.arange(-window_frames, 1) * dt

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kind in ("productive", "nonproductive"):
        accumulator = np.zeros(window_frames + 1, dtype=float)
        counter = np.zeros(window_frames + 1, dtype=int)
        for h in range(events.n_protons):
            if kind == "productive":
                sw_frames = events.p_pair_switch_frames[h]
            else:
                o_sw = events.o_pair_switch_frames[h]
                p_sw_set = set(events.p_pair_switch_frames[h].tolist())
                sw_frames = np.array(
                    [int(f) for f in o_sw.tolist() if int(f) not in p_sw_set],
                    dtype=np.int64,
                )
            for sw in sw_frames.tolist():
                start = max(0, sw - window_frames)
                length = sw - start + 1
                slot_start = window_frames + 1 - length
                accumulator[slot_start:] += sharing.is_sharing[start : sw + 1, h]
                counter[slot_start:] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = np.where(counter >= 5, accumulator / counter, np.nan)
        out[kind] = (offsets_ps, fraction)
    return out
