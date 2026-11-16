"""Long-range proton hop distances and per-role polyanion-oxygen displacements.

Two per-event geometric quantities measured over the rotation period
``[t_s, t_f]`` that :mod:`superproton.rotation` assigns to each pair switch:

- :func:`hop_distances` -- how far the proton itself moves when it leaves its
  original oxygen and commits to a new acceptor. One value per productive
  (polyanion-pair) switch.
- :func:`rotation_displacements` -- how far each participating oxygen moves
  during the same window, split by the role that oxygen plays in the switch.

Oxygen roles
------------
At a switch the proton's unordered oxygen pair changes from ``{a, b}`` to
``{b, c}``. The three roles follow from that transition:

==============  ====================================  =========================
role            definition                            paper label
==============  ====================================  =========================
``carrier``     the O common to both pairs; it holds  ``host`` (nonproductive)
                the proton through the rotation       ``courier`` (productive)
``acceptor``    the O only in the new pair            ``exchanger`` (NP)
                                                      ``recipient`` (P)
``donor``       the O only in the old pair            ``provider`` (productive)
==============  ====================================  =========================

A switch with no common oxygen (the proton's pair turns over completely) has no
carrier and is dropped; its role identifiers are marked ``-1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from superproton.reorientation import HOPositions
from superproton.rotation import RotationEvents
from superproton.tracking import ProtonTracks


@dataclass(frozen=True)
class HopDistances:
    """Long-range proton hop distance, one entry per productive switch."""

    distance_A: np.ndarray  # (n_events,)
    proton_index: np.ndarray  # (n_events,) local H index in the tracks
    t_s_frame: np.ndarray  # (n_events,) rotation-period start, local frame index
    t_f_frame: np.ndarray  # (n_events,) the switch frame, local frame index


@dataclass(frozen=True)
class RotationDisplacements:
    """Per-role oxygen displacement over the rotation period of each switch.

    Arrays are grouped by whether the switch was productive. ``donor`` is only
    meaningful for productive switches, mirroring the reference npz, which
    stores ``P_dr_O_all_donor`` but no nonproductive counterpart.
    """

    productive_carrier_A: np.ndarray
    productive_acceptor_A: np.ndarray
    productive_donor_A: np.ndarray
    nonproductive_carrier_A: np.ndarray
    nonproductive_acceptor_A: np.ndarray


def _roles_for_switches(
    pair_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carrier / acceptor / donor O id for each switch, from the pair sequence.

    ``pair_ids`` is the ``(n_switches + 1, 2)`` sequence of unordered pairs
    produced by :func:`superproton.rotation._unordered_pair_switches`: entry
    ``i`` is the pair held before switch ``i``, entry ``i + 1`` the pair held
    after it. Switches with no shared oxygen get ``-1`` in all three arrays.
    """
    pairs = np.atleast_2d(np.asarray(pair_ids, dtype=np.int64))
    n_sw = max(pairs.shape[0] - 1, 0)
    carrier = np.full(n_sw, -1, dtype=np.int64)
    acceptor = np.full(n_sw, -1, dtype=np.int64)
    donor = np.full(n_sw, -1, dtype=np.int64)

    for i in range(n_sw):
        prev_pair, new_pair = pairs[i], pairs[i + 1]
        common = np.intersect1d(new_pair, prev_pair)
        common = common[common >= 0]
        if common.size == 0:
            continue
        gained = np.setdiff1d(new_pair, prev_pair)
        lost = np.setdiff1d(prev_pair, new_pair)
        carrier[i] = common[0]
        if gained.size:
            acceptor[i] = gained[0]
        if lost.size:
            donor[i] = lost[0]
    return carrier, acceptor, donor


def hop_distances(
    positions: HOPositions,
    tracks: ProtonTracks,
    events: RotationEvents,
) -> HopDistances:
    """Proton displacement across each productive polyanion-pair switch.

    For every productive switch, the distance is the straight-line separation
    between the proton's position at the rotation-period start ``t_s`` and at
    the switch frame ``t_f``:

    .. code-block:: text

        d_hop = | r_H(t_f) - r_H(t_s) |

    This is the "instantaneous position" method of
    ``jump_dist.compute_jumps_by_instantaneous_position``; the alternative
    segment-averaged method in that script is not what the paper reports.

    Positions must be **unwrapped** (``xu yu zu``), so no minimum-image
    correction is applied -- a hop is a real displacement, not a pair
    separation, and wrapping it would fold long hops back into the box.
    """
    if not np.array_equal(positions.frame_indices, tracks.frame_indices):
        raise ValueError("positions and tracks frame_indices must match")

    h_id_to_col = {int(p): i for i, p in enumerate(positions.H_ids)}

    dists: list[float] = []
    h_idx: list[int] = []
    t_s_out: list[int] = []
    t_f_out: list[int] = []
    for h in range(events.n_protons):
        sw = np.asarray(events.p_pair_switch_frames[h], dtype=int)
        t_s = np.asarray(events.productive_t_s_frames[h], dtype=int)
        if sw.size == 0:
            continue
        col = h_id_to_col[int(tracks.proton_ids[h])]
        traj = positions.H_pos[:, col, :]
        for start, end in zip(t_s, sw, strict=True):
            dists.append(float(np.linalg.norm(traj[end] - traj[start])))
            h_idx.append(h)
            t_s_out.append(int(start))
            t_f_out.append(int(end))

    return HopDistances(
        distance_A=np.asarray(dists, dtype=float),
        proton_index=np.asarray(h_idx, dtype=np.int64),
        t_s_frame=np.asarray(t_s_out, dtype=np.int64),
        t_f_frame=np.asarray(t_f_out, dtype=np.int64),
    )


def _displacement(positions, o_id_to_col, o_id, t_s: int, t_f: int) -> float | None:
    """|r_O(t_f) - r_O(t_s)| for one oxygen, or None if it is not tracked."""
    col = o_id_to_col.get(int(o_id))
    if col is None:
        return None
    traj = positions.O_pos[:, col, :]
    return float(np.linalg.norm(traj[t_f] - traj[t_s]))


def rotation_displacements(
    positions: HOPositions,
    tracks: ProtonTracks,
    events: RotationEvents,
) -> RotationDisplacements:
    """Oxygen displacement over each switch's rotation period, split by role.

    For every O-pair switch, each participating oxygen's displacement is

    .. code-block:: text

        dr_O = | r_O(t_f) - r_O(t_s) |

    with ``t_s`` the rotation-period start and ``t_f`` the switch frame, and
    the roles assigned by :func:`_roles_for_switches`. Productive switches use
    the polyanion-pair rotation period; nonproductive ones use the O-pair
    period, as in :func:`superproton.rotation.classify_events`.

    Positions must be unwrapped; see :func:`hop_distances`.
    """
    if not np.array_equal(positions.frame_indices, tracks.frame_indices):
        raise ValueError("positions and tracks frame_indices must match")

    o_id_to_col = {int(p): i for i, p in enumerate(positions.O_ids)}
    out: dict[str, list[float]] = {
        "p_carrier": [], "p_acceptor": [], "p_donor": [],
        "np_carrier": [], "np_acceptor": [],
    }

    for h in range(events.n_protons):
        o_sw = np.asarray(events.o_pair_switch_frames[h], dtype=int)
        if o_sw.size == 0:
            continue
        p_sw = {int(f) for f in events.p_pair_switch_frames[h]}
        carrier, acceptor, donor = _roles_for_switches(events.o_pair_ids[h])

        # t_s for a switch depends on its kind; both lists are ordered by frame.
        t_s_prod = dict(
            zip(
                (int(f) for f in events.p_pair_switch_frames[h]),
                (int(t) for t in events.productive_t_s_frames[h]),
                strict=True,
            )
        )
        nonprod_frames = [int(f) for f in o_sw if int(f) not in p_sw]
        t_s_nonprod = dict(
            zip(
                nonprod_frames,
                (int(t) for t in events.nonproductive_t_s_frames[h]),
                strict=True,
            )
        )

        n = min(o_sw.size, carrier.size)
        for k in range(n):
            if carrier[k] < 0:
                continue
            t_f = int(o_sw[k])
            productive = t_f in p_sw
            t_s = t_s_prod.get(t_f) if productive else t_s_nonprod.get(t_f)
            if t_s is None:
                continue

            def dr(o_id: int, _ts: int = t_s, _tf: int = t_f) -> float | None:
                # t_s/t_f bound as defaults: the closure is rebuilt per switch,
                # so binding by value keeps it independent of the loop variable.
                return _displacement(positions, o_id_to_col, o_id, _ts, _tf)

            d_carrier = dr(carrier[k])
            d_acceptor = dr(acceptor[k]) if acceptor[k] >= 0 else None
            if d_carrier is None or d_acceptor is None:
                continue
            if productive:
                d_donor = dr(donor[k]) if donor[k] >= 0 else None
                if d_donor is None:
                    continue
                out["p_carrier"].append(d_carrier)
                out["p_acceptor"].append(d_acceptor)
                out["p_donor"].append(d_donor)
            else:
                out["np_carrier"].append(d_carrier)
                out["np_acceptor"].append(d_acceptor)

    arr = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    return RotationDisplacements(
        productive_carrier_A=arr["p_carrier"],
        productive_acceptor_A=arr["p_acceptor"],
        productive_donor_A=arr["p_donor"],
        nonproductive_carrier_A=arr["np_carrier"],
        nonproductive_acceptor_A=arr["np_acceptor"],
    )
