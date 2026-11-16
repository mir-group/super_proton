"""O-H bond reorientation diagnostics: dH/dO ratio and cos(dH, dO).

Reproduces the analyses in Section "O-H bond reorientation and slingshot
migration" (Figure 5) and SI Figure 12 (cos_HO):

For each rotation event with rotation period ``[t_s, t_f]``, the dH/dO ratio
``|dr_H(t_n)| / |dr_O(t_n)|`` is computed between consecutive saved frames
``(t_{n-1}, t_n)`` for ``t_n`` in the period. Per event the baseline (mean
over all instantaneous ratios) and the outlier (mean over instants with
Z-score > 2) are reported separately. ``cos(dr_H, dr_O)`` is averaged over
the same period as a directional sanity check on the slingshot mechanism.

The "carrier" O for productive events and the "exchanger" O for nonproductive
events are both taken to be ``donor_O_id[t_f - 1, h]`` -- the O the proton
rides through the rotation. Reading ``t_f`` itself gives the O it lands on
*after* the switch, a spectator it is not bonded to; see
:func:`compute_reorientation`.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from superproton.rotation import RotationEvents
from superproton.tracking import ProtonTracks

if TYPE_CHECKING:
    from ovito.pipeline import Pipeline


@dataclass(frozen=True)
class HOPositions:
    """H, O and polyanion-center position arrays over a contiguous frame range.

    ``X_pos`` / ``X_ids`` hold the polyanion centers (P in CsH2PO4, S in
    CsHSO4). They are read per frame, like H and O: the covalent X-O bond is
    rigid but the polyanion translates and rotates, so a single-frame X
    position is not a usable reference for any intramolecular geometry.
    They default to ``None`` so callers that only need H and O -- the dH/dO
    and pointing analyses -- can construct this without them.
    """

    H_pos: np.ndarray  # (n_frames, n_H, 3)
    O_pos: np.ndarray  # (n_frames, n_O, 3)
    H_ids: np.ndarray  # (n_H,) LAMMPS identifiers, sorted by index used in H_pos
    O_ids: np.ndarray  # (n_O,) ditto
    frame_indices: np.ndarray  # (n_frames,) the trajectory frame for each row
    X_pos: np.ndarray | None = None  # (n_frames, n_X, 3)
    X_ids: np.ndarray | None = None  # (n_X,) LAMMPS identifiers


@dataclass(frozen=True)
class ReorientationData:
    """Per-event dH/dO and cos(dH, dO) metrics."""

    proton_index: np.ndarray  # (n_events,) local H index in tracks
    kind: np.ndarray  # (n_events,) array of "productive" or "nonproductive" strings
    t_s: np.ndarray  # (n_events,) rotation-period start frame (local index)
    t_f: np.ndarray  # (n_events,) rotation-period end frame (local index)
    baseline: np.ndarray  # (n_events,) per-event mean dH/dO over [t_s, t_f]
    outlier: np.ndarray  # (n_events,) per-event mean dH/dO over Z>z_threshold instants
    cos_HO: np.ndarray  # (n_events,) per-event mean cos(dH, dO) over [t_s, t_f]

    @property
    def productive_mask(self) -> np.ndarray:
        return self.kind == "productive"


def read_HO_positions(
    pipeline: Pipeline,
    *,
    material: str,
    start_frame: int = 0,
    end_frame: int | None = None,
    stride: int = 1,
    show_progress: bool = True,
    with_X: bool = True,
) -> HOPositions:
    """Walk a trajectory once and stash all H, O and polyanion-center positions.

    Memory cost is ``8 * n_frames * (n_H + n_O) * 3`` bytes; for a 1.5 ns
    CDP run (1000 atoms, 30k frames at 50 fs stride) about 540 MB total if
    every frame is read. For tutorial-scale examples (50 ps) it is ~18 MB.
    Use ``stride`` to thin if memory is tight.
    """
    from superproton.io import (
        check_stable_particle_order,
        polyanion_element,
        species_map,
    )

    species = species_map(pipeline, material, frame=start_frame)
    inv = {v: k for k, v in species.items()}
    H_type_id = inv["H"]
    O_type_id = inv["O"]
    X_type_id = inv[polyanion_element(material)]

    n_total_frames = int(pipeline.source.num_frames)
    last_frame = n_total_frames if end_frame is None else min(end_frame, n_total_frames)
    frame_indices = np.arange(start_frame, last_frame, stride, dtype=int)
    n_frames = frame_indices.size
    # Indices below are resolved from the first frame and reused for all of them.
    check_stable_particle_order(pipeline, frame_indices)

    data0 = pipeline.compute(int(frame_indices[0]))
    p_types0 = np.array(data0.particles.particle_types).astype(int)
    p_ids0 = np.array(data0.particles.identifiers).astype(int)
    H_idxs = np.where(p_types0 == H_type_id)[0]
    O_idxs = np.where(p_types0 == O_type_id)[0]
    H_ids = p_ids0[H_idxs].copy()
    O_ids = p_ids0[O_idxs].copy()

    H_pos = np.zeros((n_frames, H_idxs.size, 3), dtype=np.float64)
    O_pos = np.zeros((n_frames, O_idxs.size, 3), dtype=np.float64)
    if with_X:
        X_idxs = np.where(p_types0 == X_type_id)[0]
        X_ids = p_ids0[X_idxs].copy()
        X_pos = np.zeros((n_frames, X_idxs.size, 3), dtype=np.float64)
    else:
        X_idxs = X_ids = X_pos = None

    iterator = tqdm(frame_indices, disable=not show_progress, desc="positions")
    for t, fr_idx in enumerate(iterator):
        data = pipeline.compute(int(fr_idx))
        pos = np.array(data.particles.positions).astype(np.float64)
        H_pos[t] = pos[H_idxs]
        O_pos[t] = pos[O_idxs]
        if with_X:
            X_pos[t] = pos[X_idxs]
    return HOPositions(
        H_pos=H_pos, O_pos=O_pos, H_ids=H_ids, O_ids=O_ids,
        frame_indices=frame_indices, X_pos=X_pos, X_ids=X_ids,
    )


def _iter_events(
    events: RotationEvents,
) -> list[tuple[int, str, int, int]]:
    """Flatten per-proton events into ``(proton_index, kind, t_s, t_f)`` rows."""
    rows: list[tuple[int, str, int, int]] = []
    for h in range(events.n_protons):
        p_sw = events.p_pair_switch_frames[h]
        prod_ts = events.productive_t_s_frames[h]
        for k in range(p_sw.size):
            rows.append((h, "productive", int(prod_ts[k]), int(p_sw[k])))

        o_sw = events.o_pair_switch_frames[h]
        p_sw_set = set(p_sw.tolist())
        nonprod_sw = np.array(
            [int(f) for f in o_sw.tolist() if int(f) not in p_sw_set],
            dtype=np.int64,
        )
        nonprod_ts = events.nonproductive_t_s_frames[h]
        for k in range(nonprod_sw.size):
            rows.append((h, "nonproductive", int(nonprod_ts[k]), int(nonprod_sw[k])))
    return rows


def compute_reorientation(
    positions: HOPositions,
    tracks: ProtonTracks,
    events: RotationEvents,
    *,
    z_threshold: float = 2.0,
) -> ReorientationData:
    """Per-event baseline + outlier dH/dO ratios and cos(dH, dO).

    Both ``positions`` and ``tracks`` must cover the same frame range; the
    function aligns the two via ``positions.frame_indices`` and
    ``tracks.frame_indices``.

    For each event with rotation period ``[t_s, t_f]`` (frame indices into
    ``tracks``), the carrier (productive) or exchanger (nonproductive) O is
    taken from ``tracks.donor_O_id[t_f - 1, h]``. Δr_H and Δr_O are then
    computed across the period in the lab frame; the ratio Δr_H/Δr_O is
    averaged for the baseline, and instants whose ratio exceeds the period
    mean by more than ``z_threshold`` standard deviations are averaged
    separately for the outlier metric.
    """
    if not np.array_equal(positions.frame_indices, tracks.frame_indices):
        raise ValueError(
            "positions.frame_indices and tracks.frame_indices must match."
        )

    h_id_to_idx = {int(p): i for i, p in enumerate(positions.H_ids)}
    o_id_to_idx = {int(p): i for i, p in enumerate(positions.O_ids)}

    rows = _iter_events(events)
    if not rows:
        return ReorientationData(
            proton_index=np.empty(0, dtype=np.int64),
            kind=np.empty(0, dtype=object),
            t_s=np.empty(0, dtype=np.int64),
            t_f=np.empty(0, dtype=np.int64),
            baseline=np.empty(0, dtype=float),
            outlier=np.empty(0, dtype=float),
            cos_HO=np.empty(0, dtype=float),
        )

    proton_index = np.empty(len(rows), dtype=np.int64)
    kind = np.empty(len(rows), dtype=object)
    t_s = np.empty(len(rows), dtype=np.int64)
    t_f = np.empty(len(rows), dtype=np.int64)
    baseline = np.full(len(rows), np.nan, dtype=float)
    outlier = np.full(len(rows), np.nan, dtype=float)
    cos_HO = np.full(len(rows), np.nan, dtype=float)

    for i, (h, this_kind, ts, tf) in enumerate(rows):
        proton_index[i] = h
        kind[i] = this_kind
        t_s[i] = ts
        t_f[i] = tf

        if tf - ts < 2:
            continue

        h_idx = h_id_to_idx[int(tracks.proton_ids[h])]
        # t_f IS the pair-switch frame, so donor_O_id[t_f] is already the O the
        # proton lands on AFTER the switch -- never the carrier it was bonded to
        # during [t_s, t_f]. Read the frame before: donor_O_id[t_f - 1] is the
        # donor for every frame inside the rotation period.
        carrier_O_id = int(tracks.donor_O_id[tf - 1, h])
        try:
            o_idx = o_id_to_idx[carrier_O_id]
        except KeyError:
            continue

        H_seg = positions.H_pos[ts : tf + 1, h_idx, :]
        O_seg = positions.O_pos[ts : tf + 1, o_idx, :]
        ratios, cos_inst = _increment_metrics(H_seg, O_seg)
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size:
            baseline[i] = float(ratios.mean())
            std = float(ratios.std(ddof=0))
            if std > 0:
                spikes = ratios > ratios.mean() + z_threshold * std
                if spikes.any(): outlier[i] = float(ratios[spikes].mean())
        finite_cos = cos_inst[np.isfinite(cos_inst)]
        if finite_cos.size: cos_HO[i] = float(finite_cos.mean())

    return ReorientationData(
        proton_index=proton_index,
        kind=kind,
        t_s=t_s,
        t_f=t_f,
        baseline=baseline,
        outlier=outlier,
        cos_HO=cos_HO,
    )


def reorientation_samples(positions, tracks, events, *, z_threshold=2.0):
    """Frame-resolved lab-frame increments underlying SI S15 and Fig. 5.

    Each row is one saved-frame increment in an event, including invalid
    zero-displacement samples as NaN. Event IDs follow ``_iter_events``.
    Ratios use the same bonded oxygen and period endpoints as
    ``compute_reorientation``. The outlier flag uses each period's mean and
    population standard deviation. Original IDs and source frames are kept.
    """
    import pandas as pd
    if not np.array_equal(positions.frame_indices, tracks.frame_indices):
        raise ValueError('Positions and tracks must cover identical frames')
    h_index = {int(a):i for i,a in enumerate(positions.H_ids)}
    o_index = {int(a):i for i,a in enumerate(positions.O_ids)}
    chunks = []
    for event_id,(h,kind,ts,tf) in enumerate(_iter_events(events)):
        if tf-ts < 2: continue
        oxygen_id = int(tracks.donor_O_id[tf-1,h])
        if oxygen_id not in o_index: continue
        hs = positions.H_pos[ts:tf+1,h_index[int(tracks.proton_ids[h])]]
        os = positions.O_pos[ts:tf+1,o_index[oxygen_id]]
        ratio, cosine = _increment_metrics(hs,os)
        valid = np.isfinite(ratio)
        outlier = np.zeros(len(ratio),dtype=bool)
        if valid.any() and ratio[valid].std() > 0:
            outlier = ratio > ratio[valid].mean()+z_threshold*ratio[valid].std()
        chunks.append(pd.DataFrame(dict(event_id=event_id,proton_id=int(tracks.proton_ids[h]),
            oxygen_id=oxygen_id,kind=kind,local_frame=np.arange(ts+1,tf+1),
            source_frame=tracks.frame_indices[ts+1:tf+1],
            time_in_period_ps=np.arange(1,tf-ts+1)*tracks.frame_to_ps,
            ratio=ratio,cos_HO=cosine,outlier=outlier)))
    columns=['event_id','proton_id','oxygen_id','kind','local_frame','source_frame',
             'time_in_period_ps','ratio','cos_HO','outlier']
    return pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame(columns=columns)


def _increment_metrics(h_positions, o_positions):
    """Shared numerical kernel for frame exports and per-event summaries."""
    dh, do = np.diff(h_positions,axis=0), np.diff(o_positions,axis=0)
    nh, no = np.linalg.norm(dh,axis=1), np.linalg.norm(do,axis=1)
    ratios, cosine = np.full(len(nh),np.nan), np.full(len(nh),np.nan)
    valid = no > 1e-9
    ratios[valid] = nh[valid]/no[valid]
    both = valid & (nh > 1e-9)
    cosine[both] = np.sum(dh[both]*do[both],axis=1)/(nh[both]*no[both])
    return ratios,cosine


def pretransfer_direction_samples(positions, tracks, events, *, preceding_frames=50):
    """SI S16's pre-transfer direction samples, following record_HO_dis_ratio.

    Walk backward from t_f-1, at most preceding_frames-1 increments, stopping
    when the donor oxygen changes. The transfer increment itself is excluded.
    As in the source, the oldest retained increment can begin one frame before
    the stable-donor run. Missing pre-window data are excluded rather than
    wrapping negative indices. Rows retain reverse-time order within each
    productive event. This is distinct from full-period reorientation samples.
    """
    import pandas as pd
    if preceding_frames < 2: raise ValueError('preceding_frames must be at least 2')
    if not np.array_equal(positions.frame_indices, tracks.frame_indices):
        raise ValueError('Positions and tracks must cover identical frames')
    hi = {int(a):i for i,a in enumerate(positions.H_ids)}
    oi = {int(a):i for i,a in enumerate(positions.O_ids)}
    chunks=[]
    for h,switches in enumerate(events.p_pair_switch_frames):
        for tf in switches:
            tf=int(tf)
            if tf < 2: continue
            oxygen=int(tracks.donor_O_id[tf-1,h])
            frames=[]
            for f in range(tf-1,max(tf-preceding_frames,0),-1):
                if tracks.donor_O_id[f,h] != oxygen: break
                frames.append(f)
            if not frames: continue
            first=frames[-1]
            ratio,cosine=_increment_metrics(positions.H_pos[first-1:tf,hi[int(tracks.proton_ids[h])]],
                                            positions.O_pos[first-1:tf,oi[oxygen]])
            chunks.append(pd.DataFrame(dict(proton_id=int(tracks.proton_ids[h]),oxygen_id=oxygen,
                switch_source_frame=int(tracks.frame_indices[tf]),source_frame=tracks.frame_indices[frames],
                time_before_transfer_ps=(np.asarray(frames)-tf)*tracks.frame_to_ps,
                ratio=ratio[::-1],cos_HO=cosine[::-1])))
    columns=['proton_id','oxygen_id','switch_source_frame','source_frame','time_before_transfer_ps','ratio','cos_HO']
    return pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame(columns=columns)
