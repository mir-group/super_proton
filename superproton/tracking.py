"""Per-proton nearest-O and parent-polyanion tracking over a trajectory.

Each O atom is permanently associated with its parent X atom (P or S),
assigned at frame 0. This assumes that covalent X-O bonds persist throughout
the simulation even while the polyanion rotates.

At each frame, this module records for every proton:

- the LAMMPS identifier of its nearest O (the **donor** O), and
- the LAMMPS identifier of that O's parent polyanion.

These two integer time series are the sole input that
:func:`superproton.rotation.classify_events` consumes to detect O-pair and
P-pair switches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    import pandas as pd
    from ovito.pipeline import Pipeline


@dataclass(frozen=True)
class ProtonTracks:
    """Per-frame nearest-O and parent-polyanion identifiers for every proton.

    Attributes
    ----------
    n_protons
        Number of protons in the simulation cell.
    n_frames
        Number of frames recorded (``len(frame_indices)``).
    frame_to_ps
        Time per recorded frame in picoseconds. When the trajectory was
        sampled with ``stride`` > 1 this is ``stride`` times the raw
        per-frame dt of the LAMMPS dump.
    frame_indices
        The original LAMMPS frame index of each recorded frame
        (length ``n_frames``).
    proton_ids
        Length ``n_protons`` array of LAMMPS atom identifiers, in the order
        used by the per-frame columns. Stable across the trajectory.
    donor_O_id
        ``(n_frames, n_protons)`` int array; ``donor_O_id[t, h]`` is the
        LAMMPS identifier of proton ``h``'s nearest O at the ``t``-th
        recorded frame.
    donor_X_id
        ``(n_frames, n_protons)`` int array; ``donor_X_id[t, h]`` is the
        identifier of the polyanion (P or S) that ``donor_O_id[t, h]``
        belongs to. Constant per O across frames because parent polyanions
        are assigned at frame 0 only.
    acceptor_O_id
        ``(n_frames, n_protons)`` int array of each proton's **second**
        nearest O at every recorded frame. This is the raw second-nearest
        candidate; the hydrogen-bond distance hysteresis used in
        :mod:`superproton.pointing` is applied separately.
    acceptor_X_id
        Parent polyanion of ``acceptor_O_id[t, h]``.
    bound_polyanion_ids, mutual_bound_counts
        Optional sorted host IDs and ``(n_frames, n_hosts)`` strict mutual
        O-H bond counts, collected during the same trajectory pass. These
        include all protons, even when ``select_proton_ids`` selects a subset.
    """

    n_protons: int
    n_frames: int
    frame_to_ps: float
    frame_indices: np.ndarray
    proton_ids: np.ndarray
    donor_O_id: np.ndarray
    donor_X_id: np.ndarray
    acceptor_O_id: np.ndarray
    acceptor_X_id: np.ndarray
    bound_polyanion_ids: np.ndarray | None = None
    mutual_bound_counts: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("donor_O_id", "donor_X_id", "acceptor_O_id", "acceptor_X_id"):
            arr = getattr(self, name)
            if arr.shape != (self.n_frames, self.n_protons):
                raise ValueError(f"{name} shape {arr.shape} != ({self.n_frames}, {self.n_protons})")
        if (self.bound_polyanion_ids is None) != (self.mutual_bound_counts is None):
            raise ValueError("Bound host IDs and mutual counts must be supplied together")
        if self.mutual_bound_counts is not None and self.mutual_bound_counts.shape != (
            self.n_frames, len(self.bound_polyanion_ids)
        ):
            raise ValueError("Mutual bound counts must align with the tracked frames and hosts")


def _parent_polyanion_at_frame_zero(data, X_type_id: int, O_type_id: int) -> dict[int, int]:
    """Return ``{O_identifier: parent_X_identifier}`` from a frame-0 snapshot.

    Each X is assumed to have exactly four nearest neighbors, all of element
    O. The frame-0 nearest neighbor of each O atom is then the X atom whose
    cluster the O belongs to.
    """
    from ovito.data import NearestNeighborFinder

    p_ids = np.array(data.particles.identifiers).astype(int)
    p_types = np.array(data.particles.particle_types).astype(int)
    X_idxs = np.where(p_types == X_type_id)[0]

    finder = NearestNeighborFinder(4, data)
    neigh_idxs, _ = finder.find_all(X_idxs)
    # Sanity check: every neighbor of an X must be an O.
    neigh_types = p_types[neigh_idxs.flatten()]
    if not np.all(neigh_types == O_type_id):
        bad = np.where(neigh_types != O_type_id)[0]
        raise ValueError(
            f"At frame 0, {bad.size} X atoms have a non-O nearest neighbor. "
            "Cannot assign parent polyanions."
        )

    parent: dict[int, int] = {}
    for X_idx, O_idxs in zip(X_idxs, neigh_idxs, strict=True):
        X_id = int(p_ids[X_idx])
        for O_idx in O_idxs:
            if int(p_ids[O_idx]) in parent:
                raise ValueError("An oxygen has more than one proposed parent")
            parent[int(p_ids[O_idx])] = X_id
    if set(parent) != set(p_ids[p_types == O_type_id]):
        raise ValueError("The four-O grouping does not cover every selected oxygen")
    return parent


def parent_polyanions(pipeline: Pipeline, material: str, *, frame: int = 0) -> dict[int, int]:
    """Map oxygen identifiers to their fixed P/S parent at a reference frame.

    Uses the same four-nearest-oxygen assignment as :func:`proton_neighbors`.
    The shared workflow uses frame zero consistently, including for slices.
    """
    from superproton.io import AtomMapping, polyanion_element, species_map

    species = species_map(pipeline, material, frame=frame)
    inverse = {symbol: type_id for type_id, symbol in species.items()}
    if isinstance(material, AtomMapping) and material.oxygen_parents is not None:
        data = pipeline.compute(frame)
        ids = np.asarray(data.particles.identifiers)
        types = np.asarray(data.particles.particle_types)
        parents = dict(material.oxygen_parents)
        if set(parents) != set(ids[types == material.oxygen_type]):
            raise ValueError("Explicit mapping must cover exactly the selected oxygens")
        if not set(parents.values()) <= set(ids[types == material.center_type]):
            raise ValueError("Explicit mapping contains an unselected/missing center")
        return parents
    return _parent_polyanion_at_frame_zero(
        pipeline.compute(frame), inverse[polyanion_element(material)], inverse["O"]
    )


def proton_neighbors(
    pipeline: Pipeline,
    *,
    material: str,
    start_frame: int = 0,
    end_frame: int | None = None,
    stride: int = 1,
    frame_to_ps: float = 0.05,
    show_progress: bool = True,
    select_proton_ids: list[int] | None = None,
    collect_coordination: bool = False,
) -> ProtonTracks:
    """Walk a LAMMPS trajectory and build :class:`ProtonTracks`.

    Parameters
    ----------
    pipeline
        Ovito pipeline returned by :func:`superproton.io.read_lammps_dump`.
    material
        ``"CDP"`` or ``"CHS"``; selects the heavy-atom element (P vs S) and
        is forwarded to :func:`superproton.io.species_map` for type-id
        inference.
    start_frame, end_frame, stride
        Standard Python-style slice into the trajectory's frames. ``None``
        for ``end_frame`` means "to the last frame".
    frame_to_ps
        Time per dumped frame in picoseconds. The constructor multiplies by
        ``stride`` to get the time between recorded frames.
    show_progress
        If true, render a tqdm bar over the frame loop.
    select_proton_ids
        Optional persistent atom IDs to track; defaults to all protons.
        Selection uses IDs, not particle row numbers.
    collect_coordination
        Also count strict mutual O-H bonds in each loaded frame, avoiding a
        later trajectory pass for residence/coordination analysis. Counts
        retain their exact integer values in the smallest sufficient dtype.

    Returns
    -------
    ProtonTracks
        Two ``(n_frames, n_protons)`` integer arrays of LAMMPS identifiers.
    """
    from ovito.data import NearestNeighborFinder

    from superproton.io import (
        check_stable_particle_order,
        species_map,
    )

    species = species_map(pipeline, material, frame=start_frame)
    inv = {v: k for k, v in species.items()}
    H_type_id = inv["H"]
    O_type_id = inv["O"]

    n_total_frames = int(pipeline.source.num_frames)
    last_frame = n_total_frames if end_frame is None else min(end_frame, n_total_frames)
    frame_indices = np.arange(start_frame, last_frame, stride, dtype=int)
    n_frames = frame_indices.size
    if n_frames == 0:
        raise ValueError(f"Empty frame range [{start_frame}, {last_frame}) with stride {stride}.")
    # Indices below are resolved from the first frame and reused for all of them.
    check_stable_particle_order(pipeline, frame_indices)

    data0 = pipeline.compute(int(frame_indices[0]))
    p_types0 = np.array(data0.particles.particle_types).astype(int)
    p_ids0 = np.array(data0.particles.identifiers).astype(int)
    H_idxs = np.where(p_types0 == H_type_id)[0]
    if select_proton_ids is not None:
        requested = np.asarray(select_proton_ids)
        if (
            requested.ndim != 1
            or not requested.size
            or len(np.unique(requested)) != len(requested)
            or not np.all(np.isin(requested, p_ids0[H_idxs]))
        ):
            raise ValueError("select_proton_ids must contain unique IDs of selected H atoms")
        H_idxs = H_idxs[np.isin(p_ids0[H_idxs], requested)]
    proton_ids = p_ids0[H_idxs].copy()
    n_H = H_idxs.size

    parent = parent_polyanions(pipeline, material, frame=0)
    parent_by_row = np.full(len(p_ids0), -1, dtype=np.int64)
    oxygen_rows = np.flatnonzero(p_types0 == O_type_id)
    parent_by_row[oxygen_rows] = [parent[int(p_ids0[o])] for o in oxygen_rows]

    donor_O_id = np.zeros((n_frames, n_H), dtype=np.int64)
    donor_X_id = np.zeros((n_frames, n_H), dtype=np.int64)
    acceptor_O_id = np.zeros((n_frames, n_H), dtype=np.int64)
    acceptor_X_id = np.zeros((n_frames, n_H), dtype=np.int64)

    bound_ids, bound_counts = None, None
    if collect_coordination:
        from .coordination import _mutual_bound_counts

        # Match mutual_bound_protons' reference convention for sliced windows.
        bound_parents = parent_polyanions(pipeline, material, frame=int(frame_indices[0]))
        bound_ids, oxygen_counts = np.unique(list(bound_parents.values()), return_counts=True)
        bound_counts = np.zeros(
            (n_frames, len(bound_ids)), dtype=np.min_scalar_type(int(oxygen_counts.max()))
        )

    # Use 6 nearest neighbors so we have a comfortable margin in case the
    # first one or two are not Os during a transient.
    n_neighbors = 6
    iterator = tqdm(frame_indices, disable=not show_progress, desc="tracking")
    for t, fr_idx in enumerate(iterator):
        data = pipeline.compute(int(fr_idx))
        types_t = np.array(data.particles.particle_types).astype(int)
        ids_t = np.array(data.particles.identifiers).astype(int)
        finder = NearestNeighborFinder(n_neighbors, data)
        # find_all returns indices and delta vectors; we only need indices.
        neigh_idxs, _ = finder.find_all(H_idxs)

        is_oxygen = types_t[neigh_idxs] == O_type_id
        insufficient = np.flatnonzero(is_oxygen.sum(axis=1) < 2)
        if insufficient.size:
            raise RuntimeError(
                f"Frame {fr_idx}, proton {proton_ids[insufficient[0]]}: "
                f"fewer than 2 Os among the {n_neighbors} nearest neighbors."
            )
        # Stable partition preserves the original distance ordering and ties.
        order = np.argsort(~is_oxygen, axis=1, kind='stable')[:, :2]
        oxygen = np.take_along_axis(neigh_idxs, order, axis=1)
        donor_O_id[t] = ids_t[oxygen[:, 0]]
        acceptor_O_id[t] = ids_t[oxygen[:, 1]]
        donor_X_id[t] = parent_by_row[oxygen[:, 0]]
        acceptor_X_id[t] = parent_by_row[oxygen[:, 1]]
        if bound_counts is not None:
            bound_counts[t] = _mutual_bound_counts(
                data, bound_parents, bound_ids, hydrogen_type=H_type_id, oxygen_type=O_type_id
            )

    return ProtonTracks(
        n_protons=n_H,
        n_frames=n_frames,
        frame_to_ps=float(frame_to_ps * stride),
        frame_indices=frame_indices,
        proton_ids=proton_ids,
        donor_O_id=donor_O_id,
        donor_X_id=donor_X_id,
        acceptor_O_id=acceptor_O_id,
        acceptor_X_id=acceptor_X_id,
        bound_polyanion_ids=bound_ids,
        mutual_bound_counts=bound_counts,
    )


@dataclass(frozen=True)
class CandidateTrace:
    """Single-proton Figure 3c-e / S17 data, including original atom IDs.

    ``track`` uses first-encounter labels for plotting, as PO_indx_Hdist.py.
    ``labels`` maps those labels back to persistent IDs. ``events`` contains
    O/X pair switches and the productive reorientation start times. Boundary
    residence bands are displayed but are not complete residence samples.
    """

    track: pd.DataFrame
    events: pd.DataFrame
    labels: pd.DataFrame
    proton_id: int
    reference_frame: int


def candidate_trace(
    pipeline,
    *,
    material,
    proton_id,
    frame_to_ps=0.05,
    start_frame=0,
    end_frame=None,
    stride=1,
    reference_frame=0,
):
    """Compute a single-proton candidate trace from a LAMMPS pipeline.

    Reuses proton_neighbors and classify_events. Displacement uses OVITO's
    CalculateDisplacementsModifier (reference frame zero by default, including
    its minimum-image convention). Pair histories
    and encounter labels start at start_frame, so use start_frame=0 to reproduce
    the published candidates. Select a display window only after this analysis.
    """
    import pandas as pd
    from ovito.modifiers import CalculateDisplacementsModifier

    from .rotation import classify_events

    tracks = proton_neighbors(
        pipeline,
        material=material,
        frame_to_ps=frame_to_ps,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        select_proton_ids=[proton_id],
        show_progress=False,
    )
    events = classify_events(tracks)
    label_rows = []
    columns = {}
    for kind, values in [("O", tracks.donor_O_id[:, 0]), ("X", tracks.donor_X_id[:, 0])]:
        mapping = {int(atom): i + 1 for i, atom in enumerate(dict.fromkeys(values))}
        columns[kind] = np.array([mapping[int(atom)] for atom in values])
        label_rows.extend(
            {"species": kind, "label": label, "atom_id": atom} for atom, label in mapping.items()
        )
    initial = pipeline.compute(int(tracks.frame_indices[0]))
    h_index = int(np.flatnonzero(np.asarray(initial.particles.identifiers) == proton_id)[0])
    modifier = CalculateDisplacementsModifier(reference_frame=reference_frame)
    pipeline.modifiers.append(modifier)
    try:
        displacement = [
            float(pipeline.compute(int(f)).particles["Displacement Magnitude"][h_index])
            for f in tracks.frame_indices
        ]
    finally:
        del pipeline.modifiers[-1]
    table = pd.DataFrame(
        {
            "source_frame": tracks.frame_indices,
            "t_ps": tracks.frame_indices * frame_to_ps,
            "H_displacement_A": displacement,
            "O_nearest_neighbor": columns["O"],
            "X_nearest_neighbor": columns["X"],
            "oxygen_id": tracks.donor_O_id[:, 0],
            "center_id": tracks.donor_X_id[:, 0],
        }
    )
    rows = []
    for kind, local in [
        ("O_pair_switch", events.o_pair_switch_frames[0]),
        ("X_pair_switch", events.p_pair_switch_frames[0]),
        ("reorient_start", events.productive_t_s_frames[0]),
    ]:
        rows.extend(
            {
                "kind": kind,
                "source_frame": int(tracks.frame_indices[f]),
                "time_ps": float(tracks.frame_indices[f] * frame_to_ps),
            }
            for f in local
        )
    return CandidateTrace(
        table,
        pd.DataFrame(rows, columns=["kind", "source_frame", "time_ps"]),
        pd.DataFrame(label_rows),
        int(proton_id),
        int(reference_frame),
    )
