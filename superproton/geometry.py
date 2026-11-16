"""Geometric diagnostics: X-O-H angle, X-O VCF, polyanion orientation, RDF.

Each function operates on pre-loaded position arrays so it can be unit-
tested without ovito. Position-loading helpers live alongside (or in
:mod:`superproton.reorientation` / :mod:`superproton.correlation`); the
geometry functions consume their outputs.

Reproduces:

- Table S6 (covalent X-O-H geometry over stable-bond intervals).
- Figure 8d (vector autocorrelation function of X-O unit vectors).
- Figure 8b-c, e-f (polyanion orientation cumulative distribution in
  spherical angles).
- SI Figure (polyanion-center radial distribution function).

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from superproton.correlation import PolyanionCenters
from superproton.io import min_image as io_min_image
from superproton.io import pairwise_min_image_distances
from superproton.reorientation import HOPositions
from superproton.tracking import ProtonTracks


@dataclass(frozen=True)
class XOHGeometry:
    """Means and standard deviations of the proton's covalent X-O-H geometry."""

    d_OH_mean_A: float
    d_OH_std_A: float
    d_XO_mean_A: float
    d_XO_std_A: float
    angle_XOH_mean_deg: float
    angle_XOH_std_deg: float
    n_samples: int


def xoh_geometry(
    positions: HOPositions,
    tracks: ProtonTracks,
    centers: PolyanionCenters,
    *,
    dwell_ps: float = 2.0,
) -> XOHGeometry:
    """X-O-H geometry sampled over stable-bond intervals.

    A "stable" interval is a maximal run of consecutive frames during which
    proton ``h``'s donor O is unchanged AND the run length exceeds
    ``dwell_ps``. Within each stable run, every frame contributes one sample
    of ``d_OH``, ``d_XO``, and ``angle_XOH``.

    All three positions are read **per frame**, X included, from
    ``positions.X_pos``. Measuring d_XO against a single-frame X position
    instead tracks the polyanion's translation rather than the covalent bond
    and inflates its spread about sevenfold, so ``positions`` must carry
    ``X_pos`` (the default from :func:`superproton.reorientation.read_HO_positions`)
    and this raises if it does not.

    ``centers`` is used only for its simulation cell, to apply the
    minimum-image convention; ``centers.positions`` is not read.

    The paper samples these statistics over the O-H reorientation period rather
    than over fast proton rattling, which is what ``dwell_ps`` selects: 2 ps for
    CsH2PO4 and 10 ps for CsHSO4 in SI Table S6.
    """
    if not np.array_equal(positions.frame_indices, tracks.frame_indices):
        raise ValueError("positions and tracks frame_indices must match")
    if positions.X_pos is None or positions.X_ids is None:
        raise ValueError(
            "xoh_geometry needs per-frame polyanion-center positions. Call "
            "read_HO_positions(..., with_X=True) (the default) so X_pos is populated; "
            "a single-frame X position inflates the X-O bond-length spread by ~7x."
        )

    dwell_frames = int(np.ceil(dwell_ps / float(tracks.frame_to_ps)))
    h_id_to_idx = {int(p): i for i, p in enumerate(positions.H_ids)}
    o_id_to_idx = {int(p): i for i, p in enumerate(positions.O_ids)}
    x_id_to_idx = {int(p): i for i, p in enumerate(positions.X_ids)}

    # Minimum image so wrapped and unwrapped LAMMPS coordinates both give the
    # correct intramolecular distances.
    cell = np.asarray(centers.cell, dtype=float)
    inv_cell = np.linalg.inv(cell)

    def min_image(diff: np.ndarray) -> np.ndarray:
        return io_min_image(diff, cell, inv_cell)

    d_OH_samples: list[float] = []
    d_XO_samples: list[float] = []
    angle_samples_deg: list[float] = []

    for h in range(tracks.n_protons):
        donor_O = tracks.donor_O_id[:, h]
        donor_X = tracks.donor_X_id[:, h]
        # Identify runs of constant donor O.
        change_points = np.where(np.diff(donor_O) != 0)[0] + 1
        run_starts = np.concatenate([[0], change_points])
        run_ends = np.concatenate([change_points, [donor_O.size]])
        for s, e in zip(run_starts, run_ends):
            if e - s < dwell_frames:
                continue
            o_id = int(donor_O[s])
            x_id = int(donor_X[s])
            if o_id not in o_id_to_idx or x_id not in x_id_to_idx:
                continue
            h_col = h_id_to_idx[int(tracks.proton_ids[h])]
            H_pos = positions.H_pos[s:e, h_col, :]
            O_pos = positions.O_pos[s:e, o_id_to_idx[o_id], :]
            # Per frame, not frame zero: the polyanion translates and rotates,
            # so d_XO against a fixed X position is dominated by that motion
            # rather than by the covalent bond.
            X_pos = positions.X_pos[s:e, x_id_to_idx[x_id], :]
            OH = min_image(H_pos - O_pos)
            XO = min_image(O_pos - X_pos)
            d_OH = np.linalg.norm(OH, axis=1)
            d_XO = np.linalg.norm(XO, axis=1)
            cos_xoh = -(OH * XO).sum(axis=1) / (d_OH * d_XO)
            # X-O-H angle is the angle at vertex O between vectors O->X and O->H.
            cos_xoh = np.clip(cos_xoh, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_xoh))
            d_OH_samples.extend(d_OH.tolist())
            d_XO_samples.extend(d_XO.tolist())
            angle_samples_deg.extend(angle_deg.tolist())

    if not d_OH_samples:
        return XOHGeometry(
            d_OH_mean_A=float("nan"),
            d_OH_std_A=float("nan"),
            d_XO_mean_A=float("nan"),
            d_XO_std_A=float("nan"),
            angle_XOH_mean_deg=float("nan"),
            angle_XOH_std_deg=float("nan"),
            n_samples=0,
        )

    return XOHGeometry(
        d_OH_mean_A=float(np.mean(d_OH_samples)),
        d_OH_std_A=float(np.std(d_OH_samples)),
        d_XO_mean_A=float(np.mean(d_XO_samples)),
        d_XO_std_A=float(np.std(d_XO_samples)),
        angle_XOH_mean_deg=float(np.mean(angle_samples_deg)),
        angle_XOH_std_deg=float(np.std(angle_samples_deg)),
        n_samples=len(d_OH_samples),
    )


def _representative_oxygen(O_pos, X_pos, parents, cell, selection):
    """Select once, retaining oxygen order to resolve ties as in the source."""
    if selection not in {"first", "max_abs_z", "min_theta"}:
        raise ValueError("selection must be first, max_abs_z, or min_theta")
    chosen = []
    for x in range(len(X_pos)):
        children = np.flatnonzero(parents == x)
        if not children.size:
            raise ValueError("Some polyanion has no children in parent_X_index")
        vec = O_pos[children] - X_pos[x]
        if cell is not None:
            vec = io_min_image(vec, cell)
        norms = np.linalg.norm(vec, axis=1)
        if np.any(norms == 0):
            raise ValueError("Zero-length X-O bond")
        z = vec[:, 2] / norms
        score = np.abs(z) if selection == "max_abs_z" else z
        chosen.append(children[0 if selection == "first" else np.argmax(score)])
    return np.asarray(chosen)


def xo_unit_vectors_per_frame(
    O_pos: np.ndarray, X_pos: np.ndarray, parent_X_index: np.ndarray,
    *, cell: np.ndarray | None = None, selection: str = "first",
) -> np.ndarray:
    """Track one fixed oxygen per X; return (frames, polyanions, 3) unit vectors.

    ``first`` retains the original API. ``max_abs_z`` reproduces the archived
    Fig. 8 extraction; ``min_theta`` selects the most positive initial z.
    Supply a row-vector lattice matrix for wrapped/periodic coordinates.
    Selection is made only at the first supplied frame, never per frame.
    """
    O_pos, X_pos = np.asarray(O_pos), np.asarray(X_pos)
    X0 = X_pos[0] if X_pos.ndim == 3 else X_pos
    chosen = _representative_oxygen(O_pos[0], X0, np.asarray(parent_X_index),
                                    cell, selection)
    vec = O_pos[:, chosen] - X_pos
    if cell is not None:
        vec = io_min_image(vec, cell)
    norms = np.linalg.norm(vec, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-length X-O bond")
    return vec / norms


def align_sulfate_orientations(unit_vectors: np.ndarray, *,
                               initial_vectors: np.ndarray | None = None,
                               azimuth_deg: float = -4.0,
                               tolerance_deg: float = 1.0) -> np.ndarray:
    """Copy and align the original CHS I41/amd sublattices for Fig. 8.

    Vectors whose *initial* azimuth lies within 1 degree of -4 degrees receive
    the fixed operation (x,y,z) -> (y,x,z) throughout the trajectory.
    This mapping is specific to the paper's initial axes,
    not a general space-group symmetry solver or instantaneous folding.
    """
    vectors = np.asarray(unit_vectors, dtype=float).copy()
    initial = vectors[0] if initial_vectors is None else np.asarray(initial_vectors)
    phi, _ = spherical_angles(initial)
    delta = (np.degrees(phi) - azimuth_deg + 180) % 360 - 180
    mask = np.abs(delta) <= tolerance_deg
    vectors[:, mask, :2] = vectors[:, mask, :2][..., ::-1]
    return vectors


def read_xo_unit_vectors(pipeline, material, *, start_frame=0, end_frame=None,
                         stride=1, align_sulfate=True, selection="max_abs_z"):
    """Periodic X-O extraction without storing all atomic positions.

    Select bonds and the CHS alignment mask at trajectory frame zero, even
    when returning a later frame slice. Track atom identifiers across frames.
    Returns (unit_vectors, X_identifiers). Paper Fig. 8 uses frames [0,80000),
    550 K, and single-polyanion column 23 (zero based).
    """
    from ovito.data import NearestNeighborFinder
    from superproton.io import polyanion_element, species_map

    if stride < 1 or start_frame < 0:
        raise ValueError("stride must be positive and start_frame nonnegative")
    data = pipeline.compute(0)
    species = {v: k for k, v in species_map(pipeline, material).items()}
    ids = np.asarray(data.particles.identifiers)
    types = np.asarray(data.particles.particle_types)
    x_idx = np.flatnonzero(types == species[polyanion_element(material)])
    from superproton.io import AtomMapping
    from superproton.tracking import parent_polyanions
    membership = parent_polyanions(pipeline, material)
    if isinstance(material, AtomMapping) and material.oxygen_parents is not None:
        o_idx = np.flatnonzero(types == species["O"])
        x_columns = {int(ids[idx]): col for col,idx in enumerate(x_idx)}
        parents = np.array([x_columns[membership[int(ids[idx])]] for idx in o_idx])
    else:
        neighbors, _ = NearestNeighborFinder(4, data).find_all(x_idx)
        if not np.all(types[neighbors] == species["O"]):
            raise ValueError("Each polyanion must have four oxygen neighbors")
        o_idx = neighbors.ravel()
        parents = np.repeat(np.arange(len(x_idx)), 4)
    pos = np.asarray(data.particles.positions)
    cell0 = np.asarray(data.cell.matrix)[:, :3].T
    selected = _representative_oxygen(pos[o_idx], pos[x_idx], parents, cell0, selection)
    o_ids, x_ids = ids[o_idx[selected]], ids[x_idx].copy()
    initial = io_min_image(pos[o_idx[selected]] - pos[x_idx], cell0)
    initial /= np.linalg.norm(initial, axis=1, keepdims=True)
    stop = pipeline.source.num_frames if end_frame is None else end_frame
    if not start_frame < stop <= pipeline.source.num_frames:
        raise ValueError("Empty or out-of-bounds frame range")
    vectors = np.empty((len(range(start_frame, stop, stride)), len(x_idx), 3))
    for row, frame in enumerate(range(start_frame, stop, stride)):
        data = pipeline.compute(frame)
        frame_ids = np.asarray(data.particles.identifiers)
        order = np.argsort(frame_ids)
        ix = order[np.searchsorted(frame_ids[order], x_ids)]
        io = order[np.searchsorted(frame_ids[order], o_ids)]
        if not (np.array_equal(frame_ids[ix], x_ids) and np.array_equal(frame_ids[io], o_ids)):
            raise ValueError("Particle identifiers changed during the trajectory")
        pos = np.asarray(data.particles.positions)
        vec = io_min_image(pos[io] - pos[ix], np.asarray(data.cell.matrix)[:, :3].T)
        vectors[row] = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    if material == "CHS" and align_sulfate:
        vectors = align_sulfate_orientations(vectors, initial_vectors=initial)
    return vectors, x_ids


def orientation_density(unit_vectors: np.ndarray, *, bins=100):
    """Angular density per degree squared (no solid-angle/Jacobian correction).

    Uses the sample min/max bin edges, as in XO_rot.py's hist2d. Returns
    (density, phi_edges_deg, theta_edges_deg), with phi on the first axis.
    """
    phi, theta = spherical_angles(unit_vectors)
    return np.histogram2d(np.degrees(phi).ravel(), np.degrees(theta).ravel(),
                          bins=bins, density=True)


def vector_autocorrelation(
    unit_vectors: np.ndarray, *, max_lag_frames: int, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Time-window averaged X-O VCF.

    ``unit_vectors`` shape ``(n_frames, n_polyanions, 3)``. Computes

        VCF(dt) = (1 / N_tau) sum_t (1 / M) sum_i a_i(t) . a_i(t + dt)

    for lags ``dt`` from 0 to ``max_lag_frames`` (stride ``stride``).
    Returns (lag_frames, vcf).
    """
    n_frames, _, _ = unit_vectors.shape
    max_lag = min(max_lag_frames, n_frames - 1)
    lags = np.arange(0, max_lag + 1, stride, dtype=int)
    vcf = np.zeros(lags.size, dtype=float)
    for k, lag in enumerate(lags):
        if lag == 0:
            vcf[k] = 1.0
            continue
        n_pairs = n_frames - lag
        dot = (unit_vectors[:n_pairs] * unit_vectors[lag : lag + n_pairs]).sum(axis=2)
        vcf[k] = float(dot.mean())
    return lags.astype(float), vcf


def spherical_angles(unit_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian unit vectors to ``(phi_rad, theta_rad)`` in [-pi, pi] x [0, pi].

    ``phi`` is the azimuthal angle measured from +x in the xy-plane; ``theta``
    is the polar angle measured from +z.
    """
    z = np.clip(unit_vectors[..., 2], -1.0, 1.0)
    theta = np.arccos(z)
    phi = np.arctan2(unit_vectors[..., 1], unit_vectors[..., 0])
    return phi, theta


def polyanion_center_rdf(
    positions_per_frame: np.ndarray,
    cell: np.ndarray,
    *,
    r_max_A: float = 10.0,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Time-averaged radial distribution function ``g(r)`` between polyanion centers.

    ``positions_per_frame`` shape ``(n_frames, n_polyanions, 3)``. Minimum-image
    convention is applied using ``cell`` (3x3 lattice matrix, row-vector
    convention). Returns ``(r_centers, g_of_r)`` arrays.
    """
    n_frames, n_P, _ = positions_per_frame.shape
    inv_cell = np.linalg.inv(cell)
    bin_edges = np.linspace(0.0, r_max_A, n_bins + 1)
    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    histogram = np.zeros(n_bins, dtype=float)
    for t in range(n_frames):
        d = pairwise_min_image_distances(positions_per_frame[t], cell, inv_cell)
        # Drop the zero self-distance on the diagonal, not a genuine close pair.
        d = d[d > 1e-9]
        histogram += np.histogram(d, bins=bin_edges)[0]
    # Normalize by ideal-gas shell density: g(r) = histogram / (n_frames *
    # n_P * (n_P - 1) / V) / (4 pi r^2 dr) ... use the standard prefactor.
    volume = float(abs(np.linalg.det(cell)))
    density = n_P / volume
    shell_volume = 4.0 * np.pi * bin_centers**2 * bin_widths
    norm = n_frames * n_P * density * shell_volume
    g = np.where(norm > 0, histogram / norm, 0.0)
    return bin_centers, g


def orientation_plot_data(vectors, *, single_index=23, experimental_sites=None, bins=100):
    """Build Fig. 8 inputs from mapped vectors; optional [Oxyz,Xxyz] site rows.

    Experimental all-polyanion markers are omitted when no sites are supplied;
    the single-bond panel always marks its first observed direction. ``bins``
    accepts a bin count or explicit (azimuth, polar) edges in degrees, as in
    :func:`orientation_density`; the default preserves the paper's maps.
    """
    if not 0 <= single_index < vectors.shape[1]:
        raise ValueError("single_index outside the polyanion array")
    result = {}
    for scope, vec in [('single',vectors[:,single_index]),('all',vectors)]:
        density,phi,theta = orientation_density(vec, bins=bins)
        markers = vectors[0,single_index:single_index+1]
        if scope == 'all':
            markers = np.empty((0,3))
            if experimental_sites is not None:
                sites = np.asarray(experimental_sites,dtype=float)
                markers = sites[:,:3]-sites[:,3:]
                markers /= np.linalg.norm(markers,axis=1,keepdims=True)
        mp,mt = spherical_angles(markers)
        result[scope] = dict(density=density,phi_edges_deg=phi,theta_edges_deg=theta,
                            marker_phi_deg=np.degrees(mp),marker_theta_deg=np.degrees(mt))
    return result


def rigid_rotation_comparison(geometry, oxygen_displacement_A, hop_distance_A):
    """Paper's linear/bent lever estimates from measured average geometry.

    These are geometric comparisons under the stated rigid-lever model, not
    an event-wise angular fit. Distances are in Angstrom. Keep them separate
    from instantaneous dH/dO statistics and oxygen angular displacement.
    """
    import pandas as pd
    xo, oh = geometry.d_XO_mean_A, geometry.d_OH_mean_A
    angle = np.radians(geometry.angle_XOH_mean_deg)
    bent = np.sqrt(xo*xo+oh*oh-2*xo*oh*np.cos(angle))
    levers = np.array([xo+oh,bent])
    prediction = oxygen_displacement_A*levers/xo
    return pd.DataFrame(dict(geometry=['linear','measured_bent'],lever_A=levers,
        oxygen_displacement_A=oxygen_displacement_A,predicted_proton_displacement_A=prediction,
        measured_hop_A=hop_distance_A,excess_A=hop_distance_A-prediction))
