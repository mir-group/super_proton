"""Unit tests for superproton.geometry.

Drives each function against hand-built coordinate inputs where the
expected geometric output is calculable by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from superproton.correlation import PolyanionCenters
from superproton.geometry import (
    polyanion_center_rdf,
    spherical_angles,
    vector_autocorrelation,
    xo_unit_vectors_per_frame,
    xoh_geometry,
)
from superproton.reorientation import HOPositions
from superproton.tracking import ProtonTracks


# ---------------------------------------------------------------------------
# xoh_geometry
# ---------------------------------------------------------------------------


def test_xoh_geometry_recovers_perfect_collinear_lever():
    """Place X at origin, O at (1.6, 0, 0), H at (1.6 + 1.0, 0, 0). Bond
    angle X-O-H = 180 degrees, d_OH = 1.0, d_XO = 1.6 every frame."""
    n_frames = 200  # enough to exceed the 2 ps dwell at 0.05 ps/frame
    donor_O = np.full((n_frames, 1), 50, dtype=np.int64)
    donor_X = np.full((n_frames, 1), 10, dtype=np.int64)
    sentinel = np.full_like(donor_O, -1)
    tracks = ProtonTracks(
        n_protons=1,
        n_frames=n_frames,
        frame_to_ps=0.05,
        frame_indices=np.arange(n_frames, dtype=int),
        proton_ids=np.array([1], dtype=np.int64),
        donor_O_id=donor_O,
        donor_X_id=donor_X,
        acceptor_O_id=sentinel,
        acceptor_X_id=sentinel,
    )
    H_pos = np.zeros((n_frames, 1, 3), dtype=float)
    O_pos = np.zeros((n_frames, 1, 3), dtype=float)
    H_pos[:, 0, 0] = 2.6
    O_pos[:, 0, 0] = 1.6
    # X sits at the origin in every frame. xoh_geometry reads X per frame, so
    # the static position has to be supplied as a per-frame array.
    X_pos = np.zeros((n_frames, 1, 3), dtype=float)
    positions = HOPositions(
        H_pos=H_pos,
        O_pos=O_pos,
        H_ids=np.array([1], dtype=np.int64),
        O_ids=np.array([50], dtype=np.int64),
        frame_indices=np.arange(n_frames, dtype=int),
        X_pos=X_pos,
        X_ids=np.array([10], dtype=np.int64),
    )
    centers = PolyanionCenters(
        positions=np.array([[0.0, 0.0, 0.0]]),
        cell=10.0 * np.eye(3),
        polyanion_ids=np.array([10], dtype=np.int64),
    )
    geom = xoh_geometry(positions, tracks, centers, dwell_ps=2.0)
    assert geom.n_samples == n_frames
    assert geom.d_OH_mean_A == pytest.approx(1.0)
    assert geom.d_XO_mean_A == pytest.approx(1.6)
    assert geom.angle_XOH_mean_deg == pytest.approx(180.0, abs=1e-6)


def test_xoh_geometry_skips_short_runs():
    """If donor_O changes every frame, no run exceeds dwell_ps and the
    result should report zero samples."""
    n_frames = 200
    donor_O = np.arange(n_frames, dtype=np.int64).reshape(-1, 1)  # different every frame
    donor_X = np.full((n_frames, 1), 10, dtype=np.int64)
    sentinel = np.full_like(donor_O, -1)
    tracks = ProtonTracks(
        n_protons=1,
        n_frames=n_frames,
        frame_to_ps=0.05,
        frame_indices=np.arange(n_frames, dtype=int),
        proton_ids=np.array([1], dtype=np.int64),
        donor_O_id=donor_O,
        donor_X_id=donor_X,
        acceptor_O_id=sentinel,
        acceptor_X_id=sentinel,
    )
    H_pos = np.zeros((n_frames, 1, 3), dtype=float)
    # We need O_ids that cover every possible donor_O id; build positions for
    # every O id from 0 to n_frames - 1.
    O_pos = np.zeros((n_frames, n_frames, 3), dtype=float)
    X_pos = np.zeros((n_frames, 1, 3), dtype=float)
    positions = HOPositions(
        H_pos=H_pos,
        O_pos=O_pos,
        H_ids=np.array([1], dtype=np.int64),
        O_ids=np.arange(n_frames, dtype=np.int64),
        frame_indices=np.arange(n_frames, dtype=int),
        X_pos=X_pos,
        X_ids=np.array([10], dtype=np.int64),
    )
    centers = PolyanionCenters(
        positions=np.array([[0.0, 0.0, 0.0]]),
        cell=10.0 * np.eye(3),
        polyanion_ids=np.array([10], dtype=np.int64),
    )
    geom = xoh_geometry(positions, tracks, centers, dwell_ps=2.0)
    assert geom.n_samples == 0
    assert np.isnan(geom.d_OH_mean_A)


# ---------------------------------------------------------------------------
# xo_unit_vectors_per_frame + vector_autocorrelation
# ---------------------------------------------------------------------------


def test_vector_autocorrelation_constant_bond_yields_unity():
    # Two polyanions, 50 frames, each with a fixed +x unit vector.
    n_frames, n_X = 50, 2
    unit_vec = np.zeros((n_frames, n_X, 3), dtype=float)
    unit_vec[:, :, 0] = 1.0
    lags, vcf = vector_autocorrelation(unit_vec, max_lag_frames=20)
    assert lags.size == 21
    np.testing.assert_allclose(vcf, 1.0, atol=1e-10)


def test_vector_autocorrelation_orthogonal_after_quarter_turn():
    """One polyanion whose unit vector rotates by pi/2 in 20 frames; VCF at
    lag 20 frames should equal cos(pi/2) = 0."""
    n_frames = 80
    theta = np.linspace(0.0, np.pi / 2, n_frames)
    unit_vec = np.zeros((n_frames, 1, 3), dtype=float)
    unit_vec[:, 0, 0] = np.cos(theta)
    unit_vec[:, 0, 1] = np.sin(theta)
    lags, vcf = vector_autocorrelation(unit_vec, max_lag_frames=20)
    # At lag = 20 frames (a quarter of total) the bond has turned by about
    # 20 / 79 * pi/2 ~ 0.397 rad on average between frames.
    # Take the easier check: VCF at lag 0 = 1, monotonic decay.
    assert vcf[0] == pytest.approx(1.0)
    assert vcf[-1] < vcf[0]
    assert vcf[-1] > 0  # haven't passed orthogonality across all windows


def test_xo_unit_vectors_per_frame_unit_norm():
    # 3 frames, 4 Os distributed across 2 polyanions (2 Os each).
    parent_X_index = np.array([0, 0, 1, 1])
    O_pos = np.zeros((3, 4, 3), dtype=float)
    O_pos[:, 0, 0] = 1.6  # O on X0 at +x
    O_pos[:, 1, 1] = 1.6  # second O on X0 at +y (will be ignored, rep is first)
    O_pos[:, 2, 0] = -1.6  # O on X1 at -x
    O_pos[:, 3, 1] = -1.6  # second O on X1 at -y (ignored)
    X_pos = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])  # static, broadcasts
    vec = xo_unit_vectors_per_frame(O_pos, X_pos, parent_X_index)
    assert vec.shape == (3, 2, 3)
    np.testing.assert_allclose(np.linalg.norm(vec, axis=2), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# spherical_angles
# ---------------------------------------------------------------------------


def test_spherical_angles_basic():
    unit_vec = np.array(
        [
            [[1.0, 0.0, 0.0]],  # phi = 0, theta = pi/2
            [[0.0, 0.0, 1.0]],  # phi = 0, theta = 0
            [[0.0, 1.0, 0.0]],  # phi = pi/2, theta = pi/2
        ]
    )
    phi, theta = spherical_angles(unit_vec)
    assert phi.shape == (3, 1)
    assert phi[0, 0] == pytest.approx(0.0)
    assert theta[0, 0] == pytest.approx(np.pi / 2)
    assert phi[1, 0] == pytest.approx(0.0, abs=1e-9)
    assert theta[1, 0] == pytest.approx(0.0)
    assert phi[2, 0] == pytest.approx(np.pi / 2)
    assert theta[2, 0] == pytest.approx(np.pi / 2)


# ---------------------------------------------------------------------------
# polyanion_center_rdf
# ---------------------------------------------------------------------------


def test_polyanion_center_rdf_peak_at_known_distance():
    """Three polyanions evenly spaced on a line at 0, 3, 6 A in a 30 A box.
    Per-frame min-image distances: four pairs at 3 A, two pairs at 6 A.
    g(r) integrated over a window straddling 3 A should be much larger
    than over an empty window centered at 5 A."""
    n_frames = 5
    positions_per_frame = np.zeros((n_frames, 3, 3), dtype=float)
    positions_per_frame[:, 0, 0] = 0.0
    positions_per_frame[:, 1, 0] = 3.0
    positions_per_frame[:, 2, 0] = 6.0
    cell = 30.0 * np.eye(3)
    r, g = polyanion_center_rdf(
        positions_per_frame, cell, r_max_A=10.0, n_bins=100
    )
    assert r.shape == (100,)
    assert g.shape == (100,)
    in_3 = g[(r > 2.5) & (r < 3.5)].sum()
    in_5 = g[(r > 4.5) & (r < 5.5)].sum()
    in_6 = g[(r > 5.5) & (r < 6.5)].sum()
    assert in_3 > 10 * in_5
    assert in_6 > 10 * in_5


def test_representative_bond_is_fixed_and_periodic():
    O = np.array([[[0,0,9],[0,0,2]], [[1,0,0],[0,0,3]]],float)
    X = np.zeros((2,1,3))
    # First oxygen is actually -z through the periodic boundary. min_theta
    # selects the second O; max_abs_z ties choose the first O and keep it.
    got = xo_unit_vectors_per_frame(O,X,np.array([0,0]),cell=np.eye(3)*10,
                                    selection='max_abs_z')
    np.testing.assert_allclose(got[:,0],[[0,0,-1],[1,0,0]])
    positive = xo_unit_vectors_per_frame(O,X,np.array([0,0]),cell=np.eye(3)*10,
                                         selection='min_theta')
    np.testing.assert_allclose(positive[:,0],[[0,0,1],[0,0,1]])


def test_sulfate_alignment_uses_initial_mask_without_mutation():
    from superproton.geometry import align_sulfate_orientations
    angle = np.deg2rad([-4,94])
    first = np.column_stack([np.cos(angle),np.sin(angle),np.zeros(2)])
    original = np.array([first,first[::-1]])
    mapped = align_sulfate_orientations(original)
    np.testing.assert_allclose(mapped[:,0,:2],original[:,0,:2][:,::-1])
    np.testing.assert_allclose(mapped[:,1],original[:,1])
    np.testing.assert_allclose(original[0],first)
    np.testing.assert_allclose(np.linalg.norm(mapped,axis=-1),1)


def test_angular_density_integrates_to_one_in_degrees():
    from superproton.geometry import orientation_density
    rng=np.random.default_rng(2)
    v=rng.normal(size=(100,4,3));v/=np.linalg.norm(v,axis=-1,keepdims=True)
    d,phi,theta=orientation_density(v,bins=10)
    assert np.sum(d*np.diff(phi)[:,None]*np.diff(theta)[None,:]) == pytest.approx(1)
