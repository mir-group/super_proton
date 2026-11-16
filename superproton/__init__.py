"""superproton: analysis pipeline for the proton slingshot mechanism in
CsH2PO4 and CsHSO4 from ML molecular dynamics.

This package is the companion code to the JACS manuscript on solid-acid proton
transport. Each submodule maps to one part of the analysis pipeline.
``scripts/run_analysis.py`` drives them from a trajectory to the published
numbers, ``figures/`` redraws each manuscript figure from the values already in
``data/reference/``, and ``notebooks/superproton-tutorial.ipynb`` contains six runnable tutorial
sections using the shared Analysis interface and scientific modules.

Submodules
----------
io
    LAMMPS dump reading and ovito pipeline wrappers.
tracking
    Per-proton nearest-O and nearest-P/S neighbor tracking over a trajectory.
rotation
    Productive vs nonproductive polyanion-rotation classification and rotation
    period detection.
residence
    Residence-time histograms and single-exponential, bi-exponential, and
    piecewise rate fits, with Arrhenius assembly across temperatures.
coordination
    Bound-proton counting ``N_p(t)`` and the joint density with the O-pair
    residence time.
reorientation
    dH/dO displacement ratios, baseline and outlier metrics, and the cos(H, O)
    direction diagnostic.
osharing
    Oxygen-sharing fraction vs time to rotation completion (CsH2PO4 only).
msd
    Per-species mean-square displacement and tracer diffusivity.
hopping
    Long-range proton hop distances and per-role polyanion-oxygen
    displacements over a rotation period.
pointing
    Hydrogen-bond pointing attempts and commit / revert / superseded outcomes.
correlation
    NP-P temporal overlap by role and spatial-temporal event-pair
    correlation ``G_event(r, dt)``.
geometry
    Covalent X-O-H geometry, polyanion vector autocorrelation, orientation
    distributions, and polyanion-center radial distribution functions.
published
    The values the manuscript reports, for checking results against.
"""

__version__ = "0.1.0"

__all__ = [
    "io",
    "tracking",
    "rotation",
    "residence",
    "coordination",
    "msd",
    "hopping",
    "reorientation",
    "osharing",
    "pointing",
    "correlation",
    "geometry",
    "published",
    "workflow",
    "plots",
]
