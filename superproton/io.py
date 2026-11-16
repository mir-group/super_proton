"""LAMMPS dump reading and ovito pipeline helpers.

A thin wrapper around ``ovito.io.import_file`` that exposes the few quantities
every downstream analysis needs: the pipeline itself, frame count, simulation
cell, and a type-id to element-symbol mapping derived from the known
composition of the cesium solid-acid phases.

The paper trajectories use type IDs 1 = Cs, 2 = H, 3 = P (CDP) or S (CHS),
and 4 = O. Explicit AtomMapping selections support other type orderings.
Composition-based inference remains available for the paper presets, but
cannot distinguish equal-population species by count.

Public surface
--------------
read_lammps_dump(path) -> Pipeline
species_map(pipeline, material) -> dict[int, str]
cell(pipeline, frame=0) -> np.ndarray  # 3x3 matrix
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
import zipfile
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ovito.pipeline import Pipeline

# Expected per-formula-unit composition. Used to infer the type-id -> element
# mapping by counting atom populations of each type in the first frame.
COMPOSITIONS: Mapping[str, Mapping[str, int]] = {
    "CDP": {"Cs": 1, "H": 2, "P": 1, "O": 4},  # CsH2PO4
    "CHS": {"Cs": 1, "H": 1, "S": 1, "O": 4},  # CsHSO4
}


# Everything that is not the cation, the proton or an oxygen is the polyanion
# center: P in CsH2PO4, S in CsHSO4.
_NON_POLYANION = frozenset({"Cs", "H", "O"})


def cached_arrays(
    cache_dir: str | Path | None,
    *,
    source: str | Path,
    parameters: Mapping[str, object],
    compute: Callable[[], dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Cache exact arrays in compressed NPZ files, without pickle.

    Callers must include all analysis options and a method version in
    ``parameters``. Source identity uses its resolved path, size, modification
    time and change time; changing either source or options invalidates a hit.
    ``None`` disables the cache. Invalid cache files warn and are recomputed.
    """
    if cache_dir is None:
        return compute()
    source = Path(source).resolve()

    def identity():
        stat = source.stat()
        return {
            "path": str(source), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
        }

    def validate(arrays):
        for key, value in arrays.items():
            if not isinstance(key, str) or key in {"__metadata__", "file"}:
                raise ValueError("Cache array names must be strings other than __metadata__ or file")
            if not isinstance(value, np.ndarray) or value.dtype.hasobject:
                raise ValueError("Cached values must be NumPy arrays without object dtype")
        return arrays

    original_identity = identity()
    metadata = json.dumps(
        {"cache_version": 1, "source": original_identity, "parameters": parameters},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (hashlib.sha256(metadata.encode()).hexdigest() + ".npz")
    if target.exists():
        try:
            with np.load(target, allow_pickle=False) as archive:
                if archive["__metadata__"].item() != metadata:
                    raise ValueError("cache metadata mismatch")
                result = validate({key: archive[key] for key in archive.files
                                   if key != "__metadata__"})
        except PermissionError:
            raise
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, zlib.error) as exc:
            warnings.warn(f"Ignoring invalid array cache {target}: {exc}",
                          RuntimeWarning, stacklevel=2)
        else:
            if identity() != original_identity:
                raise RuntimeError("Source changed while loading cached arrays")
            return result
    result = validate(compute())
    if identity() != original_identity:
        raise RuntimeError("Source changed during calculation; arrays were not cached")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, __metadata__=np.array(metadata), **result)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result


@dataclass(frozen=True)
class AtomMapping:
    """Explicit LAMMPS type selections and optional fixed O-ID → center-ID map.

    One particle type selects each role. Unselected species are spectators.
    If ``oxygen_parents`` is omitted, the existing four-nearest-O assignment
    is used and checked for unique, complete membership. IDs are persistent
    LAMMPS identifiers, never row numbers. No stoichiometry inference is used.
    """

    hydrogen_type: int
    oxygen_type: int
    center_type: int
    oxygen_parents: dict[int, int] | None = None

    def __post_init__(self):
        values = (self.hydrogen_type, self.oxygen_type, self.center_type)
        if len(set(values)) != 3 or any(type(v) is not int or v <= 0 for v in values):
            raise ValueError("H, O and center require distinct positive integer types")

    @classmethod
    def from_json(cls, path):
        """Read an explicit mapping, rejecting unknown configuration fields."""
        import json
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if values.get("oxygen_parents") is not None:
            values["oxygen_parents"] = {
                int(k): int(v) for k, v in values["oxygen_parents"].items()
            }
        return cls(**values)


def min_image(diff: np.ndarray, cell: np.ndarray, inv_cell: np.ndarray | None = None) -> np.ndarray:
    """Fold separation vectors into the minimum image of a periodic cell.

    ``diff`` is any array of Cartesian separations with the last axis of length
    3; ``cell`` is the 3x3 row-vector lattice from :func:`cell`. Pass
    ``inv_cell`` to avoid re-inverting inside a loop.

    Correct for intramolecular distances and pair separations. It is NOT
    appropriate for a displacement that is genuinely allowed to exceed half the
    box, such as a proton's total travel over a long trajectory -- see
    :func:`superproton.hopping.hop_distances`, which deliberately does not fold.
    """
    if inv_cell is None:
        inv_cell = np.linalg.inv(cell)
    frac = diff @ inv_cell
    frac -= np.round(frac)
    return frac @ cell


def pairwise_min_image_distances(
    positions: np.ndarray, cell: np.ndarray, inv_cell: np.ndarray | None = None
) -> np.ndarray:
    """Full ``(n, n)`` minimum-image distance matrix for one frame of points.

    ``positions`` is ``(n, 3)`` Cartesian. Row ``i`` holds the distance from
    point ``i`` to every point, its own zero included; callers that want only
    genuine pairs mask the diagonal.

    Loops over rows rather than broadcasting an ``(n, n, 3)`` difference
    array, which for the polyanion counts here (125-192 centers, thousands of
    frames) keeps the working set small without costing measurable time.
    """
    if inv_cell is None:
        inv_cell = np.linalg.inv(cell)
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        out[i] = np.linalg.norm(min_image(pos - pos[i], cell, inv_cell), axis=1)
    return out


def polyanion_element(material: str | AtomMapping) -> str:
    """Element at the center of the polyanion: ``"P"`` for CDP, ``"S"`` for CHS.

    Derived from :data:`COMPOSITIONS` so the mapping lives in one place, and so
    an unknown material raises here rather than silently defaulting to sulfur.
    """
    if isinstance(material, AtomMapping):
        return "X"
    try:
        composition = COMPOSITIONS[material]
    except KeyError:
        raise KeyError(
            f"Unknown material {material!r}. Choices: {sorted(COMPOSITIONS)}"
        ) from None
    centers = set(composition) - _NON_POLYANION
    if len(centers) != 1:
        raise ValueError(
            f"{material} composition {dict(composition)} does not have exactly one "
            f"polyanion center; found {sorted(centers)}."
        )
    return centers.pop()


def read_lammps_dump(path: str | Path) -> Pipeline:
    """Open a LAMMPS dump trajectory and return an ovito ``Pipeline``.

    Wraps ``ovito.io.import_file`` with ``multiple_frames=True`` so the entire
    trajectory becomes addressable via ``pipeline.compute(frame)`` calls.

    Parameters
    ----------
    path
        Path to a LAMMPS dump file (text or binary). Must exist and be
        readable by ovito.

    Returns
    -------
    pipeline
        An ovito ``Pipeline`` whose ``source.num_frames`` reports the number
        of trajectory frames.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"LAMMPS dump not found: {p}")
    _warn_if_wrapped(p)
    from ovito.io import import_file  # local import keeps the module importable without ovito
    return import_file(str(p), multiple_frames=True, sort_particles=True)


def _warn_if_wrapped(path: Path) -> None:
    """Warn once if a text dump stores wrapped coordinates.

    Most analyses here take displacements between frames without a
    minimum-image correction, which is correct only for unwrapped coordinates
    (``xu yu zu``). With wrapped ones (``x y z``) a single periodic crossing
    turns a 0.05 A step into a box-length jump, which silently corrupts the
    MSD, the hop distances and the dH/dO ratio rather than raising. The
    production dumps are unwrapped; this catches a foreign dump that is not.

    A warning rather than an error: the polyanion RDF and the X-O-H geometry
    apply minimum-image internally and are fine either way.
    """
    import warnings

    try:
        with path.open("r", errors="ignore") as fh:
            for _ in range(64):
                line = fh.readline()
                if not line:
                    return
                if line.startswith("ITEM: ATOMS"):
                    columns = line.split()[2:]
                    if not {"xu", "yu", "zu"} <= set(columns):
                        warnings.warn(
                            f"{path.name} stores coordinates as {' '.join(columns)}, "
                            "which look wrapped. Analyses that take inter-frame "
                            "displacements (msd, hopping, reorientation) require "
                            "unwrapped 'xu yu zu' and will be wrong on this dump.",
                            RuntimeWarning,
                            stacklevel=3,
                        )
                    return
    except OSError:
        return  # binary or unreadable header: leave it to ovito


def cell(pipeline: Pipeline, frame: int = 0) -> np.ndarray:
    """Return the simulation cell as a 3x3 numpy array (rows are lattice vectors).

    For the orthorhombic supercells used in the manuscript the off-diagonal
    entries are zero. The cell is read from the requested frame so trajectories
    with cell drift (NPT) report the correct cell at that frame.
    """
    data = pipeline.compute(frame)
    matrix = np.asarray(data.cell.matrix, dtype=float)
    # ovito's matrix is 3x4 (lattice vectors as columns, last column = origin).
    # Return the 3x3 lattice block in row-vector form to match ase conventions.
    return matrix[:, :3].T.copy()


def species_map(pipeline: Pipeline, material: str | AtomMapping, frame: int = 0) -> dict[int, str]:
    """Infer ``type_id -> element_symbol`` from atom counts at ``frame``.

    Without an AtomMapping, use the legacy composition-based inference.
    Equal-population species are ambiguous under arbitrary type permutations;
    use explicit mappings for external trajectories.

    The LAMMPS dump format does not carry element labels, only integer type
    ids. We assign each type id by matching its population to the expected
    composition of the chosen material. ``KeyError`` is raised if ``material``
    is unknown; ``ValueError`` is raised if the populations do not factor
    cleanly into the expected formula.

    Parameters
    ----------
    pipeline
        Pipeline returned by :func:`read_lammps_dump`.
    material
        A CDP/CHS preset or an explicit :class:`AtomMapping`. Explicit
        mappings bypass composition inference and support changed type labels.
    frame
        Frame index used for the population count. Defaults to 0.

    Returns
    -------
    dict[int, str]
        Mapping from ovito particle type ids (1-based) to element symbols.
        For CDP the typical result is ``{1: "Cs", 2: "H", 3: "P", 4: "O"}``.

    Notes
    -----
    The mapping is determined solely by atom counts in a single frame, which
    is sufficient because the trajectory's particle ids and types are fixed
    across frames in every LAMMPS dump we ingest.
    """
    if isinstance(material, AtomMapping):
        selected = {material.hydrogen_type: "H", material.oxygen_type: "O",
                    material.center_type: "X"}
        present = set(np.asarray(pipeline.compute(frame).particles.particle_types))
        if not set(selected) <= present:
            raise ValueError("Selected H, O or center type is missing from this frame")
        return selected
    if material not in COMPOSITIONS:
        raise KeyError(
            f"Unknown material {material!r}. Choices: {sorted(COMPOSITIONS)}"
        )
    composition = COMPOSITIONS[material]

    data = pipeline.compute(frame)
    types = np.array(data.particles.particle_types).astype(int)
    type_ids, counts = np.unique(types, return_counts=True)

    n_total = int(counts.sum())
    formula_size = sum(composition.values())
    if n_total % formula_size != 0:
        raise ValueError(
            f"Atom count {n_total} not divisible by {material} formula size "
            f"{formula_size}; cannot infer species map."
        )
    n_units = n_total // formula_size

    expected = {element: count * n_units for element, count in composition.items()}
    # Greedy match: assign each type id to the element whose expected count it matches.
    remaining = dict(expected)
    mapping: dict[int, str] = {}
    for tid, c in zip(type_ids.tolist(), counts.tolist(), strict=True):
        match = next((el for el, n in remaining.items() if n == c), None)
        if match is None:
            raise ValueError(
                f"Could not match type {tid} (count {c}) to any element in "
                f"{material} composition {expected}."
            )
        mapping[tid] = match
        remaining.pop(match)

    return mapping


def check_stable_particle_order(
    pipeline: Pipeline, frame_indices: "np.ndarray | list[int]"
) -> None:
    """Verify that particle ordering does not change across the given frames.

    Every per-frame reader in this package resolves atom indices **once**, from
    the first frame, and then uses those integer positions to slice every later
    frame. That is valid only if the dump lists atoms in the same order in every
    frame.

    LAMMPS does not guarantee that. ``dump`` writes atoms in whatever order they
    sit in the processor's local arrays, which changes as atoms migrate between
    domains, unless the run used ``dump_modify <id> sort id``. The production
    trajectories for this paper did, and so do the Zenodo files; a dump that did
    not would otherwise produce silently wrong donor and acceptor assignments
    rather than an error.

    Checks the first, middle and last of ``frame_indices``, which costs three
    ``pipeline.compute`` calls and catches any reordering in practice.

    Raises
    ------
    ValueError
        If particle identifiers differ between the sampled frames.
    """
    idx = np.asarray(frame_indices, dtype=int)
    if idx.size < 2:
        return
    probes = sorted({int(idx[0]), int(idx[idx.size // 2]), int(idx[-1])})
    reference = np.array(pipeline.compute(probes[0]).particles.identifiers).astype(int)
    for frame in probes[1:]:
        other = np.array(pipeline.compute(frame).particles.identifiers).astype(int)
        if other.shape != reference.shape or not np.array_equal(other, reference):
            n_moved = (
                int((other != reference).sum())
                if other.shape == reference.shape
                else other.size
            )
            raise ValueError(
                f"Particle ordering changes between frame {probes[0]} and frame "
                f"{frame} ({n_moved} positions differ). This package indexes every "
                "frame by the first frame's atom order, so the trajectory must be "
                "written with a stable order. Re-dump with "
                "'dump_modify <dump-id> sort id', or re-run the analysis on a "
                "sorted copy of the trajectory."
            )


def validated_pipeline(path):
    """Open a sorted dump and validate fixed orthorhombic input at every read.

    This stricter tutorial entry point requires unwrapped text coordinates,
    persistent positive IDs/types, uniform timesteps and full periodicity.
    Legacy readers remain available for other analyses. The supplied time
    conversion remains the caller's responsibility (LAMMPS units vary).
    """
    path = Path(path)
    with path.open() as stream:
        for line in stream:
            if line.startswith('ITEM: ATOMS'):
                if not {'id','type','xu','yu','zu'} <= set(line.split()[2:]):
                    raise ValueError('Tutorial workflows require id type xu yu zu columns')
                break
        else: raise ValueError('Missing LAMMPS ATOMS header')
    pipe = read_lammps_dump(path)
    first = pipe.compute(0)
    ids = np.array(first.particles.identifiers)
    types = np.array(first.particles.particle_types)
    matrix = np.array(first.cell.matrix)
    if len(np.unique(ids)) != len(ids) or np.any(ids <= 0):
        raise ValueError('Atom IDs must be unique and positive')
    if not all(first.cell.pbc) or not np.allclose(matrix[:,:3],np.diag(np.diag(matrix[:,:3]))):
        raise ValueError('Initial release requires a periodic orthorhombic cell')
    initial = int(first.attributes['Timestep'])
    delta = int(pipe.compute(1).attributes['Timestep'])-initial if pipe.source.num_frames > 1 else 1
    if delta <= 0: raise ValueError('Timesteps must increase')
    def validate(frame, data):
        if not np.array_equal(data.particles.identifiers,ids) or not np.array_equal(data.particles.particle_types,types):
            raise ValueError(f'Atom IDs/types changed at frame {frame}')
        if not all(data.cell.pbc) or not np.allclose(data.cell.matrix,matrix,rtol=1e-10,atol=1e-10):
            raise ValueError(f'Cell changed at frame {frame}')
        if int(data.attributes['Timestep']) != initial+frame*delta:
            raise ValueError(f'Nonuniform timestep at frame {frame}')
    pipe.modifiers.append(validate)
    return pipe
