"""Export one frame of a LAMMPS dump as a structure file with real elements.

A LAMMPS dump carries integer type ids, not element symbols, so reading one
with a generic reader labels the atoms by atomic number instead of species --
a CsH2PO4 frame comes back as ``H125Be500He250Li125``. This script resolves the
type ids through :func:`superproton.io.species_map`, which infers them from the
known composition, and writes a correctly labelled structure.

Use it to recover the starting configuration of a production run (frame 0), or
any later frame, for visualisation or as an input to another calculation.

Examples
--------
The starting structure of the shipped CsH2PO4 example::

    python scripts/export_structure.py --dump data/cdp_525K_dump_1.trj \\
        --material CDP --out CDP_525K_frame0.vasp

The final frame of a production trajectory, as extended XYZ::

    python scripts/export_structure.py --dump CDP_525K_4ns.trj \\
        --material CDP --frame -1 --out CDP_525K_final.xyz

The output format follows the file extension, via ``ase.io.write``: ``.vasp``
for POSCAR, ``.xyz`` for extended XYZ, ``.cif``, and so on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dump", required=True, type=Path, help="LAMMPS dump trajectory.")
    p.add_argument("--material", required=True, choices=["CDP", "CHS"])
    p.add_argument("--frame", type=int, default=0,
                   help="Frame index to export; negative counts from the end (default 0).")
    p.add_argument("--out", required=True, type=Path,
                   help="Output path. The extension selects the format.")
    args = p.parse_args(argv)

    if not args.dump.exists():
        print(f"Dump not found: {args.dump}", file=sys.stderr)
        return 1

    import numpy as np
    from ase import Atoms
    from ase.io import write

    from superproton.io import cell as read_cell
    from superproton.io import read_lammps_dump, species_map

    pipe = read_lammps_dump(args.dump)
    n_frames = int(pipe.source.num_frames)
    frame = args.frame if args.frame >= 0 else n_frames + args.frame
    if not 0 <= frame < n_frames:
        print(f"Frame {args.frame} out of range for {n_frames} frames.", file=sys.stderr)
        return 1

    # species_map infers type_id -> element from atom counts, so it is valid for
    # any type ordering; read it from the frame being exported.
    species = species_map(pipe, args.material, frame=frame)
    data = pipe.compute(frame)
    # ovito property views reject a dtype-casting asarray; copy then cast.
    types = np.array(data.particles.particle_types).astype(int)
    positions = np.array(data.particles.positions).astype(float)
    cell = read_cell(pipe, frame=frame)

    atoms = Atoms(
        symbols=[species[t] for t in types],
        positions=positions,
        cell=cell,
        pbc=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(args.out, atoms)
    print(f"frame {frame} of {n_frames} -> {args.out}")
    print(f"  {atoms.get_chemical_formula()}, cell {cell.diagonal().round(4).tolist()} A")
    print(f"  type ids: {dict(sorted(species.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
