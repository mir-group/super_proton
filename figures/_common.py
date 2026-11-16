"""Shared path resolution for the figure scripts.

Set SUPERPROTON_REFERENCE_DIR and SUPERPROTON_FIGURE_DIR to override the
default directories without editing a plotter.

Every script in this directory reads its inputs from ``data/reference/<topic>/``
and writes its output PNG to ``figures/output/``. Both are resolved relative to
the repository root, so a script runs correctly from any working directory:

.. code-block:: bash

    python figures/plot_msd.py          # from the repo root
    cd figures && python plot_msd.py    # or from here

``data/reference/`` holds the published intermediate values -- the numbers
behind the manuscript figures. See ``data/README.md`` for their provenance and
``docs/paper_mapping.md`` for which script draws which figure.
"""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = Path(os.environ.get("SUPERPROTON_REFERENCE_DIR", REPO_ROOT / "data" / "reference"))
OUTPUT_DIR = Path(os.environ.get("SUPERPROTON_FIGURE_DIR", REPO_ROOT / "figures" / "output"))


def ref(topic: str) -> Path:
    """Directory holding the reference data for one analysis topic."""
    path = REFERENCE_DIR / topic
    if not path.is_dir():
        available = (
            ", ".join(sorted(p.name for p in REFERENCE_DIR.iterdir() if p.is_dir()))
            if REFERENCE_DIR.is_dir()
            else "<data/reference/ is missing>"
        )
        raise FileNotFoundError(
            f"No reference data for topic {topic!r} at {path}. Available: {available}"
        )
    return path


def out(name: str) -> Path:
    """Path to write a figure to, creating ``figures/output/`` on first use."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def load_csv_columns(path) -> dict[str, np.ndarray]:
    """A numeric CSV with one header row, as ``{column name: column array}``.

    The column names carry their units (``_ps``, ``_A2``, ``_cm2_s``), so
    keeping them as dict keys is what lets a script read
    ``d["MSD_H_A2"]`` without a separate schema. Scripts that want a DataFrame
    use ``pandas.read_csv`` directly instead.
    """
    with Path(path).open() as fh:
        names = fh.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return {name: data[:, i] for i, name in enumerate(names)}


def _is_material_input(missing: str, material: str, search_root: Path) -> bool:
    """Whether a missing path looks like this material's own input data.

    Requires both that the file sits under the directory inputs are read from
    and that its name carries the material tag, so an unrelated missing file
    is not mistaken for absent data.
    """
    path = Path(missing)
    if material not in path.name:
        return False
    try:
        path.resolve().relative_to(search_root.resolve())
    except ValueError:
        return False
    return True


def each_material(materials, draw, *, what="figures", where=None):
    """Call ``draw`` for each material, skipping any whose input files are absent.

    The figure scripts each cover both materials, but a user reproducing one
    material -- ``run_analysis.py --material CDP`` feeding
    ``make_residence_csv.py`` -- has only half the inputs. Without this the
    script dies on ``FileNotFoundError`` after correctly drawing the first
    figure, which reads as a failure when it is not.

    ``materials`` is either a sequence of material names or a mapping of
    material to temperature; ``draw`` is called with the name, plus the
    temperature when one is available.

    ``what`` and ``where`` only shape the message raised when nothing at all
    could be produced -- "figures" under ``data/reference/`` for the plot
    scripts, "CSVs" under a caller-supplied input directory for the ``make_*``
    generators.

    Only a missing *input data file for this material* counts as "absent"; any
    other ``FileNotFoundError`` propagates. Otherwise a typo in a filename deep
    inside ``draw`` would be reported as missing data and, if the other material
    happened to succeed, the script would exit 0 having silently drawn half the
    figures.
    """
    is_mapping = hasattr(materials, "items")
    search_root = Path(where) if where else REFERENCE_DIR
    drawn = 0
    for material in materials:
        try:
            if is_mapping:
                draw(material, materials[material])
            else:
                draw(material)
        except FileNotFoundError as exc:
            missing = getattr(exc, "filename", None)
            if missing is None or not _is_material_input(missing, material, search_root):
                raise
            print(f"skipping {material}: no input data ({missing})")
            continue
        drawn += 1
    if drawn == 0:
        raise SystemExit(
            f"No {what} produced: none of the expected input files were found. "
            f"Looked under {where or REFERENCE_DIR}. See data/README.md."
        )
