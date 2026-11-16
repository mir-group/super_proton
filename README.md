# superproton

Python toolkit for proton transport and polyanion dynamics in CsH<sub>2</sub>PO<sub>4</sub> and CsHSO<sub>4</sub>, with a six-part tutorial and figure-reproduction workflows for

> M. Wang, J. Ding, G. Xiong, N. Zhan, C. Owen, A. Musaelian, Y. Xie, N. Molinari, R. Adams, B. Kozinsky, S. Haile. *Revealing the Proton Slingshot Mechanism in Solid Acid Electrolytes through Machine-Learning Molecular Dynamics.* *Journal of the American Chemical Society* **148** (40), 42913–42927 (2026). [DOI: 10.1021/jacs.5c21965](https://doi.org/10.1021/jacs.5c21965).

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/mir-group/super_proton.git
cd super_proton
uv venv --python 3.11 --managed-python
source .venv/bin/activate
uv pip install -e ".[tutorial]"
```

This creates a separate Python environment with the scientific dependencies,
OVITO, Jupyter and the local package. All commands below run from the repository
root with this environment activated. Conda users can use
[`environment.yml`](environment.yml) instead.

## Learn the core methods with the tutorial

The [six-part notebook](notebooks/superproton-tutorial.ipynb) teaches the paper's
core analysis methods and how to use the code. By default, it uses the bundled
**75 ps** CDP and CHS trajectories for short demonstrations, rather than
reproducing the published results.

1. Distinguish rattling, rotation and productive rotation; examine CHS pointing before hopping.
2. Classify productive/nonproductive rotations and estimate residence-time rates.
3. Relate bound-proton coordination to rotational dynamics.
4. Measure spatial–temporal event correlations and temporal overlaps.
5. Measure rotation displacement and O–H reorientation underlying the slingshot.
6. Map orientation probability landscapes and vector autocorrelation.

```bash
jupyter lab notebooks/superproton-tutorial.ipynb
```

To recalculate the paper's results from trajectories, download the matching
[full production trajectories](data/README.md#full-trajectories-zenodo) and follow
the [figure map](docs/paper_mapping.md) with the original analysis settings.

Use `superproton.workflow.Analysis` for the same analyses in Python. For other
compatible solid-acid trajectories, supply an explicit `AtomMapping`; the paper
CLI provides CDP and CHS presets. The notebook includes input requirements,
atom mappings and examples for your own trajectories.

## Reproduce the paper's figures

**Redraw from reference data.** The numerical inputs in `data/reference/` are
included in the repository; no trajectory download is needed.

```bash
python figures/plot_residence_time.py     # Figures 4a and 7a
python figures/plot_msd.py                # Figure 2 MSD panels
for f in figures/plot_*.py; do python "$f"; done   # all plotters
```

Plots are written to `figures/output/`. See the [figure map](docs/paper_mapping.md)
for scripts, inputs and reproduction commands.

**Recalculate from production trajectories.** Download the matching full trajectory
from [Zenodo](https://zenodo.org/records/14649023) following the [data guide](data/README.md).
For example, with `CDP_525K_4ns.trj` decompressed in `data/zenodo/`:

```bash
python scripts/run_analysis.py --analysis residence xoh \
    --dump data/zenodo/CDP_525K_4ns.trj --material CDP --temperature 525 --compare
```

Results are written to `results/`; `--compare` prints available paper values
alongside them. Run `python scripts/run_analysis.py --help` for other analyses
and options. These command-line scripts run from the checkout.

Reproducing a panel requires its matching trajectory, temperature, window and
settings. This repository covers post-MD analysis and plotting;
force-field training, DFT calculations and MD simulations are outside its scope.

The [agent skill](skills/superproton-analysis/SKILL.md) provides instructions
for AI agents to use Superproton and interpret analysis results.

## Tests

```bash
uv pip install -e ".[tutorial,dev]"
pytest
```

Production-trajectory tests are skipped unless their input paths are configured.
To enable them, set the paths to the matching 4 ns trajectories:

```bash
export SUPERPROTON_CDP_525K_TRJ=/path/to/CDP_525K_4ns.trj
export SUPERPROTON_CHS_450K_TRJ=/path/to/CHS_450K_4ns.trj
pytest tests/test_reference.py
```

## Citation

If you use these analysis methods, cite the JACS paper above and record the
software version or commit and analysis settings.
[CITATION.cff](CITATION.cff) provides the preferred citation.
Cite the [Zenodo dataset](https://doi.org/10.5281/zenodo.14649023) separately when
using the supplied trajectories or DFT data.

## License

The software is licensed under MIT; see [LICENSE](LICENSE).
