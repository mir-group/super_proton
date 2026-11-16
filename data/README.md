# Data

## Example trajectories (in repo)

| File | Material | Length | Frames | Saved-frame interval |
|---|---|---|---|---|
| `cdp_525K_dump_1.trj` | CDP, 525 K | 75 ps | 1500 | 50 fs |
| `chs_450K_dump_1.trj` | CHS, 450 K | 75 ps | 1500 | 50 fs |

Each file contains the first 1,500 saved frames (0–74.95 ps) of a separate
1.5 ns run, without additional subsampling. These short inputs support the
[tutorial](../notebooks/superproton-tutorial.ipynb); slow-rate estimates require
longer trajectories. Dumps contain `id type xu yu zu`, with unwrapped coordinates
in Å and a saved-frame interval of 0.05 ps.

## Reference data (in repo)

`reference/<topic>/` contains numerical inputs for the paper's plots, including
experimental comparisons. The [figure map](../docs/paper_mapping.md) connects
each topic to its figure and plotting script. These plots need no download.

CSV column suffixes specify units: `_ps`, `_A`, `_A2`, `_K`, `_cm2_s`.
NPZ files contain numeric arrays and load with `np.load(path, allow_pickle=False)`.
Some arrays use a generic `P` key for polyanions: phosphate in CDP, sulfate in CHS.

| Topic | Key fields and conventions |
|---|---|
| `orientation` | `{mat}_550K_{single,all}.npz`: `density` (phi × theta, per degree²), `phi_edges_deg`, `theta_edges_deg`, marker angles. `{mat}_experimental_sites.csv`: O and X coordinates in Å. |
| `coordination` | `tau_ps`, `mean_Np`, optional `proton_index`. Complete O-pair intervals; strict reciprocal-neighbor O–H counts on the least-occupied visited host, fixed for each interval. |
| `correlation` | `shells_A`, `dt_centers_ps`, `dt_edges_ps`, `G`, `H_obs`, `H_exp`, `rate_per_ps`. Recipient events; r ≤ 15 Å, Δt ≤ 200 ps, 1 ps bins, expected count ≥ 5. |
| `residence_comparison` | `tau_ps`, `grid_ps`, `{piecewise,biexp}_{pdf,survival,parameters}`. Parameter order: λ₁, λ₂, τc, intercept (piecewise); amplitude, λ₁, λ₂ (mixture); rates in ps⁻¹. Uses the SI S10 comparison fit, distinct from the main-text rate fit. |
| `residence_categories` | `category`, `tau_ps`. NP–NP, NP–P and P–P label the two switches bounding an interval. |
| `pointing` | `{mat}_{T}K_lookahead.csv`: `material`, `temperature_K`, `lookahead_ps`, `fraction`. S20 measures extra time after episode end, including subsequent episodes; its denominator includes terminally censored fresh episodes. |

SO₄ orientation maps use a fixed initial-sublattice mapping: swap x and y for
the initial −4° azimuth sublattice at every frame. Representative bonds maximize
initial `abs(z)` and retain atom identity. Angular density has no solid-angle
Jacobian. See the figure map for the sampling window.

`overlap/` contains the role-specific NP–P interval-overlap counts for Tables
S2–S3. Summaries include `n_np_events`, unrounded percentages and `overlap_count`
bins `0`, `1`, `2`, `>=3`. Time intervals overlap when they intersect, including
shared endpoints.

## Full trajectories (Zenodo)

Download the matching production trajectory from the
[Zenodo dataset](https://zenodo.org/records/14649023). Its
[README](https://zenodo.org/records/14649023/preview/README.md?include_deleted=0)
documents available temperatures and lengths, simulation settings, file formats,
atom types and the DFT dataset.

From the repository root, create the download folder:

```bash
mkdir -p data/zenodo
```

Save the selected `.trj.gz` file there, then decompress it, for example:

```bash
gunzip -k data/zenodo/CDP_525K_4ns.trj.gz
```

`data/zenodo/` is gitignored. Follow the [figure map](../docs/paper_mapping.md)
for analysis commands and the matching trajectory, window and fit settings.

## Export a structure

Extract a frame with element labels for visualization or further analysis:

```bash
python scripts/export_structure.py --dump data/cdp_525K_dump_1.trj \
    --material CDP --frame 0 --out CDP_525K_frame0.vasp
```

The output extension selects `.vasp`, `.xyz` or `.cif`.
