# Paper figure to code mapping

Find the plotting script and analysis module for each paper figure below.
Scripts are in `figures/`; reference topics are folders under `data/reference/`.

## Main text

| Figure | Numerical plotters | Reference topics | Analysis modules / notes |
|---|---|---|---|
| 1 | Structure illustration; no numerical plot | — | `export_structure.py` exports MD configurations, not the experimental disorder artwork |
| 2b–c | `plot_msd.py`, `plot_diffusion.py` | `msd`, `diffusion` | `msd` |
| 3c–e | `plot_pair_tracking.py` | `pair_tracking` | `tracking.candidate_trace`, `plots.candidate_tracking`; tutorial H268 at 525 K |
| 4a,b,d | `plot_residence_time.py`, `plot_rotation_rate.py`, `plot_coordination.py` | `residence`, `rotation_rates`, `coordination` | `residence`, `coordination`; Fig. 4d unweighted short-residence CDP mean ≈1.72, compared with the paper annotation ≈1.8 |
| 5b | `plot_dHdO_ratio.py` | `dhdo_ratio` | `reorientation` |
| 6c | `plot_O_sharing.py` | `o_sharing` | `osharing`; oxygen-neighbor selection differs from the published calculation. Use reference data to redraw the published curve. |
| 7a–d | `plot_residence_time.py`, `plot_coordination.py`, `plot_rotation_rate.py`, `plot_dHdO_ratio.py` | `residence`, `coordination`, `rotation_rates`, `dhdo_ratio` | `residence`, `coordination`, `reorientation` |
| 8b,c,e,f | `plot_orientation.py` | `orientation` | `geometry` (including SO₄ alignment) |
| 8d | `plot_vcf.py` | `vcf` | `geometry`; source frames `[200:30200:10]`, sampled every 0.5 ps |

Manual assets: Fig. 1, 2a, 3a–b, 4c, 5a, 6a–b, 8a and TOC.

For Fig. 5b/7d, retain the analysis windows, event denominators and bin edges;
recomputed distributions are not identical to the archived curves in every bin.

## Supporting Information

| Figure/table | Plotters | Reference topics | Analysis modules / notes |
|---|---|---|---|
| S1 | `plot_force_parity.py` | `model_validation` | Training/DFT outside post-MD scope |
| S2 | `plot_eos.py` | `model_validation` | Training/DFT outside post-MD scope |
| S3 | `plot_PO_auto_corr.py` | `finite_size` | Archived model comparison |
| S4 | `plot_thermal_expansion.py` | `thermal_expansion` | `msd` |
| S5 | `plot_diffusion.py` | `diffusion` | `msd` |
| S6 | `plot_msd.py` | `msd` | `msd` |
| S7 | `plot_rotation_period.py` | `rotation_period` | `rotation` |
| S8–S9 | `plot_event_correlation.py` | `correlation` | `correlation.g_event`; changing the time origin can shift bin-edge counts through floating-point roundoff |
| S10 | `plot_residence_comparison.py` | `residence_comparison` | `residence.residence_fit_comparison` |
| S11 | `plot_residence_categories.py` | `residence_categories` | `residence.collect_taus` |
| S12 | `plot_nonproductive.py`, `plot_nonproductive_single.py` | `nonproductive` | `residence` |
| S13 | `plot_rotation_rate.py` (the two `_avg` outputs) | `rotation_rates` | `residence.global_average_rate` |
| S14 | `plot_hop_distance.py` | `hop_distance` | `hopping` |
| S15 | `plot_dHdO_vs_time.py` | `dhdo_vs_time` | `reorientation.reorientation_samples` |
| S16 | `plot_cos_angle.py` | `cos_HO` | `reorientation.pretransfer_direction_samples` |
| S17a | `plot_pair_tracking.py` | `pair_tracking` | `tracking.candidate_trace`; tutorial H342 at 500 K; S17b is a manual snapshot |
| S18 | `plot_attempt_outcome.py` | `pointing` | `read_pointing_hosts`, `pointing_outcomes`, `pointing_summary`; tutorial `plots.pointing_comparison` compares CDP and CHS |
| S19 | `plot_committed_vs_productive.py` | `pointing` | Committed outcomes from `pointing_outcomes` and productive pair switches; divide counts by the same proton count and observation time |
| S20 | `plot_pointing_lookahead.py` | `pointing` | `read_pointing_hosts`, `episode_crossing_lookahead` |
| S21 | `plot_rotation_distance.py` | `rotation_distance` | `hopping`; events without a valid carrier oxygen are excluded, so counts differ from the archived reference |
| S22 | `plot_rdf.py` | `rdf` | `geometry` |
| Tables S2–S3 | Tables, no plot | `overlap` | `correlation.temporal_overlap` |
| Table S6 | Table, no plot | — | `geometry.xoh_geometry` |

## Plot from reference data

Run from the repository root:

```bash
python figures/plot_msd.py                    # one script
for f in figures/plot_*.py; do python "$f"; done  # all plotters
```

Inputs default to `data/reference/<topic>/` and PNGs to `figures/output/`.
To plot your own results, set `SUPERPROTON_REFERENCE_DIR=results/reference`
and keep topic subfolders such as `results/reference/residence/`. Set
`SUPERPROTON_FIGURE_DIR` to choose a different PNG output directory.

## Recalculate from trajectories

Obtain the matching complete source trajectories from the [data guide](../data/README.md)
and use each figure's original temperature, analysis window, sampling and fitting
settings. Selected analysis commands are shown below; the tables above identify
the library functions for other quantities. Each call processes one temperature.
Output folders mirror the reference topics under a separate `results/reference/`.

```bash
python scripts/run_analysis.py --analysis orientation --dump data/zenodo/CDP_550K_4ns.trj \
  --material CDP --temperature 550 --equilibration-ps 0 --end-frame 80000 \
  --out results/reference/orientation
python scripts/run_analysis.py --analysis coordination --dump data/zenodo/CDP_525K_4ns.trj \
  --material CDP --temperature 525 --out results/reference/coordination
python scripts/run_analysis.py --analysis event_correlation --dump data/zenodo/CDP_525K_4ns.trj \
  --material CDP --temperature 525 --out results/reference/correlation
python scripts/run_analysis.py --analysis residence_comparison --dump data/zenodo/CDP_525K_4ns.trj \
  --material CDP --temperature 525 --out results/reference/residence_comparison
python scripts/run_analysis.py --analysis residence_categories --dump data/zenodo/CDP_525K_4ns.trj \
  --material CDP --temperature 525 --out results/reference/residence_categories
python scripts/run_analysis.py --analysis pointing_lookahead --dump data/zenodo/CDP_525K_4ns.trj \
  --material CDP --temperature 525 --out results/reference/pointing
python scripts/run_analysis.py --analysis rotation_period --dump data/zenodo/CDP_525K_4ns.trj \
  --material CDP --temperature 525 --out results/reference/rotation_period
SUPERPROTON_REFERENCE_DIR=results/reference python figures/plot_coordination.py
```

Repeat orientation for CHS at 550 K; coordination/correlation/rotation-period
for CHS at 450 K; the comparison for CDP at 550 and 575 K; and pointing for
each temperature shown. Orientation is selected/mapped at trajectory frame
zero, with the histogram using frames `[0,80000)`, 100×100 bins spanning
the sampled angular range. The VCF uses the separate window listed for Fig. 8d.
Other analyses use the default 10 ps equilibration cut. `--orientation-index`
is the zero-based polyanion column (default 23).

### Residence-time distributions from a trajectory

Residence analysis exports per-event times. Generate the binned distributions
and fitted curves before plotting:

```bash
python scripts/run_analysis.py --analysis residence --dump data/zenodo/CDP_525K_4ns.trj \
  --material CDP --temperature 525 --out results/raw
python figures/make_residence_csv.py --raw results/raw \
  --out results/reference/residence --material CDP
SUPERPROTON_REFERENCE_DIR=results/reference python figures/plot_residence_time.py
```

For S12, `figures/make_nonproductive_csv.py` converts archived or library-generated
nonproductive-count tables into plot inputs. Both `make_*` generators accept
`--raw` and require `--out`; store their outputs under the matching topic folder.

For diffusivity, set and record the MSD lag-time fit window with
`msd.diffusivity(..., fit_range_ps=(start, end))` or
`--msd-fit-range-ps START END` (ps). Check `DiffusivityFit.loglog_slope` (near 1
in the diffusive regime) and `is_diffusive` before interpreting the fitted value.
