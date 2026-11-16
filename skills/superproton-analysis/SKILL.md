---
name: superproton-analysis
description: Analyze local proton rattling, productive transfer, hydrogen-bond pointing, polyanion rotations, residence rates, coordination, event correlations and O–H reorientation in supported solid-acid LAMMPS trajectories using Superproton. Use for post-MD mechanism analysis; not force-field training or running MD.
---

Use the installed `superproton` package and its shared `workflow.Analysis` interface. The repository's `notebooks/superproton-tutorial.ipynb` contains six sections; locate the checkout supplied by the user rather than assuming a personal filesystem path. If only the package is installed, the example below works without the notebook.

```python
from superproton.io import AtomMapping
from superproton.workflow import Analysis
analysis = Analysis("trajectory.dump", material=AtomMapping(
    hydrogen_type=2, oxygen_type=4, center_type=3), frame_to_ps=0.05)
motion = analysis.motion
```

The example type IDs/time are the paper's presets, not universal defaults. Confirm the user's saved-frame time in ps, coordinate units in Å, atom selections, and fixed oxygen-to-center membership. A supplied `oxygen_parents` dictionary maps persistent oxygen IDs to center IDs; otherwise the checked four-nearest-O grouping is used. The current workflow requires unwrapped `id type xu yu zu`, persistent positive IDs, uniform sampling, full periodicity and a fixed orthorhombic cell.

Route by question:

| Question | Public interface |
|---|---|
| Rattling or productive transfer? | `analysis.motion`, `analysis.tracks`, `analysis.events` |
| Pointing before a hop? | `analysis.pointing()` → attempts, outcome summary, lookahead |
| Rotation/residence rates? | `residence.rate_summary(analysis.events, overall_model="single")`; use `"piecewise"` for a two-rate model |
| Coordination and rotation? | `analysis.coordination` |
| Event correlations? | `analysis.correlations()`; `correlation.temporal_overlap(events, parent_map)` |
| Rotation displacement / O–H reorientation? | `analysis.displacements(dwell_ps=...)` |
| Orientation landscape / relaxation? | `analysis.orientations(align_chs=False)` |

Inspect numerical outputs and source IDs/frames before interpreting plots. The short bundled examples demonstrate execution, not converged slow kinetics. Choose fitting, geometry dwell and symmetry settings explicitly. CHS alignment is specific to the original paper axes; it is not a general symmetry solver.

Keep these definitions distinct: donor swaps versus pair transitions; event-bounded pointing outcomes versus S20 post-episode lookahead; instantaneous versus per-event reorientation; characteristic fitted rates versus direct-count averages. Preserve failed/censored attempts. Low temporal overlap alone does not establish independence. State model assumptions explicitly and base mechanistic interpretations on the analyzed trajectory.

Export with `analysis.export(output_directory, result_dictionary, settings=actual_choices)` and inspect fit statuses, numerical data and plots. Nonempty destinations are rejected. Report the methodological citation [JACS 10.1021/jacs.5c21965](https://doi.org/10.1021/jacs.5c21965), software version and actual settings. Cite the supplied dataset separately when used.
