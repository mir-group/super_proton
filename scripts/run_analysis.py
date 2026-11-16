"""Run one paper analysis on a trajectory and write reference-comparable CSVs.

This is the entry point for reproducing the manuscript's numbers from a full
production trajectory (see ``data/README.md`` for the Zenodo download). Each
``--analysis`` writes CSVs or numeric NPZ arrays into ``--out``. The figure map
(``docs/paper_mapping.md``) gives plotting commands and required conversions;
for example, residence-event times need ``figures/make_residence_csv.py``
before plotting their distributions.

Examples
--------
Residence-time distributions and rate fits for CsH2PO4 at 525 K::

    python scripts/run_analysis.py --analysis residence \\
        --dump data/zenodo/CDP_525K_4ns.trj --material CDP --temperature 525

Every analysis that a single trajectory supports::

    python scripts/run_analysis.py --analysis all \\
        --dump data/zenodo/CDP_525K_4ns.trj --material CDP --temperature 525

Check the result against the published values shipped in the repo::

    python scripts/run_analysis.py --analysis residence --dump ... \\
        --material CDP --temperature 525 --compare

Notes
-----
- The trajectory must have **unwrapped** coordinates (``xu yu zu``), which the
  production runs and the Zenodo files use.
- ``--equilibration-ps`` drops the head of the trajectory. The paper uses 10 ps
  and that is the default; changing it changes the numbers.
- Runtime on a 4 ns / 80k-frame CDP dump: about 2 minutes for ``residence``,
  about 4 minutes for ``msd``, and longer for ``all`` (which includes separate coordination and pointing passes). Memory peaks
  near 2 GB for ``msd`` and ``xoh``, which hold whole position blocks; pass
  ``--stride`` to thin if that is too much.
- Multi-temperature figures (Arrhenius plots) need one run per temperature;
  concatenate the per-temperature CSVs afterwards.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from superproton.geometry import xoh_geometry  # noqa: E402
from superproton.hopping import hop_distances, rotation_displacements  # noqa: E402
from superproton.io import polyanion_element, read_lammps_dump  # noqa: E402
from superproton.correlation import temporal_overlap  # noqa: E402
from superproton.published import DIFFUSIVITY_CM2_S as PUBLISHED_D  # noqa: E402
from superproton.published import OVERLAP_PERCENT as PUBLISHED_OVERLAP  # noqa: E402
from superproton.published import RATES_PS as PUBLISHED_RATES  # noqa: E402
from superproton.published import XOH_GEOMETRY as PUBLISHED_XOH  # noqa: E402
from superproton.msd import diffusivity, species_msd  # noqa: E402
from superproton.reorientation import read_HO_positions  # noqa: E402
from superproton.residence import (  # noqa: E402
    DEFAULT_RHO_FLOOR_OO,
    collect_taus,
    fit_piecewise,
    fit_single_exp,
)
from superproton.rotation import classify_events  # noqa: E402
from superproton.tracking import parent_polyanions, proton_neighbors  # noqa: E402

ANALYSES = ("residence", "xoh", "msd", "hop", "rotation_distance", "overlap",
            "orientation", "coordination", "event_correlation", "residence_comparison",
            "residence_categories", "pointing_lookahead", "rotation_period")

# The --compare view of the published values. Sourced from
# superproton.published so the paper's numbers live in exactly one place;
# the key names here are the labels each runner prints.
def _published(material: str, temperature: int) -> dict[str, dict[str, float]]:
    key = (material, temperature)
    out: dict[str, dict[str, float]] = {}
    if key in PUBLISHED_RATES:
        rates = PUBLISHED_RATES[key]
        out["residence"] = {
            f"1/lambda{n}_ps": rates[f"inv_lambda{n}"]
            for n in (1, 2, 3) if f"inv_lambda{n}" in rates
        }
        if "t_c" in rates:
            out["residence"]["t_c_ps"] = rates["t_c"]
    if key in PUBLISHED_XOH:
        geom = PUBLISHED_XOH[key]
        out["xoh"] = {"d_OH_A": geom["d_OH"][0], "d_XO_A": geom["d_XO"][0],
                      "angle_deg": geom["angle_deg"][0]}
    if key in PUBLISHED_D:
        out["msd"] = {"D_cm2_s": PUBLISHED_D[key]}
    if key in PUBLISHED_OVERLAP:
        out["overlap"] = {
            f"{nr}_{pr}_no_overlap_pct": bins[0]
            for (nr, pr), bins in PUBLISHED_OVERLAP[key].items()
        }
    return out


# Stable-bond dwell used for the X-O-H geometry table (SI Table S6): the donor
# oxygen must persist this long for the interval to contribute samples.
XOH_DWELL_PS = {"CDP": 2.0, "CHS": 10.0}

# Species column order of the published MSD CSVs. It differs between materials;
# see data/README.md.
MSD_COLUMN_ORDER = {"CDP": ["Cs", "P", "H", "O"], "CHS": ["Cs", "H", "S", "O"]}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _load_tracks_and_events(pipe, args):
    """Proton tracks plus classified rotation events, the shared front half.

    Cached on ``args``: several analyses share this step, and walking the
    trajectory with a per-frame neighbour search is the dominant cost.
    """
    cached = getattr(args, "_tracks_events", None)
    if cached is not None:
        return cached
    tracks = proton_neighbors(
        pipe,
        material=args.material,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
        frame_to_ps=args.frame_to_ps,
        show_progress=not args.quiet,
    )
    args._tracks_events = (tracks, classify_events(tracks))
    return args._tracks_events


def run_residence(pipe, args) -> dict[str, float]:
    """tau_O-O and tau_polyanion distributions, plus the exponential rate fits."""
    _, events = _load_tracks_and_events(pipe, args)
    taus = collect_taus(events)
    poly = polyanion_element(args.material)

    rows = [{"pair_type": "O", "residence_time_ps": f"{v:.4f}"} for v in taus.tau_OO_ps]
    rows += [{"pair_type": poly, "residence_time_ps": f"{v:.4f}"} for v in taus.tau_PP_ps]
    _write_csv(
        args.out / f"{args.material}_{args.temperature}K_residence_time.csv",
        ["pair_type", "residence_time_ps"], rows,
    )

    got: dict[str, float] = {}
    f3 = fit_single_exp(taus.tau_PP_ps)
    got["1/lambda3_ps"] = 1.0 / f3.lambda_per_ps
    if args.material == "CDP":
        fp = fit_piecewise(taus.tau_OO_ps)
        got["1/lambda1_ps"] = 1.0 / fp.lambda_1_per_ps
        got["1/lambda2_ps"] = 1.0 / fp.lambda_2_per_ps
        got["t_c_ps"] = fp.tau_c_ps
    else:
        f1 = fit_single_exp(
            taus.tau_OO_ps, rho_floor=DEFAULT_RHO_FLOOR_OO, trim_to_peak=True
        )
        got["1/lambda1_ps"] = 1.0 / f1.lambda_per_ps

    _write_csv(
        args.out / f"{args.material}_{args.temperature}K_rates.csv",
        ["quantity", "value"],
        [{"quantity": k, "value": f"{v:.6g}"} for k, v in got.items()]
        + [{"quantity": "n_tau_OO", "value": taus.tau_OO_ps.size},
           {"quantity": f"n_tau_{poly}{poly}", "value": taus.tau_PP_ps.size}],
    )
    return got


def run_xoh(pipe, args) -> dict[str, float]:
    """Covalent X-O-H geometry over stable-bond intervals (SI Table S6)."""
    from superproton.correlation import polyanion_centers

    tracks, _ = _load_tracks_and_events(pipe, args)
    positions = read_HO_positions(
        pipe,
        material=args.material,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
        show_progress=not args.quiet,
    )
    centers = polyanion_centers(pipe, args.material, frame=args.start_frame)
    dwell = args.xoh_dwell_ps or XOH_DWELL_PS[args.material]
    g = xoh_geometry(positions, tracks, centers, dwell_ps=dwell)

    got = {"d_OH_A": g.d_OH_mean_A, "d_XO_A": g.d_XO_mean_A,
           "angle_deg": g.angle_XOH_mean_deg}
    _write_csv(
        args.out / f"{args.material}_{args.temperature}K_xoh_geometry.csv",
        ["quantity", "mean", "std", "n_samples"],
        [
            {"quantity": "d_OH_A", "mean": f"{g.d_OH_mean_A:.4f}",
             "std": f"{g.d_OH_std_A:.4f}", "n_samples": g.n_samples},
            {"quantity": "d_XO_A", "mean": f"{g.d_XO_mean_A:.4f}",
             "std": f"{g.d_XO_std_A:.4f}", "n_samples": g.n_samples},
            {"quantity": "angle_XOH_deg", "mean": f"{g.angle_XOH_mean_deg:.2f}",
             "std": f"{g.angle_XOH_std_deg:.2f}", "n_samples": g.n_samples},
        ],
    )
    return got


def run_msd(pipe, args) -> dict[str, float]:
    """Per-species MSD versus lag time, and the proton tracer diffusivity."""
    m = species_msd(
        pipe,
        material=args.material,
        start_frame=0,  # the MSD estimator averages over all time origins
        end_frame=args.end_frame,
        stride=args.stride,
        frame_to_ps=args.frame_to_ps,
        frac=args.msd_frac,
        show_progress=not args.quiet,
    )
    order = MSD_COLUMN_ORDER[args.material]
    cols = m.as_columns(order)
    fieldnames = list(cols)
    rows = [
        {k: f"{cols[k][i]:.6f}" for k in fieldnames}
        for i in range(len(m.time_ps))
    ]
    _write_csv(args.out / f"{args.material}_{args.temperature}K_msd.csv", fieldnames, rows)

    fit_kw = {}
    if args.msd_fit_range_ps:
        fit_kw["fit_range_ps"] = tuple(args.msd_fit_range_ps)
    fit = diffusivity(m.time_ps, m.msd_A2["H"], skip_ps=args.msd_skip_lag_ps, **fit_kw)
    _write_csv(
        args.out / f"{args.material}_{args.temperature}K_diffusivity.csv",
        ["T_K", "D_cm2_s", "D_err_cm2_s", "fit_start_ps", "r2", "loglog_slope"],
        [{"T_K": args.temperature, "D_cm2_s": f"{fit.D_cm2_s:.4e}",
          "D_err_cm2_s": f"{fit.D_err_cm2_s:.2e}",
          "fit_start_ps": f"{fit.fit_start_ps:.1f}", "r2": f"{fit.r2:.5f}",
          "loglog_slope": f"{fit.loglog_slope:.4f}"}],
    )
    if not fit.is_diffusive:
        print(
            f"  warning: log-log slope of the fitted window is {fit.loglog_slope:.2f}, "
            "not ~1, so this trajectory is not yet in the diffusive regime and\n"
            "           D is NOT a diffusion coefficient. The 75 ps example dumps are "
            "far too short for D;\n           use a full 4 ns trajectory (see data/README.md).",
            file=sys.stderr,
        )
    return {"D_cm2_s": fit.D_cm2_s, "loglog_slope": fit.loglog_slope}


def _positions_and_events(pipe, args):
    """Tracks, events and H/O positions. Positions are cached for the same reason."""
    tracks, events = _load_tracks_and_events(pipe, args)
    cached = getattr(args, "_positions", None)
    if cached is not None:
        return tracks, events, cached
    positions = read_HO_positions(
        pipe,
        material=args.material,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
        with_X=False,
        show_progress=not args.quiet,
    )
    args._positions = positions
    return tracks, events, positions


def run_hop(pipe, args) -> dict[str, float]:
    """Long-range proton hop distance, one value per productive switch."""
    tracks, events, positions = _positions_and_events(pipe, args)
    hops = hop_distances(positions, tracks, events)
    _write_csv(
        args.out / f"{args.material}_{args.temperature}K_hop_distance.csv",
        ["temperature_K", "hop_distance_A"],
        [{"temperature_K": args.temperature, "hop_distance_A": f"{d:.6f}"}
         for d in hops.distance_A],
    )
    return {"n_hops": float(hops.distance_A.size),
            "mean_hop_A": float(hops.distance_A.mean()) if hops.distance_A.size else float("nan")}


def run_rotation_distance(pipe, args) -> dict[str, float]:
    """Per-role oxygen displacement over each switch's rotation period."""
    tracks, events, positions = _positions_and_events(pipe, args)
    d = rotation_displacements(positions, tracks, events)
    series = {
        "P_carrier": d.productive_carrier_A,
        "P_acceptor": d.productive_acceptor_A,
        "P_donor": d.productive_donor_A,
        "NP_carrier": d.nonproductive_carrier_A,
        "NP_acceptor": d.nonproductive_acceptor_A,
    }
    _write_csv(
        args.out / f"{args.material}_{args.temperature}K_O_displacement.csv",
        ["role", "displacement_A"],
        [{"role": role, "displacement_A": f"{v:.6f}"}
         for role, arr in series.items() for v in arr],
    )
    return {f"n_{k}": float(v.size) for k, v in series.items()}


def run_overlap(pipe, args) -> dict[str, float]:
    """NP-centred temporal overlap, separated by role (SI Tables S2-S3)."""
    _, events = _load_tracks_and_events(pipe, args)
    parents = parent_polyanions(pipe, args.material, frame=args.start_frame)
    result = temporal_overlap(events, parents)
    summary = result.summary()
    stem = f"{args.material}_{args.temperature}K_overlap"
    _write_csv(
        args.out / f"{stem}.csv",
        ["NP_role", "P_role", "overlap_count", "n_events", "n_np_events", "percent"],
        summary,
    )
    role_columns = {f"{nr}_{pr}": counts for (nr, pr), counts in result.counts.items()}
    _write_csv(
        args.out / f"{stem}_events.csv",
        ["proton_index", "t_s_frame", "t_f_frame", "exchanger_id", "host_id", *role_columns],
        [{"proton_index": result.proton_index[i], "t_s_frame": result.t_s_frame[i],
          "t_f_frame": result.t_f_frame[i],
          "exchanger_id": result.polyanion_ids["exchanger"][i],
          "host_id": result.polyanion_ids["host"][i],
          **{key: values[i] for key, values in role_columns.items()}}
         for i in range(result.proton_index.size)],
    )
    return {
        **{f"{row['NP_role']}_{row['P_role']}_no_overlap_pct": row["percent"]
           for row in summary if row["overlap_count"] == "0"},
        "n_productive": float(result.n_productive),
        "n_nonproductive": float(result.proton_index.size),
        "n_skipped_productive": float(result.n_skipped_productive),
        "n_skipped_nonproductive": float(result.n_skipped_nonproductive),
    }



def _write_arrays(path, arrays):
    # NumPy here is serialization only; calculations stay in the library.
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def run_orientation(pipe, args):
    from superproton.geometry import read_xo_unit_vectors, orientation_plot_data
    vectors, ids = read_xo_unit_vectors(pipe, args.material, start_frame=args.start_frame,
                                       end_frame=args.end_frame, stride=args.stride)
    sites_path = REPO_ROOT/'data/reference/orientation'/f'{args.material}_experimental_sites.csv'
    sites = None
    if sites_path.exists():
        with sites_path.open() as fh:
            reader = csv.reader(fh)
            next(reader)
            sites = [[float(x) for x in row] for row in reader]
    panels = orientation_plot_data(vectors, single_index=args.orientation_index,
                                   experimental_sites=sites)
    for scope, data in panels.items():
        _write_arrays(args.out/f'{args.material}_{args.temperature}K_{scope}.npz',data)
    return {'n_frames':float(len(vectors)), 'n_polyanions':float(len(ids))}


def run_coordination(pipe, args):
    from superproton.coordination import mutual_bound_protons, mean_Np_per_residence
    tracks, events = _load_tracks_and_events(pipe,args)
    bound = mutual_bound_protons(pipe,args.material,tracks.frame_indices)
    tau, count = mean_Np_per_residence(tracks,events,bound)
    _write_csv(args.out/f'{args.material}_{args.temperature}K_coordination.csv',
               ['tau_ps','mean_Np'],
               ({'tau_ps':t,'mean_Np':n} for t,n in zip(tau,count)))
    return {'n_intervals':float(len(tau))}


def run_event_correlation(pipe,args):
    from superproton.correlation import g_event
    _,events = _load_tracks_and_events(pipe,args)
    result = g_event(events,pipe,args.material,r_max_A=15,dt_max_ps=200,
                     dt_bin_ps=1,min_H_exp=5)
    _write_arrays(args.out/f'{args.material}_{args.temperature}K_g_event.npz',vars(result))
    return {'n_shells':float(len(result.shells_A)), 'event_rate_per_ps':result.rate_per_ps}


def run_residence_comparison(pipe,args):
    from superproton.residence import residence_fit_comparison
    _,events = _load_tracks_and_events(pipe,args)
    tau = collect_taus(events).tau_OO_ps
    data = residence_fit_comparison(tau)
    _write_arrays(args.out/f'{args.material}_{args.temperature}K_comparison.npz',data)
    return {'n_intervals':float(len(tau))}


def run_residence_categories(pipe,args):
    _,events = _load_tracks_and_events(pipe,args)
    tau = collect_taus(events)
    groups = {'NP-NP':tau.tau_OO_NN_ps, 'NP-P':tau.tau_OO_PN_ps,'P-P':tau.tau_OO_PP_ps}
    _write_csv(args.out/f'{args.material}_{args.temperature}K_categories.csv',
        ['category','tau_ps'], ({'category':name,'tau_ps':t}
                              for name,values in groups.items() for t in values))
    return {f'n_{name}':float(len(values)) for name,values in groups.items()}


def run_pointing_lookahead(pipe,args):
    from superproton.pointing import read_pointing_hosts, episode_crossing_lookahead
    donor, acceptor = read_pointing_hosts(pipe,args.material,start_frame=args.start_frame,
                                         end_frame=args.end_frame,stride=args.stride)
    result = episode_crossing_lookahead(donor,acceptor,[2,5,10,20,40,60,80,100,120,150],
                                       frame_to_ps=args.frame_to_ps*args.stride)
    _write_csv(args.out/f'{args.material}_{args.temperature}K_lookahead.csv',
        ['material','temperature_K','lookahead_ps','fraction'],
        (dict(material=args.material,temperature_K=args.temperature,lookahead_ps=t,fraction=p)
         for t,p in zip(result.lookahead_ps,result.fraction)))
    return {'n_fresh':float(result.n_fresh)}


def run_rotation_period(pipe,args):
    from superproton.rotation import rotation_period_histograms
    tracks,events = _load_tracks_and_events(pipe,args)
    rows = rotation_period_histograms(tracks,events)
    for row in rows:
        if row['switch_type']=='X-pair': row['switch_type']=polyanion_element(args.material)+'-pair'
    _write_csv(args.out/f'{args.material}_{args.temperature}_rot_period.csv',
        ['switch_type','bin_left_ps','bin_right_ps','bin_center_ps','prob_density'],rows)
    return {'n_bins':float(len(rows))}

RUNNERS = {
    "residence": run_residence,
    "xoh": run_xoh,
    "msd": run_msd,
    "hop": run_hop,
    "rotation_distance": run_rotation_distance,
    "overlap": run_overlap,
    "orientation": run_orientation,
    "coordination": run_coordination,
    "event_correlation": run_event_correlation,
    "residence_comparison": run_residence_comparison,
    "residence_categories": run_residence_categories,
    "pointing_lookahead": run_pointing_lookahead,
    "rotation_period": run_rotation_period,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dump", required=True, type=Path,
                   help="LAMMPS dump trajectory with unwrapped coordinates.")
    p.add_argument("--material", required=True, choices=["CDP", "CHS"])
    p.add_argument("--temperature", required=True, type=int,
                   help="Temperature in K. Labels the output only.")
    p.add_argument("--analysis", required=True,
                   choices=[*ANALYSES, "all"], nargs="+",
                   help="One or more analyses to run, or 'all'.")
    p.add_argument("--out", type=Path, default=Path("results"),
                   help="Directory for the output CSVs (default: ./results).")
    p.add_argument("--equilibration-ps", type=float, default=10.0,
                   help="Drop this much of the trajectory head (default 10, as in the paper). "
                        "Does not apply to --analysis msd, whose estimator averages over every "
                        "time origin; use --msd-skip-lag-ps there.")
    p.add_argument("--end-frame", type=int, default=None,
                   help="Exclusive final frame index (default: end of trajectory).")
    p.add_argument("--stride", type=int, default=1,
                   help="Read every Nth frame. Changes the numbers; use only to fit memory.")
    p.add_argument("--frame-to-ps", type=float, default=0.05,
                   help="Time per dumped frame in ps (default 0.05 = 50 fs).")
    p.add_argument("--xoh-dwell-ps", type=float, default=None,
                   help="Stable-bond dwell for --analysis xoh (default: 2 for CDP, 10 for CHS).")
    p.add_argument("--msd-skip-lag-ps", type=float, default=10.0,
                   help="For --analysis msd: drop lags shorter than this before fitting D "
                        "(default 10). This trims the ballistic short-time head of the MSD "
                        "curve; it is a lag cut, not an equilibration cut.")
    p.add_argument("--msd-frac", type=float, default=1.0,
                   help="Report MSD lags out to this fraction of the trajectory (default 1.0).")
    p.add_argument("--msd-fit-range-ps", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Fit D over an explicit lag window instead of the automatic scan. "
                        "D varies by a few percent with the window; see superproton.msd.")
    p.add_argument("--orientation-index", type=int, default=23,
                   help="Zero-based single-polyanion column for orientation (paper: 23).")
    p.add_argument("--compare", action="store_true",
                   help="Print the published value alongside each result, where one is known.")
    p.add_argument("--quiet", action="store_true", help="Suppress progress bars.")
    args = p.parse_args(argv)

    if not args.dump.exists():
        print(f"Dump not found: {args.dump}", file=sys.stderr)
        return 1
    if args.stride != 1:
        print(f"warning: --stride {args.stride} changes every reported quantity; "
              "the published values assume stride 1.", file=sys.stderr)

    args.start_frame = int(round(args.equilibration_ps / args.frame_to_ps))
    requested = ANALYSES if "all" in args.analysis else tuple(dict.fromkeys(args.analysis))
    if "all" in args.analysis and args.material != "CDP":
        # The model comparison is a CDP-specific paper analysis, not a CHS fit.
        requested = tuple(name for name in requested if name != "residence_comparison")

    t_open = time.time()
    pipe = read_lammps_dump(args.dump)
    n_frames = int(pipe.source.num_frames)
    print(f"{args.dump.name}: {n_frames} frames "
          f"({n_frames * args.frame_to_ps:.1f} ps), opened in {time.time() - t_open:.1f} s")
    print(f"dropping {args.start_frame} frames ({args.equilibration_ps} ps) of equilibration")

    for name in requested:
        print(f"\n--- {name} ---", flush=True)
        t0 = time.time()
        got = RUNNERS[name](pipe, args)
        published = _published(args.material, args.temperature).get(name, {})
        width = max((len(k) for k in got), default=0)
        for key, value in got.items():
            line = f"  {key:<{width}} = {value:.4g}"
            if args.compare and key in published:
                ref = published[key]
                line += f"   published {ref:.4g}   ratio {value / ref:.4f}"
            print(line)
        if args.compare and not published:
            print(f"  (no published reference recorded for {args.material} "
                  f"{args.temperature} K)")
        print(f"  wrote data to {args.out}/  ({time.time() - t0:.0f} s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
