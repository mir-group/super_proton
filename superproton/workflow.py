"""Reusable orchestration for the six tutorial sections; science stays in modules.

All expensive trajectory products are lazy and shared within one Analysis.
This class accepts explicit mappings without altering package source.
"""

import json
from dataclasses import asdict, is_dataclass, replace
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from . import (
    coordination,
    correlation,
    geometry,
    hopping,
    io,
    pointing,
    reorientation,
    residence,
    rotation,
    tracking,
)


class Analysis:
    """One uniformly sampled, unwrapped LAMMPS analysis window.

    ``frame_to_ps`` is time between saved frames, BEFORE stride. All bounds
    are source-frame indices and ``end_frame`` is exclusive. Material may be
    CDP/CHS or an explicit AtomMapping. Fits and symmetry are chosen explicitly
    by the caller rather than inferred for a new material.

    ``cache_dir`` optionally persists exact pointing histories, tracks and
    coordination results. Keep inputs/settings fixed within an instance;
    construct another Analysis to analyze a changed file or window.
    ``collect_coordination=True`` saves strict mutual O-H bond counts while
    reading tracks so a later coordination calculation needs no second pass.
    """

    def __init__(self, path, *, material, frame_to_ps, start_frame=0, end_frame=None, stride=1,
                 cache_dir=None, collect_coordination=False):
        self.path = Path(path).resolve()
        self.material = material
        self.collect_coordination = collect_coordination
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser().resolve()
        if any(type(v) is not int for v in (start_frame, stride)) or (
            end_frame is not None and type(end_frame) is not int
        ):
            raise ValueError("Frame bounds and stride must be integers")
        if not np.isfinite(frame_to_ps) or frame_to_ps <= 0 or stride < 1 or start_frame < 0:
            raise ValueError("Positive dt/stride and nonnegative frame start required")
        self.frame_to_ps = float(frame_to_ps)
        self.start_frame, self.end_frame, self.stride = start_frame, end_frame, stride

    @cached_property
    def pipeline(self):
        self._check_source()
        pipeline = io.validated_pipeline(self.path)
        self._check_source()
        return pipeline

    def _check_source(self):
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        previous = getattr(self, "_source_identity", identity)
        if previous != identity:
            raise RuntimeError("Trajectory changed; create a new Analysis before calculating")
        self._source_identity = identity
        return stat

    @property
    def window(self):
        return {"start_frame": self.start_frame, "end_frame": self.end_frame, "stride": self.stride}

    def _cache_parameters(self, operation, **settings):
        # Bump the operation's version when its scientific definition changes.
        configuration = {
            "material": asdict(self.material) if is_dataclass(self.material) else self.material,
            "frame_to_ps": self.frame_to_ps,
            **self.window,
        }
        signature = json.dumps(configuration, sort_keys=True)
        if getattr(self, "_analysis_configuration", signature) != signature:
            raise RuntimeError("Analysis settings changed; create a new Analysis before calculating")
        self._analysis_configuration = signature
        return {"operation": operation, **configuration, **settings}

    def _cached_arrays(self, parameters, compute):
        self._check_source()
        result = io.cached_arrays(
            self.cache_dir, source=self.path, parameters=parameters, compute=compute
        )
        self._check_source()
        return result

    @cached_property
    def tracks(self):
        def compute():
            result = tracking.proton_neighbors(
                self.pipeline, material=self.material, frame_to_ps=self.frame_to_ps,
                show_progress=False, collect_coordination=self.collect_coordination, **self.window,
            )
            return {k: v for k, v in vars(result).items() if isinstance(v, np.ndarray)}

        operation = ("proton-tracks-with-coordination-v1" if self.collect_coordination
                     else "proton-tracks-v1")
        arrays = self._cached_arrays(self._cache_parameters(operation), compute)
        return tracking.ProtonTracks(
            n_protons=len(arrays["proton_ids"]), n_frames=len(arrays["frame_indices"]),
            frame_to_ps=self.frame_to_ps * self.stride, **arrays,
        )

    def prefix(self, end_frame):
        """Reuse tracks/counts up to an exclusive absolute source-frame end.

        The start and stride stay unchanged. Arrays share storage with this
        Analysis; no trajectory frames are reread. Events and residence
        intervals are classified anew so the new end is correctly censored.
        Other analyses read only the shorter window if they need coordinates.
        The requested end cannot exceed the last available sample plus stride.
        """
        if type(end_frame) is not int or end_frame <= self.start_frame:
            raise ValueError("Prefix end must be an integer after the analysis start")
        if self.end_frame is not None and end_frame > self.end_frame:
            raise ValueError("A prefix cannot extend the parent analysis window")
        self._check_source()
        self._cache_parameters("prefix-v1")  # Check that existing settings are unchanged.
        if end_frame == self.end_frame:
            return self
        tracks = self.tracks
        if end_frame > int(tracks.frame_indices[-1]) + self.stride:
            raise ValueError("Prefix end exceeds the available sampled frames")
        stop = int(np.searchsorted(tracks.frame_indices, end_frame))
        child = Analysis(
            self.path, material=self.material, frame_to_ps=self.frame_to_ps,
            start_frame=self.start_frame, end_frame=end_frame, stride=self.stride,
            cache_dir=self.cache_dir, collect_coordination=self.collect_coordination,
        )
        child._cache_parameters("prefix-v1")  # Freeze the settings of the shared tracks.
        fields = {name: getattr(tracks, name)[:stop] for name in (
            "frame_indices", "donor_O_id", "donor_X_id", "acceptor_O_id", "acceptor_X_id"
        )}
        if tracks.mutual_bound_counts is not None:
            fields["mutual_bound_counts"] = tracks.mutual_bound_counts[:stop]
        child.__dict__["tracks"] = replace(tracks, n_frames=stop, **fields)
        self._check_source()
        child._source_identity = self._source_identity
        return child

    @cached_property
    def events(self):
        return rotation.classify_events(self.tracks)

    @cached_property
    def motion(self):
        return rotation.proton_motion_table(self.tracks, self.events)

    @cached_property
    def residence(self):
        return residence.collect_taus(self.events)

    @cached_property
    def positions(self):
        return reorientation.read_HO_positions(
            self.pipeline, material=self.material, show_progress=False, **self.window
        )

    def pointing(self, *, d_form=1.8, d_break=2.4, lookahead_ps=(0, 2, 5, 10, 25, 50, 100, 150)):
        """S18 outcomes, optionally with the S20 post-episode lookahead.

        Pass ``lookahead_ps=None`` for S18 alone. Repeated identical calls reuse
        the most recent result. With ``cache_dir``, compact host histories are
        also reused across Analysis instances/kernel restarts; full neighbor
        lists and positions are not retained for pointing.
        """
        if not np.isfinite([d_form, d_break]).all() or not 0 < d_form < d_break:
            raise ValueError("Require 0 < d_form < d_break")
        d_form, d_break = float(d_form), float(d_break)
        grid = None if lookahead_ps is None else np.asarray(lookahead_ps, dtype=float)
        if grid is not None and (grid.ndim != 1 or not grid.size
                                or not np.isfinite(grid).all() or np.any(grid < 0)):
            raise ValueError("Require finite nonnegative lookaheads, or None")
        parameters = self._cache_parameters("pointing-hosts-v1", d_form=d_form, d_break=d_break)
        stat = self._check_source()
        key = (json.dumps(parameters, sort_keys=True), stat.st_size, stat.st_mtime_ns,
               stat.st_ctime_ns, None if grid is None else tuple(grid))
        memo = getattr(self, "_pointing_memo", None)
        if memo is not None and memo[0] == key:
            return {name: table.copy() for name, table in memo[1].items()}

        def compute():
            donor, acceptor = pointing.read_pointing_hosts(
                self.pipeline, self.material, d_form=d_form, d_break=d_break, **self.window
            )
            data = self.pipeline.compute(self.start_frame)
            species = {v: k for k, v in io.species_map(self.pipeline, self.material).items()}
            proton_ids = np.asarray(data.particles.identifiers)[
                np.asarray(data.particles.particle_types) == species["H"]
            ].copy()
            return {"donor": donor, "acceptor": acceptor, "proton_ids": proton_ids,
                    "frame_indices": self.start_frame + np.arange(len(donor)) * self.stride}

        hosts = self._cached_arrays(parameters, compute)
        dt = self.frame_to_ps * self.stride
        attempts = pointing.pointing_outcomes(
            hosts["donor"], hosts["acceptor"], frame_to_ps=dt,
            proton_ids=hosts["proton_ids"], source_frames=hosts["frame_indices"],
        )
        tables = {"attempts": attempts, "summary": pointing.pointing_summary(attempts)}
        if grid is not None:
            result = pointing.episode_crossing_lookahead(
                hosts["donor"], hosts["acceptor"], grid, frame_to_ps=dt
            )
            tables["lookahead"] = pd.DataFrame(
                {
                    "lookahead_ps": result.lookahead_ps,
                    "fraction": result.fraction,
                    "n_crossed": result.n_crossed,
                    "n_fresh": result.n_fresh,
                }
            )
        self._pointing_memo = (key, tables)
        return {name: table.copy() for name, table in tables.items()}

    @cached_property
    def coordination(self):
        def compute():
            if self.tracks.mutual_bound_counts is None:
                bound = coordination.mutual_bound_protons(
                    self.pipeline, self.material, self.tracks.frame_indices
                )
            else:
                bound = coordination.CoordinationData(
                    self.tracks.n_frames, len(self.tracks.bound_polyanion_ids),
                    self.tracks.bound_polyanion_ids, self.tracks.mutual_bound_counts,
                )
            chunks = []
            for h in range(self.tracks.n_protons):
                tau, np_mean = coordination.residence_coordination(
                    self.events.o_pair_switch_frames[h], self.tracks.donor_X_id[:, h],
                    bound, frame_to_ps=self.tracks.frame_to_ps,
                )
                chunks.append(pd.DataFrame({
                    "proton_id": int(self.tracks.proton_ids[h]),
                    "residence_ps": tau, "mean_bound_protons": np_mean,
                }))
            table = pd.concat(chunks, ignore_index=True)
            return {name: table[name].to_numpy() for name in table}

        return pd.DataFrame(self._cached_arrays(
            self._cache_parameters("residence-coordination-v1"), compute
        ))

    def correlations(self, *, dt_max_ps=50, dt_bin_ps=1, r_max_A=10):
        return correlation.g_event(
            self.events,
            self.pipeline,
            self.material,
            dt_max_ps=dt_max_ps,
            dt_bin_ps=dt_bin_ps,
            r_max_A=r_max_A,
        )

    def displacements(self, *, dwell_ps):
        centers = correlation.polyanion_centers(self.pipeline, self.material)
        geom = geometry.xoh_geometry(self.positions, self.tracks, centers, dwell_ps=dwell_ps)
        hop = hopping.hop_distances(self.positions, self.tracks, self.events)
        rot = hopping.rotation_displacements(self.positions, self.tracks, self.events)
        return {
            "hops": hop,
            "rotation": rot,
            "geometry": geom,
            "event_metrics": reorientation.compute_reorientation(
                self.positions, self.tracks, self.events
            ),
            "direction_samples": reorientation.pretransfer_direction_samples(
                self.positions, self.tracks, self.events
            ),
            "samples": reorientation.reorientation_samples(
                self.positions, self.tracks, self.events
            ),
            "rigid": geometry.rigid_rotation_comparison(
                geom, np.nanmean(rot.productive_carrier_A), np.nanmean(hop.distance_A)
            ),
        }

    def orientations(
        self,
        *,
        align_chs=False,
        max_lag_ps=25,
        single_index=0,
        lag_step_ps=0.5,
        experimental_sites=None,
        vcf_sample_slice=None,
        map_bins=100,
    ):
        """Angular maps and VCF from one vector read.

        ``vcf_sample_slice`` optionally selects/subsamples the loaded vectors
        for VCF only; map sampling is unchanged. Its indices refer to the
        loaded analysis window, and its step multiplies the saved-frame time.
        For Figure 8, load frames 0:80000 and use slice(200, 30200, 10):
        10 ps equilibration, then 1500 ps sampled every 0.5 ps.
        """
        vectors, ids = geometry.read_xo_unit_vectors(
            self.pipeline, self.material, align_sulfate=False, **self.window
        )
        if align_chs:
            initial, _ = geometry.read_xo_unit_vectors(
                self.pipeline, self.material, start_frame=0, end_frame=1, align_sulfate=False
            )
            vectors = geometry.align_sulfate_orientations(vectors, initial_vectors=initial[0])
        dt = self.frame_to_ps * self.stride
        vcf_vectors = vectors
        if vcf_sample_slice is not None:
            if not isinstance(vcf_sample_slice, slice):
                raise ValueError("vcf_sample_slice must be a slice with a positive step")
            sample_step = 1 if vcf_sample_slice.step is None else vcf_sample_slice.step
            if sample_step <= 0:
                raise ValueError("vcf_sample_slice must be a slice with a positive step")
            vcf_vectors = vectors[vcf_sample_slice]
            dt *= sample_step
        if not len(vcf_vectors):
            raise ValueError("VCF sample window is empty")
        lag, vcf = geometry.vector_autocorrelation(
            vcf_vectors, max_lag_frames=int(max_lag_ps / dt), stride=max(1, int(lag_step_ps / dt))
        )
        return {
            "vectors": vectors,
            "center_ids": ids,
            "maps": geometry.orientation_plot_data(
                vectors, single_index=single_index, experimental_sites=experimental_sites,
                bins=map_bins,
            ),
            "vcf": pd.DataFrame({"lag_ps": lag * dt, "vcf": vcf}),
        }

    def export(self, out, results, *, settings):
        """Save supplied computed results, including exact choices and provenance.

        Does not recompute analyses. Nonempty directories are rejected.
        NaNs are retained in numeric files; JSON metadata uses null.
        """
        from . import __version__

        out = Path(out)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(out)
        out.mkdir(parents=True, exist_ok=True)
        files = []

        def write(name, value):
            if is_dataclass(value):
                value = asdict(value)
            if isinstance(value, pd.DataFrame):
                dest = out / (name + ".csv")
                value.to_csv(dest, index=False)
                files.append(dest.name)
            elif isinstance(value, dict):
                arrays = {
                    ("_".join(map(str, k)) if isinstance(k, tuple) else str(k)): (
                        v.astype(str) if v.dtype.kind == "O" else v
                    )
                    for k, v in value.items()
                    if isinstance(v, np.ndarray)
                }
                if arrays:
                    dest = out / (name + ".npz")
                    np.savez_compressed(dest, **arrays)
                    files.append(dest.name)
                for k, v in value.items():
                    if not isinstance(v, np.ndarray):
                        write(
                            name
                            + "_"
                            + ("_".join(map(str, k)) if isinstance(k, tuple) else str(k)),
                            v,
                        )
            elif isinstance(value, np.ndarray):
                dest = out / (name + ".npz")
                np.savez_compressed(dest, values=value)
                files.append(dest.name)
            elif np.isscalar(value) or value is None:
                dest = out / (name + ".json")
                scalar = value.item() if isinstance(value, np.generic) else value
                if isinstance(scalar, float) and not np.isfinite(scalar):
                    scalar = None
                dest.write_text(json.dumps(scalar))
                files.append(dest.name)
            else:
                raise TypeError(f"Unsupported result {name}: {type(value)}")

        for name, value in results.items():
            write(name, value)
        stat = self.path.stat()
        config = dict(
            material=asdict(self.material) if is_dataclass(self.material) else self.material,
            frame_to_ps=self.frame_to_ps,
            **self.window,
            **settings,
        )
        manifest = {
            "schema_version": 1,
            "software_version": __version__,
            "input": {
                "path": str(self.path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "checksum_status": "not_computed",
            },
            "settings": config,
            "files": files,
            "paper_doi": "10.1021/jacs.5c21965",
            "units": {"length": "angstrom", "time": "ps"},
            "analysis_status": {
                name: (
                    "insufficient_data"
                    if isinstance(value, pd.DataFrame)
                    and "status" in value
                    and (value.status != "complete").any()
                    else "complete"
                )
                for name, value in results.items()
            },
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return out
