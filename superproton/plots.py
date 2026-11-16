"""Small reusable displays for computed tutorial results (no trajectory analysis)."""

from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .coordination import joint_density_Np_tau


def proton_history(analysis, *, proton_index=0):
    t = analysis.tracks
    time = (t.frame_indices - t.frame_indices[0]) * analysis.frame_to_ps
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, layout="constrained")
    for ax, values, label in zip(
        axes, [t.donor_O_id, t.donor_X_id], ["Donor oxygen ID", "Donor polyanion ID"]
    ):
        ax.step(time, values[:, proton_index], where="post")
        ax.set_ylabel(label)
    axes[-1].set_xlabel("Time since analysis start (ps)")
    fig.suptitle(f"Proton {t.proton_ids[proton_index]}: donor identity history")
    return fig


def pointing_results(result):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    s = result["summary"]
    decided = s[s.outcome != "censored"]
    axes[0].bar(decided.outcome, decided.fraction_decided)
    axes[0].set(ylabel="Fraction of fresh, decided attempts", ylim=(0, 1))
    axes[0].tick_params(axis="x", rotation=20)
    a = result["attempts"]
    values = a.loc[a.fresh & (a.outcome == "committed"), "delay_ps"]
    axes[1].hist(values, bins=30)
    axes[1].set(xlabel="Pointing onset → hop (ps)", ylabel="Completed attempts")
    lk = result["lookahead"]
    axes[2].plot(lk.lookahead_ps, lk.fraction, "o-")
    axes[2].set(
        xlabel="Extra lookahead after episode end (ps)",
        ylabel="Fraction of all fresh episodes",
        ylim=(0, 1),
    )
    return fig


def residence_distributions(samples, *, counts=None, probability=None, overall_model=None):
    """Show raw distributions; the model curve uses p fitted to selected bins.

    Supply all complete-interval ``counts`` so the observed probabilities and
    the unconditioned geometric model use the same normalization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for values, label in [
        (samples.tau_OO_ps, "Oxygen pair"),
        (samples.tau_PP_ps, "Polyanion pair"),
    ]:
        if len(values):
            axes[0].hist(values, bins="fd", density=True, histtype="step", label=label)
    if overall_model is not None:
        from .residence import DEFAULT_RHO_FLOOR_OO, fit_piecewise, fit_single_exp

        try:
            positive = samples.tau_OO_ps[samples.tau_OO_ps > 0]
            x = np.linspace(0, positive.max(), 500)
            if overall_model == "piecewise":
                fit = fit_piecewise(positive)
                lo = x <= fit.tau_c_ps
                y = np.exp(fit.intercept - fit.lambda_1_per_ps * x)
                axes[0].plot(
                    x[lo], y[lo], color="blue", label=f"1/λ₁ = {1 / fit.lambda_1_per_ps:.1f} ps"
                )
                y = np.exp(
                    fit.intercept
                    - fit.lambda_1_per_ps * fit.tau_c_ps
                    - fit.lambda_2_per_ps * (x - fit.tau_c_ps)
                )
                axes[0].plot(
                    x[~lo],
                    y[~lo],
                    color="teal",
                    ls="-.",
                    label=f"1/λ₂ = {1 / fit.lambda_2_per_ps:.1f} ps",
                )
                axes[0].axvline(fit.tau_c_ps, color="gray", ls=":", lw=1)
            elif overall_model == "single":
                fit = fit_single_exp(positive, rho_floor=DEFAULT_RHO_FLOOR_OO, trim_to_peak=True)
                rate = fit.lambda_per_ps
                axes[0].plot(
                    x, rate * np.exp(-rate * x), color="blue", label=f"1/λ = {1 / rate:.1f} ps"
                )
            else:
                raise ValueError("Choose single or piecewise")
            fit = fit_single_exp(samples.tau_PP_ps)
            x = np.linspace(0, samples.tau_PP_ps.max(), 500)
            rate = fit.lambda_per_ps
            axes[0].plot(
                x,
                rate * np.exp(-rate * x),
                color="tab:orange",
                ls="--",
                label=f"1/λP = {1 / rate:.1f} ps",
            )
        except (ValueError, RuntimeError):
            axes[0].text(
                0.98,
                0.98,
                "Fit unavailable for this sample",
                ha="right",
                va="top",
                transform=axes[0].transAxes,
                fontsize=8,
            )
    axes[0].set_ylim(5e-5, 0.15)
    axes[0].set(xlabel="Residence time (ps)", ylabel="Probability density", yscale="log")
    axes[0].legend(fontsize=8)
    values = np.asarray([] if counts is None else counts)
    if values.size:
        n = np.arange(values.max() + 1)
        axes[1].hist(values, bins=np.arange(len(n) + 1) - 0.5, density=True, label="Observed")
        if probability is not None:
            axes[1].plot(
                n,
                probability.p * (1 - probability.p) ** n,
                label=f"Geometric fit p={probability.p:.3f}",
            )
        else:
            axes[1].text(
                0.98, 0.9, "Insufficient counts for fit", ha="right", transform=axes[1].transAxes
            )
        axes[1].legend()
    else:
        axes[1].text(
            0.5, 0.5, "No complete productive intervals", ha="center", transform=axes[1].transAxes
        )
    axes[1].set(
        xlabel="Nonproductive events between productive events", ylabel="Probability", yscale="log"
    )
    return fig


def coordination_distribution(table, *, transition_ps=None, tau_bins=99, np_bins=70):
    """Joint density with a shared-y marginal and optional τc means.

    ``tau_bins`` controls logarithmic residence bins and ``np_bins`` controls
    coordination bins. Defaults preserve the Fig. 4d/7b display; fewer bins
    can make shorter tutorial samples easier to read without changing events.
    """
    if not isinstance(tau_bins, (int, np.integer)) or tau_bins < 1:
        raise ValueError("tau_bins must be a positive integer")
    if not isinstance(np_bins, (int, np.integer)) or np_bins < 1:
        raise ValueError("np_bins must be a positive integer")
    number = table.mean_bound_protons.to_numpy()
    tau = table.residence_ps.to_numpy()
    bounds = (number.min(), number.max())
    if bounds[0] == bounds[1]:
        bounds = (bounds[0] - 0.5, bounds[1] + 0.5)
    positive = tau[tau > 0]
    edges = np.geomspace(positive.min(), max(positive.max(), positive.min() * 1.01), tau_bins + 1)
    density, tau_edges, np_edges = joint_density_Np_tau(
        tau, number, tau_bins=edges, np_bins=np_bins, np_range=bounds
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9, 4.5),
        sharey=True,
        gridspec_kw={"width_ratios": [4, 1]},
        layout="constrained",
    )
    mesh = axes[0].pcolormesh(
        tau_edges, np_edges, np.ma.masked_less_equal(density.T, 0), shading="auto", norm=LogNorm()
    )
    fig.colorbar(mesh, ax=axes[0], label="Joint probability density (ps⁻¹)")
    axes[0].set(xlabel="O-pair residence (ps)", ylabel="Mean bound protons", xscale="log")
    axes[1].hist(
        number, bins="auto", density=True, orientation="horizontal", color="tab:blue", alpha=0.65
    )
    axes[1].set_xlabel("Density")
    if transition_ps is not None:
        for mask, left, right, color, label in [
            (tau < transition_ps, edges[0], transition_ps, "black", "τ < τc"),
            (tau >= transition_ps, transition_ps, edges[-1], "tab:orange", "τ ≥ τc"),
        ]:
            if mask.any():
                axes[0].hlines(number[mask].mean(), left, right, color=color, ls="--", label=label)
        axes[0].legend(fontsize=9)
    return fig


def event_correlations(result):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    mesh = axes[0].pcolormesh(
        result.dt_centers_ps,
        result.shells_A,
        result.G,
        shading="nearest",
        cmap="coolwarm",
        vmin=0,
        vmax=2,
    )
    fig.colorbar(mesh, ax=axes[0], label="G_event")
    axes[0].set(xlabel="Time lag (ps)", ylabel="Polyanion separation (Å)")
    for i in range(min(3, len(result.shells_A))):
        axes[1].plot(result.dt_centers_ps, result.G[i], label=f"{result.shells_A[i]:.2f} Å")
    axes[1].axhline(1, color="gray", ls="--")
    axes[1].legend()
    axes[1].set(xlabel="Time lag (ps)", ylabel="G_event")
    return fig


def displacements(result):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    rot = result["rotation"]
    metrics = result["event_metrics"]
    samples = result["samples"]
    for values, label in [
        (rot.productive_carrier_A, "Productive carrier O"),
        (rot.nonproductive_acceptor_A, "Nonproductive exchanger O"),
        (result["hops"].distance_A, "Proton hop"),
    ]:
        values = values[np.isfinite(values)]
        if len(values):
            axes[0, 0].hist(values, bins=40, density=True, histtype="step", label=label)
    axes[0, 0].set(xlabel="Displacement over rotation period (Å)", ylabel="Density")
    axes[0, 0].legend()
    for values, label in [
        (metrics.baseline, "Per-event baseline"),
        (metrics.outlier, "Per-event outlier"),
    ]:
        values = values[np.isfinite(values)]
        values = values[values > 0]
        if len(values):
            axes[0, 1].hist(
                values,
                bins=np.geomspace(
                    max(values.min(), 1e-3), max(values.max(), values.min() * 1.01), 40
                ),
                density=True,
                histtype="step",
                label=label,
            )
    axes[0, 1].set(xlabel="dH/dO", ylabel="Density", xscale="log")
    axes[0, 1].legend()
    if len(samples):
        chosen = samples.groupby("event_id").size().idxmax()
        trace = samples[samples.event_id == chosen]
        axes[1, 0].plot(trace.time_in_period_ps, trace.ratio)
        axes[1, 0].set(
            title=f"Longest sampled event, ID {chosen}",
            xlabel="Time in period (ps)",
            ylabel="Instantaneous dH/dO",
        )
        axes[1, 1].hist(result["direction_samples"].cos_HO.dropna(), bins=40, density=True)
        axes[1, 1].set(xlabel="cos(Δr_H, Δr_O)", ylabel="Density of pre-transfer samples")
    return fig


def orientations(result, *, include_single=True):
    """Display angular maps and vector relaxation without changing sampling.

    Set ``include_single=False`` to focus on the pooled orientation map; one
    bond's finite-window history need not sample the full angular landscape.
    """
    scopes = ["single", "all"] if include_single else ["all"]
    fig, axes = plt.subplots(
        1, len(scopes) + 1, figsize=(12 if include_single else 9, 3.5), layout="constrained"
    )
    for ax, scope in zip(axes[:-1], scopes):
        data = result["maps"][scope]
        mesh = ax.pcolormesh(
            data["phi_edges_deg"],
            data["theta_edges_deg"],
            np.ma.masked_less_equal(data["density"].T, 0),
            shading="auto",
            norm=LogNorm(),
        )
        ax.scatter(
            data["marker_phi_deg"],
            data["marker_theta_deg"],
            c="red",
            s=30,
            marker="o" if scope == "single" else "X",
        )
        ax.set(
            title="One bond: sampled window" if scope == "single" else "All polyanions (one bond each)",
            xlabel="Azimuth φ (°)",
            ylabel="Polar θ (°)",
            xlim=(-180, 180),
            ylim=(0, 180),
        )
        fig.colorbar(mesh, ax=ax, label="Density per degree²")
    axes[-1].plot(result["vcf"].lag_ps, result["vcf"].vcf)
    axes[-1].set(xlabel="Lag (ps)", ylabel="X–O vector autocorrelation")
    return fig


def pointing_example(analysis, result):
    """Show the longest completed fresh attempt as an explicitly selected example."""
    attempts = result["attempts"]
    completed = attempts.loc[attempts.fresh & (attempts.outcome == "committed")]
    fig, ax = plt.subplots(figsize=(9, 3), layout="constrained")
    if completed.empty:
        ax.text(
            0.5,
            0.5,
            "No completed fresh attempts in this window",
            ha="center",
            transform=ax.transAxes,
        )
        return fig
    event = completed.loc[completed.delay_ps.idxmax()]
    h, start, stop = int(event.proton_index), int(event.start_local), int(event.outcome_local)
    dt = analysis.tracks.frame_to_ps
    pad = max(2, int(1 / dt))
    begin, end = max(0, start - pad), min(analysis.tracks.n_frames, stop + pad + 1)
    time = (np.arange(begin, end) - start) * dt
    ax.step(time, analysis.tracks.donor_X_id[begin:end, h], where="post", label="Covalent donor")
    ax.axhline(event.target_id, color="tab:orange", ls="--", label="Pointed-at recipient")
    ax.axvline(0, color="gray", ls=":", label="Pointing onset")
    ax.axvline(event.delay_ps, color="tab:green", ls=":", label="Completed hop")
    ax.set(
        xlabel="Time from pointing onset (ps)",
        ylabel="Polyanion ID",
        title=f"Longest completed fresh attempt: proton {int(event.proton_id)}, delay {event.delay_ps:.2f} ps",
    )
    ax.legend(loc="best")
    return fig


def candidate_tracking(trace, *, material, time_range=None, title=None):
    """Figure 3c-e / S17 layout, with encounter labels and residence bands.

    Plotting a time window does not restart pair histories or displacement.
    Red shading uses the original polyanion-host backtracking convention.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    d, ev = trace.track, trace.events
    x_symbol = "P" if material == "CDP" else "S"
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 0.22, 1]},
        layout="constrained",
    )
    time = d.t_ps.to_numpy()
    axes[0].plot(time, d.H_displacement_A, lw=0.65)
    axes[1].plot(time, d.O_nearest_neighbor, lw=0.5)
    axes[3].plot(time, d.X_nearest_neighbor, lw=0.5)
    axes[0].set_ylabel("H displacement (Å)\nfrom reference frame")
    axes[1].set_ylabel("Nearest O\n(encounter label)")
    axes[3].set_ylabel(f"Nearest {x_symbol} host\n(encounter label)")
    for ax in (axes[1], axes[3]):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    o_switch = ev.loc[ev.kind == "O_pair_switch", "time_ps"].to_numpy()
    x_switch = ev.loc[ev.kind == "X_pair_switch", "time_ps"].to_numpy()
    starts = ev.loc[ev.kind == "reorient_start", "time_ps"].to_numpy()
    for ax in (axes[0], axes[1], axes[3]):
        for end in x_switch:
            ax.axvline(end, color="crimson", ls="--", lw=1.3)
        for start, end in zip(starts, x_switch):
            ax.axvspan(start, end, color="crimson", alpha=0.12)
    for switches, y, color in [(o_switch, 1, "tab:blue"), (x_switch, 0, "tab:orange")]:
        edges = np.r_[time[0], switches, time[-1]]
        for left, right in pairwise(edges):
            axes[2].plot([left, right], [y, y], color=color, marker="|", ms=9, lw=2)
    axes[2].set(
        yticks=[0, 1], yticklabels=[f"{x_symbol}–{x_symbol} pair", "O–O pair"], ylim=(-0.5, 1.5)
    )
    for spine in axes[2].spines.values():
        spine.set_visible(False)
    axes[2].tick_params(axis="both", length=0)
    axes[3].set_xlabel("Simulation time (ps)")
    if time_range is not None:
        if time_range[0] >= time[-1] or time_range[1] <= time[0]:
            raise ValueError(
                "Display window does not overlap this trajectory; use a matching production input"
            )
        axes[3].set_xlim(*time_range)
    else:
        axes[3].set_xlim(time[0], time[-1])
    handles = [
        Line2D([], [], color="crimson", ls="--", label="Productive pair switch / completed hop"),
        Patch(color="crimson", alpha=0.12, label="Productive rotation period"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8)
    fig.suptitle(title or f"{material}: proton ID {trace.proton_id}")
    return fig


def pointing_outcomes(summary, *, material, temperature_K, reference=None):
    """One-temperature slice of SI S18, optionally beside its archived fractions.

    All three outcomes use fresh decided attempts as denominator. Censored
    attempts are reported separately. The archived bar is labelled explicitly.
    """
    outcomes = ["committed", "reverted", "superseded"]
    colors = ["#2a9d8f", "tab:orange", "0.78"]
    indexed = summary.set_index("outcome")
    values = np.array([indexed.loc[k, "fraction_decided"] for k in outcomes])
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    columns = [values]
    labels = [f"Computed\n{temperature_K:g} K"]
    if reference is not None:
        columns.append(
            np.array([reference["P_commit"], reference["P_revert"], reference["P_superseded"]])
        )
        labels.append(f"Published\n{temperature_K:g} K")
    bottom = np.zeros(len(columns))
    for i, (outcome, color) in enumerate(zip(outcomes, colors)):
        heights = np.array([v[i] for v in columns])
        ax.bar(
            np.arange(len(columns)), heights, bottom=bottom, width=0.55, color=color, label=outcome
        )
        for x, (height, base) in enumerate(zip(heights, bottom)):
            if np.isfinite(height) and height > 0.035:
                ax.text(
                    x, base + height / 2, f"{height:.1%}", ha="center", va="center", fontsize=10
                )
        bottom += heights
    decided = int(indexed.loc[outcomes, "count"].sum())
    censored = int(indexed.loc["censored", "count"]) if "censored" in indexed.index else 0
    ax.set(
        xticks=np.arange(len(labels)),
        xticklabels=labels,
        ylim=(0, 1),
        ylabel="Fraction of fresh, decided attempts",
        title=f"SI Figure S18 — {material}\nComputed: {decided:,} decided; {censored:,} censored",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=9)
    return fig


def pointing_comparison(results, *, labels=None):
    """Compare computed pointing outcomes and optional directional follow-up.

    ``results`` maps sample names to ``Analysis.pointing()`` results; optional
    ``labels`` maps those names to display labels, for example material and
    temperature. No trajectories are read and no event definitions are changed.

    Bars use fresh, decided attempts. If every result includes ``lookahead``,
    a second panel shows crossings during a fresh pair episode or within the
    additional time after its end. That measure includes all fresh episodes,
    including terminally censored episodes, and can cross subsequent attempts.
    It is not a waiting-time distribution. Empty samples are marked explicitly.
    """
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    if not results:
        raise ValueError("Supply at least one pointing result")
    labels = {} if labels is None else labels
    names = list(results)
    followup = all("lookahead" in result for result in results.values())
    fig, axes = plt.subplots(
        1, 2 if followup else 1, squeeze=False,
        figsize=(12.2, 5.4) if followup else (6.2, 5.4), layout="constrained",
    )
    ax = axes[0, 0]
    outcomes = ["committed", "reverted", "superseded"]
    meanings = ["Transfer to target", "Return to previous pair", "Redirect"]
    colors = ["#2a9d8f", "tab:orange", "#b7bdc6"]
    tick_labels = []
    for x, name in enumerate(names):
        indexed = results[name]["summary"].set_index("outcome")
        decided = int(indexed.loc[outcomes, "count"].sum())
        censored = int(indexed.loc["censored", "count"]) if "censored" in indexed.index else 0
        tick_labels.append(f"{labels.get(name, name)}\n{decided:,} decided; {censored:,} censored")
        fractions = indexed.loc[outcomes, "fraction_decided"].to_numpy(dtype=float)
        if not decided or not np.isfinite(fractions).all():
            message = "No decided\nattempts" if not decided else "Outcome fractions\nunavailable"
            ax.text(x, 0.5, message, ha="center", va="center", color="0.4")
            continue
        bottom = 0.0
        for height, color in zip(fractions, colors):
            ax.bar(x, height, bottom=bottom, width=0.55, color=color)
            if height >= 0.035:
                ax.text(x, bottom + height / 2, f"{height:.1%}",
                        ha="center", va="center", fontsize=10)
            bottom += height
    ax.set(
        xticks=range(len(names)), xticklabels=tick_labels,
        xlim=(-0.6, len(names) - 0.4), ylim=(0, 1),
        ylabel="Fraction of fresh, decided attempts",
        title="What happens within each pointing attempt?",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(handles=[Patch(color=c, label=m) for c, m in zip(colors, meanings)],
              loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=1, frameon=False, fontsize=9)

    if followup:
        ax = axes[0, 1]
        missing = []
        for name in names:
            table = results[name]["lookahead"]
            if table.empty or not (table.n_fresh > 0).any():
                missing.append(str(name))
                continue
            valid = (table.n_fresh > 0) & np.isfinite(table.fraction)
            if not valid.any():
                missing.append(str(name))
                continue
            ax.plot(table.loc[valid, "lookahead_ps"], table.loc[valid, "fraction"],
                    "o-", markersize=4, label=f"{name} ({int(table.n_fresh.iloc[0]):,} fresh)")
        ax.set(
            xlabel="Additional time after pointing episode ends (ps)",
            ylabel="Fraction of all fresh episodes reaching target",
            ylim=(0, 1), title="Does the pointed-at host receive the proton later?",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend(frameon=False, fontsize=9, loc="lower right")
        if missing:
            ax.text(0.5, 0.48, "No follow-up data: " + ", ".join(missing),
                    transform=ax.transAxes, ha="center", va="center", color="0.4")
    return fig


def rotation_distances(result):
    """Fig. 3f/S21 role-resolved O displacements, plus S14 proton hop distances."""
    rot = result["rotation"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for axis, values, label, color, filled in [
        (0, rot.nonproductive_carrier_A, "Host", "tab:blue", False),
        (0, rot.nonproductive_acceptor_A, "Exchanger", "tab:blue", True),
        (1, rot.productive_carrier_A, "Carrier", "tab:orange", True),
        (1, rot.productive_acceptor_A, "Recipient", "tab:orange", False),
        (1, rot.productive_donor_A, "Provider", "tab:green", False),
    ]:
        values = values[np.isfinite(values)]
        if len(values):
            axes[axis].hist(
                values,
                bins="auto",
                density=True,
                label=label,
                color=color,
                histtype="stepfilled" if filled else "step",
                alpha=0.55 if filled else 1,
                lw=1.5,
            )
    hop = result["hops"].distance_A
    if len(hop):
        axes[2].hist(hop, bins="auto", density=True, color="tab:purple", alpha=0.6)
        axes[2].axvline(
            np.nanmean(hop), ls="--", color="black", label=f"Mean {np.nanmean(hop):.2f} Å"
        )
    for ax, title in zip(
        axes,
        ["Nonproductive — Fig. 3f / S21", "Productive — Fig. 3f / S21", "Proton hops — Fig. S14"],
    ):
        ax.set(
            title=title,
            xlabel="Displacement over rotation period (Å)",
            ylabel="Probability density",
        )
        ax.legend(fontsize=8)
    return fig


def reorientation_distributions(metrics, *, reference_bins=None):
    """Fig. 5b/7d: pooled baseline, separate productive/nonproductive outliers.

    Optional paper bin edges align the visual comparison; histogram values
    always come from the supplied events. Fractions use finite event metrics.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    series = [
        ("baseline", metrics.baseline, "gray", "Baseline (both types)"),
        (
            "outlier_nonproductive",
            metrics.outlier[metrics.kind == "nonproductive"],
            "tab:blue",
            "Outlier (nonproductive)",
        ),
        (
            "outlier_productive",
            metrics.outlier[metrics.kind == "productive"],
            "tab:orange",
            "Outlier (productive)",
        ),
    ]
    for name, values, color, label in series:
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        bins = "fd"
        if reference_bins is not None:
            rows = reference_bins[reference_bins.series == name].sort_values("bin_left")
            bins = np.r_[rows.bin_left.to_numpy(), rows.bin_right.iloc[-1]]
        hist, edges = np.histogram(values, bins=bins)
        ax.stairs(
            hist / len(values),
            edges,
            color=color,
            fill=name == "baseline",
            alpha=0.5 if name == "baseline" else 1,
            label=label,
            lw=1.7,
        )
    out = metrics.outlier[(metrics.kind == "productive") & np.isfinite(metrics.outlier)]
    if len(out):
        ax.set_xlim(0, np.percentile(out, 97))
    ax.set(
        xlabel="Per-event mean dH/dO",
        ylabel="Count / number of finite event metrics",
        title="Figure 5b / 7d: O–H reorientation",
    )
    ax.legend(frameon=False, fontsize=9)
    return fig


def reorientation_detail(result):
    """S15-style event trace with its baseline/outlier metrics, and S16 directions."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    samples = result["samples"]
    if len(samples):
        chosen = samples.groupby("event_id").size().idxmax()
        trace = samples[samples.event_id == chosen]
        axes[0].plot(trace.time_in_period_ps, trace.ratio, lw=0.7)
        axes[0].axhline(trace.ratio.mean(), ls="--", color="tab:blue", label="Baseline mean")
        out = trace.loc[trace.outlier, "ratio"]
        if len(out):
            axes[0].axhline(out.mean(), ls="-.", color="tab:orange", label="Outlier mean (Z > 2)")
        axes[0].set(
            title=f"Fig. S15-style example: event {chosen}",
            xlabel="Time in rotation period (ps)",
            ylabel="Instantaneous dH/dO",
        )
        axes[0].legend(fontsize=8)
    axes[1].hist(result["direction_samples"].cos_HO.dropna(), bins=40, density=True)
    axes[1].set(
        title="Figure S16: pre-transfer directions",
        xlabel="cos(Δr_H, Δr_O)",
        ylabel="Probability density",
        xlim=(-1, 1),
    )
    return fig
