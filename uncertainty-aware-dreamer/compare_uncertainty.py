"""Compare GJS, sampled JS, and mean variance on immutable cached latent priors.

main loads one saved Gaussian ensemble, scores each fixed input once, writes
per-trajectory results/provenance, and renders time trends and fixed-input PDFs.
No simulator, latent rollout, checkpoint mutation, or training is performed.
"""

import argparse
import csv
import json
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from evaluation.cached_comparison import cache_groups, load_ensemble, physical_discrepancy, score_inputs, sha256
from evaluation.ensemble_distribution import plot_fixed_input


FORMULAS = {
    "gjs": "Original mean pairwise geometric JS for diagonal Gaussian members (alpha=0.5)",
    "js": "mean_k E_p_k[log p_k(y) - log(mean_j p_j(y))], natural logarithms",
    "transition_var": "mean_d mean_k (mu_kd - mean_j mu_jd)^2, population variance",
    "physical": "mean over all raw position dimensions of masked absolute difference, angles wrapped to [-pi,pi)"
}
INTERPRETATION = (
    "ID/OOD caches are trajectories selected by the original GJSD procedure; these are not independent "
    "OOD labels or an unbiased benchmark. Rescoring cannot change latent dynamics. Declining uncertainty "
    "alone does not establish an attractor. Discrepancy(t+1) remains accumulated free-running error, "
    "not teacher-forced one-step transition error. No controlled contraction statistic is computed: "
    "cache provenance does not establish multiple initial conditions under common actions."
)


def correlation(x, y):
    """Descriptive within-trajectory Pearson correlation; undefined cases are null."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def write_json(path, value):
    """Write standards-compliant metadata; reject accidental NaN results."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def plot_trends(directory, scores, repeats, error, reason):
    """Draw four independent y axes; separate trajectory spread from JS MC spread."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    for axis, name in zip(axes, ("physical", "gjs", "js", "transition_var")):
        values = error if name == "physical" else scores.get(name)
        axis.set_ylabel("Physical discrepancy" if name == "physical" else name)
        if values is None:
            axis.text(0.5, 0.5, reason if name == "physical" else "Metric not requested",
                      ha="center", va="center", wrap=True, transform=axis.transAxes)
        else:
            time = np.arange(values.shape[1])
            average, spread = values.mean(0), values.std(0, ddof=0)
            axis.plot(time, average, label="Trajectory mean")
            axis.fill_between(time, average - spread, average + spread, alpha=0.2,
                              label="+/-1 trajectory population SD (not a seed CI)")
            if name == "js":
                mc = repeats.mean(1).std(0, ddof=0)
                axis.plot(time, average + mc, "--", color="tab:orange", label="+/-1 MC SD of trajectory mean")
                axis.plot(time, average - mc, "--", color="tab:orange")
            axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Cached state index t; UQ(t) scores (z_t, a_t), physical discrepancy is at t")
    fig.suptitle("Same checkpoint and cached trajectories; JS is repeat-averaged")
    fig.savefig(directory / "time_comparison.png", dpi=160)
    plt.close(fig)


def write_group(directory, features, actions, scores, repeats, mean, std, error, reason, args):
    """Export scores, distinct uncertainty spreads, associations, and shared-limit plots."""
    directory.mkdir()
    n, length = features.shape[:2]
    payload = dict(scores)
    if error is not None:
        payload["physical_discrepancy"] = error
    if repeats is not None:
        payload["js_repeats"] = repeats
        payload["js_mc_sd"] = repeats.std(0, ddof=0)
    np.savez_compressed(directory / "scores.npz", **payload)
    headers = ["trajectory", "state_t", "physical_at_t", "physical_at_t_plus_1"] + list(scores)
    if repeats is not None:
        headers += ["js_mc_sd"] + [f"js_repeat_{i}" for i in range(args.js_repeats)]
    with (directory / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for trajectory in range(n):
            for time in range(length):
                row = [trajectory, time, error[trajectory, time] if error is not None else "",
                       error[trajectory, time + 1] if error is not None and time + 1 < length else ""]
                row += [values[trajectory, time] for values in scores.values()]
                if repeats is not None:
                    row += [repeats[:, trajectory, time].std()] + repeats[:, trajectory, time].tolist()
                writer.writerow(row)
    aggregate = {}
    for name, values in payload.items():
        if values.ndim == 2:
            aggregate[name] = {"trajectory_mean": values.mean(0).tolist(),
                               "trajectory_population_sd": values.std(0).tolist()}
    if repeats is not None:
        aggregate["js_mc_sd_of_trajectory_mean"] = repeats.mean(1).std(0).tolist()
    write_json(directory / "aggregate.json", aggregate)
    associations = []
    if error is not None:
        for name, values in scores.items():
            for trajectory in range(n):
                associations.append({"metric": name, "trajectory": trajectory,
                    "pearson_uq_t_vs_accumulated_error_t_plus_1": correlation(values[trajectory, :-1], error[trajectory, 1:]),
                    "error_end_minus_start": float(error[trajectory, -1] - error[trajectory, 0]),
                    "uq_end_minus_start": float(values[trajectory, -1] - values[trajectory, 0])})
    write_json(directory / "associations.json", {"descriptive_only": True, "values": associations})
    plot_trends(directory, scores, repeats, error, reason)
    trajectory = args.plot_trajectory
    if not 0 <= trajectory < n:
        raise ValueError("plot_trajectory is outside the cache")
    times = sorted(set([0, length // 2, length - 1]))
    dims = tuple(args.dimensions)
    if len(set(dims)) != 2 or min(dims) < 0 or max(dims) >= mean.shape[-1]:
        raise ValueError("Select two distinct valid ensemble output dimensions")
    selected_mean = mean[:, trajectory, times][:, :, dims]
    selected_std = std[:, trajectory, times][:, :, dims]
    limits = [(selected_mean - 4 * selected_std).min((0, 1)),
              (selected_mean + 4 * selected_std).max((0, 1))]
    peak_limit = float((1 / (2 * np.pi * selected_std.prod(-1))).max()) * 1.05
    for time in times:
        plot_fixed_input(None, features[trajectory, time].numpy(), actions[trajectory, time].numpy(),
                         directory / f"trajectory_{trajectory}_t_{time}", dimensions=dims, surface=args.surface,
                         precomputed_predictions=(mean[:, trajectory, time], std[:, trajectory, time]),
                         coordinate_limits=limits, density_limit=peak_limit)
    write_json(directory / "group.json", {"shape": [n, length], "physical_status": reason,
               "plot_trajectory": trajectory, "plot_times": times, "dimensions": list(dims),
               "coordinate_limits": np.asarray(limits).tolist(), "density_limit": peak_limit,
               "projection_warning": "Two coordinates can omit disagreement in other output dimensions",
               "alignment": "z_t,a_t unchanged; gt/decoded physical at t; successor of final action is absent"})


def main(argv=None):
    """Validate sources, reserve an empty independent output directory, then compare."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--infos", nargs="+", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", choices=("gjs", "js", "transition_var"), default=["gjs", "js", "transition_var"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--js-num-samples", type=int, default=128)
    parser.add_argument("--js-chunk-size", type=int, default=8)
    parser.add_argument("--js-seed", type=int, default=0)
    parser.add_argument("--js-repeats", type=int, default=5)
    parser.add_argument("--dimensions", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--surface", action="store_true")
    parser.add_argument("--plot-trajectory", type=int, default=0)
    parser.add_argument("--uq-only", action="store_true", help="Explicitly omit physical discrepancy")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.js_num_samples, args.js_chunk_size, args.js_repeats) < 1:
        parser.error("Batch, sample, chunk, and repeat counts must be positive")
    if len(set(args.metrics)) != len(args.metrics):
        parser.error("Each metric must be requested once")
    run = args.run_dir.resolve()
    sources = [run / "networks.pth", run / "config.yaml"] + [p.resolve() for p in args.infos]
    output = args.output_dir.resolve()
    if output == run or output.is_relative_to(run) or any(p.is_relative_to(output) for p in sources):
        parser.error("Output must be separate from the run directory and cannot contain source files")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output directory is not empty; choose a new directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    fingerprints = {str(p): sha256(p) for p in sources}
    output.mkdir(parents=True, exist_ok=True)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
        git_diff = subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=Path(__file__).parent)
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit, git_diff = "unavailable", b""
    (output / "working_tree.patch").write_bytes(git_diff)
    manifest = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                "sources_sha256": fingerprints, "git_commit": git_commit,
                "arguments": {k: str(v) if isinstance(v, Path) else [str(p) for p in v] if k == "infos" else v
                              for k, v in vars(args).items()},
                "torch_version": str(torch.__version__), "numpy_version": np.__version__, "python_version": sys.version,
                "analysis_code_sha256": {str(p.relative_to(Path(__file__).parent)): sha256(p) for p in
                    [Path(__file__), Path(__file__).parent / "evaluation/cached_comparison.py",
                     Path(__file__).parent / "evaluation/ensemble_distribution.py",
                     Path(__file__).parent / "uncertainty_aware_dreamer/ssm_mbrl/uncertainty/distance_measure.py",
                     Path(__file__).parent / "uncertainty_aware_dreamer/ssm_mbrl/uncertainty/ensemble.py",
                     Path(__file__).parent / "uncertainty_aware_dreamer/envs/dmc/suite_env.py"]},
                "metric_formulas": FORMULAS, "interpretation": INTERPRETATION,
                "cache_checkpoint_identity": "Supplied together by user; legacy caches contain no checkpoint hash",
                "js_seed_rule": "(js_seed + batch_index * js_repeats + repeat) modulo (2^63-1), reset per group",
                "rng_note": "CPU and selected CUDA RNG restored; batch-size changes need not be bitwise identical",
                "groups": []}
    write_json(output / "manifest.json", manifest)
    ensemble = None
    try:
        for source_index, source in enumerate(sources[2:]):
            if "post" in source.stem and "prior" not in source.stem:
                raise ValueError("Posterior cache is not a free-running prior cache")
            cache = torch.load(source, map_location="cpu", weights_only=True)
            for group_name, features, actions, group in cache_groups(cache):
                if ensemble is None:
                    ensemble, config = load_ensemble(run, features.shape[-1], actions.shape[-1], args.device)
                    (output / "source_config.yaml").write_bytes((run / "config.yaml").read_bytes())
                if features.shape[-1] != config.algorithm.world_model.model.transition.lsd + config.algorithm.world_model.model.transition.rec_state_dim:
                    raise ValueError("Inconsistent feature dimensions across caches")
                transition = config.algorithm.world_model.model.transition
                if group["states"]["sample"].shape[-1] != transition.lsd or group["states"]["gru_cell_state"].shape[-1] != transition.rec_state_dim:
                    raise ValueError("Cached stochastic/recurrent dimensions differ from configuration")
                error, reason = (None, "UQ only: explicitly requested") if args.uq_only else physical_discrepancy(group, config)
                scores, repeats, mean, std = score_inputs(ensemble, features, actions, args.metrics, args.batch_size,
                    args.js_num_samples, args.js_chunk_size, args.js_seed, args.js_repeats)
                if any(not np.isfinite(value).all() for value in scores.values()):
                    raise ValueError("Nonfinite uncertainty scores; refusing misleading aggregates")
                directory = output / f"{source_index:02d}_{source.stem}_{group_name}"
                write_group(directory, features, actions, scores, repeats, mean, std, error, reason, args)
                manifest["groups"].append({"source": str(source), "branch": group_name, "output": directory.name})
                print(f"Completed {source.name}/{group_name}: {tuple(features.shape[:2])}; {reason}", flush=True)
        if any(sha256(path) != value for path, value in fingerprints.items()):
            raise RuntimeError("A source changed during comparison")
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
