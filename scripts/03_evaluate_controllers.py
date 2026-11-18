"""Paired Monte Carlo comparison of LQR and LQR + K-MPC; Tables II-III.

    PYTHONPATH=src python scripts/03_evaluate_controllers.py configs/evaluate/kmpc_vs_lqr.toml

Every controller flies every experiment once per wind seed, with the same
wind for all controllers of one seed (simulation.monte_carlo). Writes the run
record of experiment.run_record plus:
    episodes/<experiment>/<controller>/run_<i>/log.npz   (+ mpc_log.npz for K-MPC)
    position_errors.npz   e_p per experiment and controller, (runs, T, 3), m, NED;
                          input for Figs. 3-4 (analysis/plotting, not here)
    summary.json          per-run metrics, paired comparisons, K-MPC solver stats
    summary.md            Tables II-III style comparison
    kmpc_identification.json  copy of the model run's identification report

Metrics per run (analysis.metrics): Euclidean norm (Table II), RMSE x, y, z
(Table III), all over the whole run, m. K-SMPC is not implemented yet.
"""
import argparse
import json
import logging
from pathlib import Path
import shutil

import numpy as np

from analysis.metrics import euclidean_norm_error, paired_comparison, rmse
from datagen.collect import ERR_POS, tracking_error
from experiment.augmentation import KMPCAugmentation
from experiment.builders import load_evaluation, load_experiment, load_kmpc
from experiment.run_record import save_log, start_run
from simulation.monte_carlo import run_paired

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE = "lqr"
CONTROLLERS = ("lqr", "kmpc")
METRICS = ("euclidean_norm", "rmse_x", "rmse_y", "rmse_z")
logger = logging.getLogger("evaluate_controllers")


def run_metrics(log) -> dict:
    """Table II-III metrics of one run, m."""
    e_p = tracking_error(log.x, log.x_ref)[:, ERR_POS]
    rmse_xyz = rmse(e_p, axis=0)
    return {"euclidean_norm": euclidean_norm_error(e_p),
            "rmse_x": float(rmse_xyz[0]), "rmse_y": float(rmse_xyz[1]),
            "rmse_z": float(rmse_xyz[2])}


def solver_stats(results) -> dict:
    """Fallbacks and step times over all K-MPC runs of one experiment."""
    logs = [r.augmentation.log() for r in results]
    times = np.concatenate([log.solve_time for log in logs])
    return {"solves": int(sum(len(log.t) for log in logs)),
            "fallbacks": int(sum(int((~log.solved).sum()) for log in logs)),
            "step_time_mean_ms": float(1e3 * times.mean()),
            "step_time_max_ms": float(1e3 * times.max())}


def markdown_table(summary: dict) -> str:
    lines = []
    for exp_name, exp in summary["experiments"].items():
        n = exp["comparison"]["euclidean_norm"]["n"]
        lines += [f"## {exp_name} ({n} paired runs)", "",
                  "| metric (m) | LQR | LQR + K-MPC | reduction | 95% CI of difference | p | K-MPC better |",
                  "|---|---|---|---|---|---|---|"]
        for metric in METRICS:
            c = exp["comparison"][metric]
            lines.append(f"| {metric} | {c['baseline_mean']:.4f} | {c['value_mean']:.4f} | "
                         f"{c['percent_reduction']:.2f}% | [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
                         f"{c['p_value']:.3g} | {c['wins']}/{n} |")
        s = exp["kmpc_solver"]
        lines += ["", f"K-MPC: {s['fallbacks']} of {s['solves']} solves fell back to LQR; "
                  f"step time mean {s['step_time_mean_ms']:.2f} ms, max {s['step_time_max_ms']:.2f} ms.",
                  ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Path:
    """Run the evaluation, write results/runs/<time>_<name>/, return that directory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("evaluation", type=Path, help="evaluation TOML file")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "runs")
    args = parser.parse_args(argv)

    evaluation = load_evaluation(args.evaluation)
    kmpc = load_kmpc(evaluation.kmpc)
    seeds = evaluation.run_seeds()
    config_paths = list(evaluation.config_paths) + kmpc.config_paths
    for path in evaluation.experiments:
        exp = load_experiment(path)
        kmpc.check_timing(exp.timing)
        config_paths += exp.config_paths
    config_paths = list(dict.fromkeys(config_paths))  # shared vehicle/controller files once
    run_dir = start_run(args.results_dir, evaluation.name, evaluation.seed, config_paths, REPO_ROOT)
    shutil.copyfile(kmpc.model_run / "identification.json", run_dir / "kmpc_identification.json")
    logger.info("run %s: %d experiments x %d seeds, model %s", run_dir.name,
                len(evaluation.experiments), len(seeds), kmpc.model_run.name)

    controllers = {"lqr": lambda: None,
                   "kmpc": lambda: KMPCAugmentation(kmpc.make_controller())}
    summary = {"seeds": seeds, "model_run": str(kmpc.model_run), "experiments": {}}
    position_errors = {}
    for path in evaluation.experiments:
        exp_name = load_experiment(path).name
        logger.info("%s: running %d seeds", exp_name, len(seeds))
        results = run_paired(lambda seed, path=path: load_experiment(path, seed),
                             controllers, seeds)

        per_run = {}
        for name in CONTROLLERS:
            for i, result in enumerate(results[name]):
                episode_dir = run_dir / "episodes" / exp_name / name / f"run_{i:02d}"
                episode_dir.mkdir(parents=True)
                save_log(episode_dir, result.log)
                if result.augmentation is not None:
                    result.augmentation.log().save(episode_dir / "mpc_log.npz")
            per_run[name] = [run_metrics(r.log) for r in results[name]]
            position_errors[f"{exp_name}/{name}"] = np.array(
                [tracking_error(r.log.x, r.log.x_ref)[:, ERR_POS] for r in results[name]])

        comparison = {
            metric: paired_comparison([m[metric] for m in per_run[BASELINE]],
                                      [m[metric] for m in per_run["kmpc"]])
            for metric in METRICS
        }
        summary["experiments"][exp_name] = {"per_run": per_run, "comparison": comparison,
                                            "kmpc_solver": solver_stats(results["kmpc"])}
        c = comparison["euclidean_norm"]
        logger.info("%s: Euclidean norm LQR %.4f m, K-MPC %.4f m (%.2f%%, p = %.3g, %d/%d better)",
                    exp_name, c["baseline_mean"], c["value_mean"], c["percent_reduction"],
                    c["p_value"], c["wins"], c["n"])

    np.savez(run_dir / "position_errors.npz", **position_errors)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "summary.md").write_text(markdown_table(summary))
    logger.info("saved to %s", run_dir)
    return run_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    main()
