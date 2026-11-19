"""Figures and tables for the slides (presentation/slides.typ). Reads saved runs only.

    PYTHONPATH=src python presentation/make_figures.py \
        --paper-run results/runs/<time>_kmpc_vs_lqr \
        --mc-run results/runs/<time>_presentation_mc \
        --gap-run results/runs/<time>_gap_closure \
        --model-run results/runs/<time>_outer_loop_koopman \
        --greybox-run results/runs/<time>_greybox_drag \
        --dataset-run results/runs/<time>_outer_loop_data

Writes presentation/figures/*.svg and presentation/data/*.json. Every number the
slides show is read from data/*.json, which records the run it came from.

Frames and units on every plot: position/velocity world NED converted to
North-East-Up for display (altitude = −p_z); Euler angles ZYX (roll, pitch, yaw),
plotted in degrees; errors are state − reference.
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from datagen.outer_loop import load_episodes  # noqa: E402
from dynamics.quaternion import Quaternion  # noqa: E402
from experiment.builders import load_experiment  # noqa: E402
from koopman.excitation import hankel_matrix, persistency_of_excitation  # noqa: E402
from koopman.outer_model import OuterKoopmanModel  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
FIG, DATA = HERE / "figures", HERE / "data"

# Okabe-Ito palette (colour-blind safe); one colour per controller across all slides.
COLORS = {"LTV nominal": "#7f7f7f", "Koopman": "#0072B2", "LTV grey-box": "#E69F00",
          "LTV true drag": "#009E73", "LQR": "#D55E00", "LQR + K-MPC": "#0072B2",
          "K-Res-MPC": "#CC79A7"}
DISPLAY = {"LTV nominal": "LTV nominal (no wind model)", "Koopman": "Koopman MPC",
           "LTV grey-box": "LTV grey-box drag", "LTV true drag": "LTV oracle (true drag)",
           "LQR": "LQR", "LQR + K-MPC": "LQR + K-MPC", "K-Res-MPC": "LQR + K-Res-MPC"}
FULL_MPC = ("LTV nominal", "Koopman", "LTV grey-box", "LTV true drag")
PAPER = ("LQR", "K-Res-MPC")    # the paper's architecture, from run_paper_controllers.py
AXES = {"pos": (("north", "east", "down"), "m", "position error"),
        "vel": (("north", "east", "down"), "m/s", "velocity error"),
        "eul": (("roll", "pitch", "yaw"), "deg", "attitude error")}
INPUT_LABELS = (r"$\omega_x$ [rad/s]", r"$\omega_y$ [rad/s]", r"$\omega_z$ [rad/s]",
                r"$c$ [m/s$^2$]")
SAT_TOL = 1e-3   # a command within this fraction of the bound range counts as saturated

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 13, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "legend.frameon": False, "svg.fonttype": "path",
    "figure.dpi": 100, "savefig.bbox": "tight",
})


# ----------------------------------------------------------------------------- math helpers
def euler(q: np.ndarray) -> np.ndarray:
    """Quaternions (..., 4), w first -> ZYX Euler angles (..., 3), rad."""
    flat = q.reshape(-1, 4)
    return np.array([Quaternion(row).to_euler() for row in flat]).reshape(*q.shape[:-1], 3)


def wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def state_errors(x: np.ndarray, x_ref: np.ndarray) -> dict[str, np.ndarray]:
    """13-state x and 10-state x_ref (..., T, ·) -> errors {pos m, vel m/s, eul deg}."""
    return {"pos": x[..., 0:3] - x_ref[..., 0:3],
            "vel": x[..., 7:10] - x_ref[..., 7:10],
            "eul": np.degrees(wrap(euler(x[..., 3:7]) - euler(x_ref[..., 3:7])))}


def rmse_3d(err: np.ndarray) -> np.ndarray:
    """Errors (S, T, 3) -> per-seed RMSE of the error norm (S,)."""
    return np.sqrt(np.mean(np.sum(err ** 2, axis=-1), axis=-1))


def neu(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NED positions (T, 3) -> (north, east, up)."""
    return p[:, 0], p[:, 1], -p[:, 2]


def save(fig: plt.Figure, name: str) -> None:
    """SVG for the slides (Typst), PDF for the report (pdflatex cannot include SVG)."""
    fig.savefig(FIG / f"{name}.svg")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- plot builders
def plot_3d(ref: np.ndarray, flown: dict[str, np.ndarray], title: str) -> plt.Figure:
    """Reference (T, 3) and flown paths {label: (T, 3)}, NED in, North-East-Up shown."""
    fig = plt.figure(figsize=(6.2, 5.0))
    ax = fig.add_subplot(projection="3d")
    ax.plot(*neu(ref), "k--", lw=1.6, label="reference")
    for label, p in flown.items():
        ax.plot(*neu(p), color=COLORS[label], lw=1.4, label=DISPLAY[label])
    ax.set_xlabel("north [m]")
    ax.set_ylabel("east [m]")
    ax.set_zlabel("altitude [m]")
    ax.set_title(title)
    ax.view_init(elev=28, azim=-60)
    ax.legend(loc="upper left", fontsize=10.5, bbox_to_anchor=(-0.05, 1.02))
    return fig


def plot_error_bands(t: np.ndarray, errors: dict[str, np.ndarray], kind: str,
                     title: str) -> plt.Figure:
    """Mean ± 1 std over seeds of one error kind; errors {label: (S, T, 3)}."""
    names, unit, what = AXES[kind]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7.6, 6.0))
    for label, err in errors.items():
        mean, std = err.mean(axis=0), err.std(axis=0)
        for i, ax in enumerate(axes):
            ax.fill_between(t, mean[:, i] - std[:, i], mean[:, i] + std[:, i],
                            color=COLORS[label], alpha=0.2, lw=0)
            ax.plot(t, mean[:, i], color=COLORS[label], lw=1.4, label=DISPLAY[label])
    for i, ax in enumerate(axes):
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(f"{names[i]} [{unit}]")
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(ncol=2, fontsize=10.5, loc="upper right")
    n_seeds = next(iter(errors.values())).shape[0]
    fig.suptitle(f"{title}: {what} (mean ± 1 std, {n_seeds} wind seeds)")
    fig.tight_layout()
    return fig


def plot_inputs(t: np.ndarray, u: dict[str, np.ndarray], u_lb, u_ub, title: str) -> plt.Figure:
    """Commands of one seed, rows = channel, columns = controller, bounds dashed."""
    labels = list(u)
    fig, axes = plt.subplots(4, len(labels), sharex=True, sharey="row",
                             figsize=(2.6 * len(labels), 5.6))
    for j, label in enumerate(labels):
        for i in range(4):
            ax = axes[i, j]
            ax.step(t, u[label][:, i], where="post", color=COLORS[label], lw=1.1)
            for bound in (u_lb[i], u_ub[i]):
                ax.axhline(bound, color="k", ls="--", lw=0.7)
            if j == 0:
                ax.set_ylabel(INPUT_LABELS[i])
        axes[0, j].set_title(DISPLAY[label], fontsize=12)
        axes[-1, j].set_xlabel("time [s]")
    fig.suptitle(f"{title}: MPC commands, seed 0 (dashed: MPC bounds)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- 1. paper (LQR + K-MPC)
def paper_figures(paper_run: Path) -> dict:
    """Reproduction of the CCTA paper: LQR vs LQR + K-MPC (Model 1), all logged runs."""
    out = {"source": str(paper_run)}
    for exp_dir in sorted((paper_run / "episodes").iterdir()):
        name = exp_dir.name
        logs = {}
        for key, label in (("lqr", "LQR"), ("kmpc", "LQR + K-MPC")):
            runs = sorted((exp_dir / key).iterdir())
            logs[label] = [np.load(r / "log.npz") for r in runs]
        t = logs["LQR"][0]["t"]
        errors = {label: state_errors(np.stack([l["x"] for l in ls]),
                                      np.stack([l["x_ref"] for l in ls]))
                  for label, ls in logs.items()}
        title = name.replace("_gaussian", "").replace("figure8", "figure-8")
        save(plot_3d(logs["LQR"][0]["x_ref"][:, 0:3],
                     {label: ls[0]["x"][:, 0:3] for label, ls in logs.items()},
                     f"Paper set-up, {title} (seed 0)"), f"paper_{name}_3d")
        for kind in AXES:
            save(plot_error_bands(t, {label: e[kind] for label, e in errors.items()}, kind,
                                  f"LQR vs LQR + K-MPC, {title}"), f"paper_{name}_err_{kind}")
        out[name] = {"runs": len(logs["LQR"]), "rmse": {
            label: {kind: {"mean": float(rmse_3d(e[kind]).mean()),
                           "std": float(rmse_3d(e[kind]).std())} for kind in AXES}
            for label, e in errors.items()}}
    return out


# ----------------------------------------------------------------------------- 2. full MPC Monte Carlo
def input_metrics(u: np.ndarray, u_lb, u_ub) -> dict:
    """Chattering / bang-bang indicators of commands u (S, T, 4).

    tv: mean |Δu| per step divided by the bound range (0: constant, 1: jumps
        across the whole box every step); sat: fraction of steps at a bound;
        flips: fraction of steps where Δu changes sign (oscillation).
    """
    span = np.asarray(u_ub) - np.asarray(u_lb)
    du = np.diff(u, axis=1)
    at_bound = (np.abs(u - u_lb) < SAT_TOL * span) | (np.abs(u - u_ub) < SAT_TOL * span)
    big = np.abs(du) > 1e-3 * span
    flips = (np.sign(du[:, 1:]) * np.sign(du[:, :-1]) < 0) & big[:, 1:] & big[:, :-1]
    return {"tv": (np.abs(du) / span).mean(axis=(0, 1)).tolist(),
            "sat": at_bound.mean(axis=(0, 1)).tolist(),
            "flips": flips.mean(axis=(0, 1)).tolist()}


def _merge_paper(d, keep: list[int], paper_run: Path | None, name: str):
    """(controllers, x, x_ref, u) of the full MPCs, with the paper's controllers prepended
    when paper_run is given. Raises unless both runs flew the same seeds, reference and wind."""
    x, x_ref, u = d["x"][:, keep], d["x_ref"][:, keep], d["u_cmd"][:, keep]
    if paper_run is None:
        return FULL_MPC, x, x_ref, u
    p = np.load(paper_run / f"{name}.npz")
    labels = [str(s) for s in p["labels"]]
    idx = [labels.index(lb) for lb in PAPER]
    if not (np.array_equal(p["seeds"], d["seeds"]) and np.allclose(p["t"], d["t"])):
        raise ValueError(f"{name}: paper and full-MPC runs have different seeds or times")
    if not np.allclose(p["v_w"][:, idx], d["v_w"][:, keep[:1]].repeat(len(idx), axis=1)):
        raise ValueError(f"{name}: paper and full-MPC runs did not fly the same wind")
    if not np.allclose(p["x_ref"][:, idx, :, 0:3], x_ref[:, :1, :, 0:3]):
        raise ValueError(f"{name}: paper and full-MPC runs have different references")
    return (PAPER + FULL_MPC,
            np.concatenate([p["x"][:, idx], x], axis=1),
            np.concatenate([p["x_ref"][:, idx], x_ref], axis=1),
            np.concatenate([p["u_cmd"][:, idx], u], axis=1))


def full_mpc_figures(mc_run: Path, paper_run: Path | None = None) -> dict:
    out = {"source": str(mc_run), "paper_source": str(paper_run) if paper_run else None}
    for path in sorted(mc_run.glob("*.npz")):
        name = path.stem
        d = np.load(path)
        labels = [str(s) for s in d["labels"]]
        keep = [labels.index(lb) for lb in FULL_MPC]
        controllers, x, x_ref, u = _merge_paper(d, keep, paper_run, name)
        t = d["t"]
        errors = state_errors(x, x_ref)            # (S, C, T, 3) each
        title = (name.replace("_gaussian", "").replace("figure8_", "figure-8 ")
                 .replace("circle_", "circle ").replace("s", " s"))
        save(plot_3d(x_ref[0, -1, :, 0:3], {lb: x[0, c, :, 0:3] for c, lb in enumerate(controllers)},
                     f"{title} (seed 0)"), f"mc_{name}_3d")
        for kind in AXES:
            save(plot_error_bands(t, {lb: errors[kind][:, c] for c, lb in enumerate(controllers)},
                                  kind, title), f"mc_{name}_err_{kind}")
        save(plot_inputs(t, {lb: u[0, c] for c, lb in enumerate(controllers)}, d["u_lb"], d["u_ub"],
                         title), f"mc_{name}_inputs")
        out[name] = {"title": title, "seeds": int(x.shape[0]), "controllers": {
            lb: {**{f"rmse_{kind}": {"mean": float(rmse_3d(errors[kind][:, c]).mean()),
                                     "std": float(rmse_3d(errors[kind][:, c]).std())}
                    for kind in AXES},
                 "inputs": input_metrics(u[:, c], d["u_lb"], d["u_ub"])}
            for c, lb in enumerate(controllers)}}
    return out


def gap_table(gap_run: Path, paper_run: Path | None = None) -> dict:
    """Position RMSE on every reference of a gap_closure run: its four full-MPC
    controllers, plus LQR and LQR + K-Res-MPC from paper_run on the same seeds."""
    summary = json.loads((gap_run / "summary.json").read_text())
    rows = {}
    for name, v in summary.items():
        if not (isinstance(v, dict) and "rmse_3d_m" in v):
            continue
        rows[name] = {lb: {"mean": v["rmse_3d_m"][lb]["mean"], "std": v["rmse_3d_m"][lb]["std"],
                           "reduction_pct": v["reduction_from_nominal_pct"][lb]}
                      for lb in FULL_MPC}
        if paper_run is not None:
            p = np.load(paper_run / f"{name}.npz")
            seeds = list(p["seeds"])
            sel = [seeds.index(s) for s in summary["seeds"]]   # same wind as gap_closure
            labels = [str(s) for s in p["labels"]]
            nominal = v["rmse_3d_m"]["LTV nominal"]["mean"]
            for lb in PAPER:
                c = labels.index(lb)
                err = p["x"][sel, c, :, 0:3] - p["x_ref"][sel, c, :, 0:3]
                r = rmse_3d(err)
                rows[name][lb] = {"mean": float(r.mean()), "std": float(r.std()),
                                  "reduction_pct": float(100 * (nominal - r.mean()) / nominal)}
    return {"source": str(gap_run), "paper_source": str(paper_run) if paper_run else None,
            "seeds": len(summary["seeds"]), "rows": rows,
            "greybox_drag_coeffs": summary["greybox_drag_coeffs"]}


# ----------------------------------------------------------------------------- 3. training data and PE
def training_figures(dataset_run: Path, model_run: Path, mc_run: Path) -> dict:
    train = load_episodes(dataset_run)["train"]
    model = OuterKoopmanModel.load(model_run)

    # Training paths (top view and altitude), with the unseen test references on top.
    fig, (ax_top, ax_alt) = plt.subplots(1, 2, figsize=(11, 4.4),
                                         gridspec_kw={"width_ratios": [1.1, 1]})
    for ep in train:
        p = ep.x9[:, 0:3]
        hover = ep.name.startswith("hover")
        style = dict(color="#E69F00" if hover else "#7f7f7f", lw=0.8, alpha=0.8)
        ax_top.plot(p[:, 1], p[:, 0], **style)
        ax_alt.plot(np.arange(len(p)) * model.dt, -p[:, 2], **style)
    test_colors = {"circle_7s_gaussian": "#0072B2", "figure8_10s_gaussian": "#009E73"}
    for path in sorted(mc_run.glob("*.npz")):
        ref = np.load(path)["x_ref"][0, 0, :, 0:3]
        ax_top.plot(ref[:, 1], ref[:, 0], color=test_colors.get(path.stem, "k"), lw=2.2,
                    label=f"test: {path.stem.replace('_gaussian', '')}")
    ax_top.plot([], [], color="#7f7f7f", label="train: random sum-of-sines (30)")
    ax_top.plot([], [], color="#E69F00", label="train: hover from offset (6)")
    ax_top.set_xlabel("east [m]")
    ax_top.set_ylabel("north [m]")
    ax_top.set_aspect("equal", adjustable="datalim")
    ax_top.legend(fontsize=10, loc="upper left")
    ax_top.set_title("top view")
    ax_alt.set_xlabel("time [s]")
    ax_alt.set_ylabel("altitude [m]")
    ax_alt.set_title("altitude of every training episode")
    fig.tight_layout()
    save(fig, "train_trajectories")

    # Coverage: per-episode peak tilt and speed, training vs the test references.
    meta = json.loads((dataset_run / "episodes.json").read_text())
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    tr = [e for e in meta if e["split"] == "train"]
    ax.scatter([e["max_speed_m_s"] for e in tr], np.degrees([e["max_tilt_rad"] for e in tr]),
               c=["#E69F00" if e["name"].startswith("hover") else "#7f7f7f" for e in tr],
               s=28, label="training episodes")
    for path in sorted(mc_run.glob("*.npz")):
        ref = np.load(path)["x_ref"][0, 0]
        speed = np.linalg.norm(ref[:, 7:9], axis=1).max()
        eul = euler(ref[:, 3:7])
        tilt = np.degrees(np.abs(eul[:, :2]).max())
        ax.scatter([speed], [tilt], marker="*", s=220, color=test_colors.get(path.stem, "k"),
                   label=f"test: {path.stem.replace('_gaussian', '')}", zorder=3)
    ax.set_xlabel("peak horizontal speed [m/s]")
    ax.set_ylabel("peak tilt [deg]")
    ax.set_title("test references lie inside the training envelope")
    ax.legend(fontsize=10)
    fig.tight_layout()
    save(fig, "train_coverage")

    # Persistency of excitation: mosaic Hankel singular values of δu = u − u*.
    du = [ep.u - model.u_trim for ep in train]
    n_state, n_lift = model.state_dim, model.A.shape[0]
    orders = {f"L = n+1 = {n_state + 1}": n_state + 1, f"L = N+1 = {n_lift + 1}": n_lift + 1}
    pe = {}
    fig, (ax_sv, ax_demo) = plt.subplots(1, 2, figsize=(11, 4.0))
    for (label, L), color in zip(orders.items(), ("#0072B2", "#D55E00")):
        H = np.hstack([hankel_matrix(u, L) for u in du if len(u) >= L])
        s = np.linalg.svd(H, compute_uv=False)
        pe[label] = persistency_of_excitation(du, L)
        ax_sv.semilogy(np.arange(1, len(s) + 1) / len(s), s / s[0], color=color,
                       label=f"{label}: rank {pe[label]['rank']}/{pe[label]['rows']}")
    ax_sv.axhline(1e-9, color="k", ls="--", lw=0.8, label="rank threshold 1e-9")
    ax_sv.set_xlabel("singular value index / number of rows")
    ax_sv.set_ylabel(r"$\sigma_i / \sigma_{\max}$")
    ax_sv.set_title(r"training inputs $\delta u$: mosaic Hankel spectrum")
    ax_sv.legend(fontsize=8, loc="lower left")
    ax_sv.set_ylim(1e-11, 2)

    # Same test on textbook signals: which inputs would have failed.
    rng = np.random.default_rng(0)
    T, L = 140, 13
    k = np.arange(T)[:, None]
    demo = {"constant": np.ones((T, 4)),
            "1 sine / channel": np.sin(0.3 * k + np.arange(4)),
            "3 sines / channel": sum(np.sin(w * k + np.arange(4)) for w in (0.2, 0.5, 0.9)),
            "uniform dither": rng.uniform(-1, 1, (T, 4)),
            "our data (36 ep.)": None}
    ranks = {}
    for name, u in demo.items():
        r = pe[f"L = n+1 = {n_state + 1}"] if u is None else persistency_of_excitation([u], L)
        ranks[name] = r["rank"]
    ax_demo.bar(range(len(ranks)), ranks.values(),
                color=["#bbbbbb"] * 4 + ["#0072B2"])
    ax_demo.axhline(4 * L, color="k", ls="--", lw=0.8, label=f"full row rank l·L = {4 * L}")
    ax_demo.set_xticks(range(len(ranks)), ranks.keys(), rotation=20, fontsize=9)
    ax_demo.set_ylabel(f"rank $H_{{{L}}}(u)$")
    ax_demo.set_title(f"same test (L = {L}) on example signals")
    ax_demo.legend(fontsize=8, loc="upper left")
    for i, r in enumerate(ranks.values()):
        ax_demo.text(i, r + 1, str(r), ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "pe_hankel")

    ident = json.loads((model_run / "identification.json").read_text())
    return {"source_dataset": str(dataset_run), "source_model": str(model_run),
            "train_episodes": len(train), "train_samples": int(sum(len(ep.u) for ep in train)),
            "dt": model.dt, "lifted_dim": n_lift, "state_dim": n_state,
            "features": ident["features"], "pe": pe, "pe_stored": ident["persistency_of_excitation"],
            "regressor": ident["regressor"], "demo_ranks": ranks,
            "rmse_20_steps_mean_wind": ident["rmse_at_20_steps_mean_wind"],
            "rmse_1_step_mean_wind": ident["rmse_at_1_step_mean_wind"]}


# ----------------------------------------------------------------------------- 4. commanded vs BEMT thrust
def thrust_figures(mc_run: Path, model_run: Path, greybox_run: Path) -> dict:
    """Commanded vs applied (BEMT) thrust on the flown test trajectories, and how much of
    the gap each model's feature set can represent (least-squares R²)."""
    path = mc_run / "figure8_10s_gaussian.npz"
    d = np.load(path)
    labels = [str(s) for s in d["labels"]]
    exp = load_experiment(REPO_ROOT / "configs/experiments/figure8_10s_gaussian.toml")
    model = OuterKoopmanModel.load(model_run)
    mass = exp.params.mass

    def applied_thrust(x, u, v_w):
        f_cmd = np.empty(len(x))
        f_app = np.empty(len(x))
        for i in range(len(x)):
            f_t, tau_t = exp.rate_controller(x[i], u[i])
            f, _ = exp.actuator(x[i], f_t, tau_t, v_w[i])
            f_cmd[i], f_app[i] = -f_t[2], -f[2]
        return f_cmd, f_app

    c = labels.index("Koopman")
    t = d["t"]
    f_cmd, f_app = applied_thrust(d["x"][0, c], d["u_cmd"][0, c], d["v_w"][0, c])
    airspeed = np.linalg.norm(d["x"][0, c, :, 7:10] - d["v_w"][0, c], axis=1)

    fig, (ax_f, ax_r) = plt.subplots(2, 1, sharex=True, figsize=(8.5, 5.0))
    ax_f.plot(t, f_cmd, color="#7f7f7f", lw=1.4, label=r"commanded $m\,c$ (quadratic rotor model)")
    ax_f.plot(t, f_app, color="#D55E00", lw=1.4, label="applied (BEMT in local airflow)")
    ax_f.set_ylabel("thrust [N]")
    ax_f.legend(fontsize=9, loc="upper right")
    ax_r.plot(t, 100 * (f_app / f_cmd - 1), color="#D55E00", lw=1.4, label="thrust error [%]")
    ax_r.set_ylabel("applied / commanded − 1 [%]")
    ax2 = ax_r.twinx()
    ax2.plot(t, airspeed, color="#0072B2", lw=1.0, alpha=0.7)
    ax2.set_ylabel("airspeed [m/s]", color="#0072B2")
    ax2.grid(False)
    ax_r.set_xlabel("time [s]")
    fig.suptitle("Figure-8 10 s, Koopman MPC, seed 0: the rotors do not deliver the command")
    fig.tight_layout()
    save(fig, "thrust_cmd_vs_bemt")

    # Which feature sets can represent the thrust gap? Pool every seed and controller.
    xs = d["x"].reshape(-1, d["x"].shape[-2], 13)
    us = d["u_cmd"].reshape(-1, *d["u_cmd"].shape[-2:])
    ws = d["v_w"].reshape(-1, *d["v_w"].shape[-2:])
    gaps, koop_feats, grey_feats, true_feats = [], [], [], []
    for x, u, w in zip(xs, us, ws):
        fc, fa = applied_thrust(x, u, w)
        gaps.append((fa - fc) / mass)                          # m/s², along −z_body
        koop_feats.append(np.array([model.lift_measurement(xi) for xi in x]))
        for feats, wind in ((grey_feats, np.broadcast_to(model.wind_mean, w.shape)),
                            (true_feats, w)):
            R = np.array([Quaternion(xi[3:7]).to_rotation_matrix() for xi in x])
            v_rb = np.einsum("tji,tj->ti", R, x[:, 7:10] - wind)   # body-frame airspeed
            # The grey-box's only body-z force term is d_z v_rb,z |v_rb| (one coefficient).
            feats.append(np.column_stack([v_rb[:, 2] * np.linalg.norm(v_rb, axis=1),
                                          np.ones(len(x))]))   # + constant, as Koopman has
    y = np.concatenate(gaps)

    def r2(F):
        coef, *_ = np.linalg.lstsq(F, y, rcond=None)
        return float(1 - np.sum((y - F @ coef) ** 2) / np.sum((y - y.mean()) ** 2))

    fit = {"koopman_dictionary_mean_wind": r2(np.concatenate(koop_feats)),
           "greybox_drag_structure_mean_wind": r2(np.concatenate(grey_feats)),
           "greybox_drag_structure_true_wind": r2(np.concatenate(true_feats)),
           "quadratic_rotor_model": 0.0}
    greybox = json.loads((greybox_run / "greybox_drag.json").read_text())
    return {"source": str(path), "thrust_gap_rms_m_s2": float(np.sqrt(np.mean(y ** 2))),
            "thrust_gap_range_pct": [float(100 * (f_app / f_cmd - 1).min()),
                                     float(100 * (f_app / f_cmd - 1).max())],
            "r2_of_thrust_gap": fit, "samples": int(len(y)),
            "greybox_drag_coeffs": greybox["drag_coeffs"],
            "true_drag_coeffs": greybox["simulator_drag_coeffs_for_reference"],
            "greybox_residual_rms_validation_m_s2": greybox["residual_rms_m_s2"]["validation"]}


# ----------------------------------------------------------------------------- 5. eigenvalues
EIG_PERTURBATION = 1e-4   # ‖E‖₂ of the random perturbations of A (absolute, both models)
EIG_DRAWS = 30


def eigen_figures(model_run: Path) -> dict:
    """Discrete-time eigenvalues (poles) of the Koopman A and of the hover linearisations,
    and how far they move when A is perturbed slightly.

    Hover models: the nominal one (no drag; LTV nominal's model at hover) and the
    true-drag one in mean wind (the physics the Koopman model should recover).
    Eigenvalue condition number κ_i = ‖w_i‖‖v_i‖ / |w_iᴴ v_i| (left/right eigenvectors):
    a perturbation of size ε can move λ_i by about κ_i ε. Clouds: eigenvalues of A + E for
    random E with ‖E‖₂ = EIG_PERTURBATION, far below the least-squares estimation error.
    Transmission zeros are not shown: with 4 inputs and 9 outputs the system is tall
    and generically has none.
    """
    import scipy.linalg as sla
    from experiment.ltv_mpc_config import load_ltv_mpc
    model = OuterKoopmanModel.load(model_run)
    params = load_experiment(REPO_ROOT / "configs/experiments/circle_7s_gaussian.toml").params
    hover_A = {k: load_ltv_mpc(REPO_ROOT / f"configs/controllers/{f}").hover_model(params).A
               for k, f in (("nominal", "ltv_mpc_nominal.toml"), ("true drag", "ltv_mpc.toml"))}
    hover_eig = {k: np.linalg.eigvals(A) for k, A in hover_A.items()}
    lam, VL, VR = sla.eig(model.A, left=True, right=True)
    kappa = (np.linalg.norm(VL, axis=0) * np.linalg.norm(VR, axis=0)
             / np.abs(np.sum(VL.conj() * VR, axis=0)))
    dt = model.dt

    rng = np.random.default_rng(0)

    def perturbed(A):
        eigs = []
        for _ in range(EIG_DRAWS):
            E = rng.standard_normal(A.shape)
            eigs.append(np.linalg.eigvals(A + E * EIG_PERTURBATION / np.linalg.norm(E, 2)))
        return np.array(eigs)

    cloud = {"Koopman": perturbed(model.A), "hover true drag": perturbed(hover_A["true drag"])}

    fig, (ax_u, ax_z) = plt.subplots(1, 2, figsize=(14, 4.9), gridspec_kw={"width_ratios": [1, 1.25]})
    theta = np.linspace(0, 2 * np.pi, 400)
    for ax in (ax_u, ax_z):
        ax.plot(np.cos(theta), np.sin(theta), "k", lw=0.8)
        ax.scatter(cloud["Koopman"].real, cloud["Koopman"].imag, s=4, color="#9ecae1",
                   alpha=0.5, lw=0, zorder=1,
                   label=rf"Koopman, $A + E$, $\|E\|_2 = 10^{{{int(np.log10(EIG_PERTURBATION))}}}$")
        ax.scatter(cloud["hover true drag"].real, cloud["hover true drag"].imag, s=4,
                   color="#fdae6b", alpha=0.6, lw=0, zorder=1, label="hover true drag, same $E$ size")
        sc = ax.scatter(lam.real, lam.imag, c=np.log10(kappa), cmap="viridis", vmin=2, vmax=15,
                        s=50, marker="x", lw=1.7, zorder=4, label=f"Koopman ({len(lam)} modes)")
        ax.scatter(hover_eig["nominal"].real, hover_eig["nominal"].imag, s=160, facecolors="none",
                   edgecolors="#7f7f7f", lw=1.6, label="hover LTI, no drag (9 × z = 1)", zorder=3)
        ax.scatter(hover_eig["true drag"].real, hover_eig["true drag"].imag, s=70, marker="D",
                   facecolors="none", edgecolors="#D55E00", lw=1.5,
                   label="hover LTI, true drag in mean wind", zorder=3)
        ax.set_xlabel("Re z")
        ax.set_aspect("equal")
    ax_u.set_ylabel("Im z")
    ax_u.set_xlim(-1.9, 1.9)
    ax_u.set_ylim(-1.9, 1.9)
    ax_u.set_title("all modes (unit circle black)")
    ax_u.legend(fontsize=9.5, loc="center left", bbox_to_anchor=(-1.05, 0.5), markerscale=2)
    ax_z.set_xlim(0.90, 1.03)
    ax_z.set_ylim(-0.065, 0.065)
    for sigma in (0.1, 0.3, 1.0):   # rings of constant decay rate σ: |z| = exp(−σ dt)
        r = np.exp(-sigma * dt)
        ax_z.plot(r * np.cos(theta), r * np.sin(theta), color="#bbbbbb", lw=0.7, ls="--")
        ax_z.text(r - 0.001, 0.056, f"{sigma:g}/s", fontsize=8.5, color="#888888", ha="right")
    ax_z.set_title("zoom near z = 1 (dashed: decay rate)")
    cb = fig.colorbar(sc, ax=ax_z, shrink=0.85)
    cb.set_label(r"$\log_{10}\kappa$ (eigenvalue condition number)")
    fig.tight_layout()
    save(fig, "eigenvalues")

    order = np.argsort(-np.abs(lam))
    unstable = [float(abs(lam[i])) for i in order if abs(lam[i]) > 1 + 1e-9]
    radius = {k: [float(np.abs(c).max(axis=1).min()), float(np.abs(c).max(axis=1).max())]
              for k, c in cloud.items()}
    return {"source": str(model_run), "dt": dt, "n_modes": int(len(lam)),
            "spectral_radius": float(np.abs(lam).max()),
            "n_unstable": len(unstable), "unstable_moduli": unstable,
            "unstable_doubling_time_s": [float(dt * np.log(2) / np.log(m)) for m in unstable],
            "growth_over_horizon_20": [float(m ** 20) for m in unstable],
            "n_on_unit_circle": int(np.sum(np.abs(np.abs(lam) - 1) < 1e-6)),
            "slow_moduli_range": [float(np.abs(lam[order[5:12]]).min()),
                                  float(np.abs(lam[order[5:12]]).max())],
            "hover_nominal": sorted(np.abs(hover_eig["nominal"]).tolist(), reverse=True),
            "hover_true_drag": sorted(np.abs(hover_eig["true drag"]).tolist(), reverse=True),
            "norm_A": float(np.linalg.norm(model.A, 2)), "max_abs_A": float(np.abs(model.A).max()),
            "kappa_median": float(np.median(kappa)), "kappa_max": float(kappa.max()),
            "perturbation_norm": EIG_PERTURBATION,
            "perturbed_spectral_radius": radius}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("paper", "mc", "gap", "model", "greybox", "dataset", "bemt", "paper-bemt"):
        parser.add_argument(f"--{name}-run", type=Path, required=name not in ("bemt", "paper-bemt"),
                            help="paper-bemt: run_paper_controllers.py run (adds LQR, K-Res-MPC)"
                            if name == "paper-bemt" else None)
    args = parser.parse_args()
    FIG.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    out = {"paper": paper_figures(args.paper_run),
           "full_mpc": full_mpc_figures(args.mc_run, args.paper_bemt_run),
           "gap": gap_table(args.gap_run, args.paper_bemt_run),
           "training": training_figures(args.dataset_run, args.model_run, args.mc_run),
           "thrust": thrust_figures(args.mc_run, args.model_run, args.greybox_run),
           "eigen": eigen_figures(args.model_run)}
    if args.bemt_run is not None:
        shutil.copy(args.bemt_run / "figures" / "rotor_bemt.png", FIG / "rotor_bemt.png")
    for key, value in out.items():
        (DATA / f"{key}.json").write_text(json.dumps(value, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "training"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
