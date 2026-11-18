"""Error metrics: RMSE (Tables I, III), Euclidean norm (Table II), MSE (Fig. 4)."""
import numpy as np
from scipy import stats


def rmse(err, axis=0) -> np.ndarray:
    """Root-mean-square of err along axis, in err's units."""
    err = np.asarray(err, dtype=float)
    if err.size == 0 or not np.all(np.isfinite(err)):
        raise ValueError("err must be non-empty and finite")
    return np.sqrt(np.mean(err**2, axis=axis))


def euclidean_norm_error(err) -> float:
    """RMS over time of ‖e_p(t)‖, err shape (T, 3); equals ‖per-axis RMSE‖.

    The paper does not define Table II's metric; this reading is inferred
    from it being consistent with Table III (LQR: 0.6812 vs rmse x 0.6029,
    z 0.1690, and an unreported y of about 0.27).
    """
    err = np.asarray(err, dtype=float)
    if err.ndim != 2:
        raise ValueError(f"err must have shape (T, k), got {err.shape}")
    return float(np.linalg.norm(rmse(err, axis=0)))


def percent_reduction(baseline, value):
    """100 (baseline − value) / baseline; positive when value is smaller."""
    baseline = np.asarray(baseline, dtype=float)
    value = np.asarray(value, dtype=float)
    if np.any(baseline <= 0) or not (np.all(np.isfinite(baseline)) and np.all(np.isfinite(value))):
        raise ValueError("baseline must be positive and both inputs finite")
    return 100.0 * (baseline - value) / baseline


def paired_comparison(baseline, value, confidence: float = 0.95) -> dict:
    """Compare per-run metrics of two controllers run on the same wind seeds.

    Pairing removes the run-to-run wind variation from the comparison: the
    statistic is the per-run difference d_i = value_i − baseline_i, tested
    with a paired t-test (H0: mean d = 0).

    Args:
        baseline, value: Metric per run, shape (n,), same run order, n ≥ 2.
        confidence: Level of the confidence interval on the mean difference.

    Returns:
        n, baseline_mean, value_mean, mean_difference, ci_low, ci_high (of
        the mean difference), percent_reduction (of the means), p_value,
        wins (runs where value < baseline).
    """
    baseline = np.asarray(baseline, dtype=float)
    value = np.asarray(value, dtype=float)
    if baseline.ndim != 1 or baseline.shape != value.shape or len(baseline) < 2:
        raise ValueError(f"need two equal 1-D arrays of at least 2 runs, got "
                         f"{baseline.shape} and {value.shape}")
    if not (np.all(np.isfinite(baseline)) and np.all(np.isfinite(value))):
        raise ValueError("baseline and value must be finite")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    d = value - baseline
    n = len(d)
    sem = d.std(ddof=1) / np.sqrt(n)
    half_width = stats.t.ppf(0.5 + confidence / 2, df=n - 1) * sem
    # A zero-variance difference gives NaN from ttest_rel; report it as decided.
    p_value = float(stats.ttest_rel(value, baseline).pvalue) if sem > 0 else (
        1.0 if d.mean() == 0 else 0.0)
    return {
        "n": n,
        "baseline_mean": float(baseline.mean()),
        "value_mean": float(value.mean()),
        "mean_difference": float(d.mean()),
        "ci_low": float(d.mean() - half_width),
        "ci_high": float(d.mean() + half_width),
        "percent_reduction": float(percent_reduction(baseline.mean(), value.mean())),
        "p_value": p_value,
        "wins": int(np.sum(d < 0)),
    }


def gap_closed(nominal, learned, oracle):
    """Fraction of the nominal-to-oracle error gap a learned model closes.

        g = (nominal − learned) / (nominal − oracle)

    g = 0: no better than the nominal model; g = 1: as good as the oracle;
    g < 0: worse than nominal; g > 1: better than the oracle. Element-wise, so
    per-seed RMSE arrays give per-seed fractions.

    Args:
        nominal, learned, oracle: Errors of the three controllers, same units
            and broadcastable shapes (e.g. RMSE in m per seed).

    Raises:
        ValueError: If any value is not finite, or the oracle is not strictly
            better than the nominal (there is no gap, so the ratio is undefined).
    """
    nominal, learned, oracle = (np.asarray(x, dtype=float) for x in (nominal, learned, oracle))
    if not all(np.all(np.isfinite(x)) for x in (nominal, learned, oracle)):
        raise ValueError("errors must be finite")
    gap = nominal - oracle
    if np.any(gap <= 0):
        raise ValueError(f"the oracle must be better than the nominal model, gap {gap}")
    return (nominal - learned) / gap
