"""Error metrics against hand-computed cases."""
import numpy as np
import pytest
from scipy import stats

from analysis.metrics import (
    euclidean_norm_error, gap_closed, paired_comparison, percent_reduction,
)


def test_euclidean_norm_of_constant_error_is_its_length():
    err = np.tile([0.3, 0.4, 0.0], (50, 1))
    assert euclidean_norm_error(err) == pytest.approx(0.5)


def test_euclidean_norm_equals_norm_of_per_axis_rmse():
    err = np.random.default_rng(0).standard_normal((200, 3))
    rms_of_norms = np.sqrt(np.mean(np.sum(err**2, axis=1)))
    per_axis = np.sqrt(np.mean(err**2, axis=0))
    assert euclidean_norm_error(err) == pytest.approx(rms_of_norms)
    assert euclidean_norm_error(err) == pytest.approx(np.linalg.norm(per_axis))


def test_euclidean_norm_reproduces_paper_table_ii_from_table_iii():
    # LQR row: rmse x 0.6029, z 0.1690 (Table III), norm 0.6812 (Table II) needs
    # rmse y = sqrt(0.6812² − 0.6029² − 0.1690²) ≈ 0.272. A run with exactly these
    # per-axis RMSEs (constant error) gives back 0.6812.
    y = np.sqrt(0.6812**2 - 0.6029**2 - 0.1690**2)
    err = np.tile([0.6029, y, 0.1690], (10, 1))
    assert euclidean_norm_error(err) == pytest.approx(0.6812)


def test_percent_reduction_matches_paper_table_ii():
    assert percent_reduction(0.6812, 0.6288) == pytest.approx(7.69, abs=0.01)
    assert percent_reduction(0.6812, 0.5611) == pytest.approx(17.63, abs=0.01)
    assert percent_reduction(1.0, 1.2) == pytest.approx(-20.0)


def test_percent_reduction_rejects_non_positive_baseline():
    with pytest.raises(ValueError, match="positive"):
        percent_reduction(0.0, 1.0)


def test_paired_comparison_matches_scipy_and_counts_wins():
    rng = np.random.default_rng(1)
    baseline = 1.0 + 0.3 * rng.standard_normal(12)
    value = baseline - 0.05 + 0.02 * rng.standard_normal(12)
    c = paired_comparison(baseline, value)
    d = value - baseline
    assert c["n"] == 12
    assert c["mean_difference"] == pytest.approx(d.mean())
    assert c["p_value"] == pytest.approx(stats.ttest_rel(value, baseline).pvalue)
    half = stats.t.ppf(0.975, 11) * d.std(ddof=1) / np.sqrt(12)
    assert (c["ci_low"], c["ci_high"]) == pytest.approx((d.mean() - half, d.mean() + half))
    assert c["wins"] == int(np.sum(d < 0))


def test_pairing_detects_a_small_consistent_improvement_hidden_by_run_variance():
    # Run-to-run spread (σ = 0.3) is 30x the effect (0.01): unpaired means
    # overlap, but every pair improves, so the paired test is decisive.
    rng = np.random.default_rng(2)
    baseline = 1.0 + 0.3 * rng.standard_normal(10)
    value = baseline - 0.01 + 0.001 * rng.standard_normal(10)
    c = paired_comparison(baseline, value)
    assert c["wins"] == 10
    assert c["ci_high"] < 0
    assert c["p_value"] < 1e-6
    assert stats.ttest_ind(value, baseline).pvalue > 0.5  # unpaired sees nothing


def test_paired_comparison_of_identical_runs_is_no_difference():
    c = paired_comparison([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert c["mean_difference"] == 0.0 and c["p_value"] == 1.0 and c["wins"] == 0


@pytest.mark.parametrize("baseline, value, match", [
    ([1.0], [1.0], "at least 2"),
    ([1.0, 2.0], [1.0], "equal 1-D"),
    ([1.0, np.nan], [1.0, 2.0], "finite"),
])
def test_paired_comparison_rejects_bad_input(baseline, value, match):
    with pytest.raises(ValueError, match=match):
        paired_comparison(baseline, value)


def test_gap_closed_endpoints_and_per_seed():
    assert gap_closed(1.0, 1.0, 0.2) == pytest.approx(0.0)      # no better than nominal
    assert gap_closed(1.0, 0.2, 0.2) == pytest.approx(1.0)      # as good as the oracle
    assert gap_closed(1.0, 0.6, 0.2) == pytest.approx(0.5)
    assert gap_closed(1.0, 1.4, 0.2) == pytest.approx(-0.5)     # worse than nominal
    np.testing.assert_allclose(gap_closed([1.0, 2.0], [0.5, 1.0], [0.0, 1.0]), [0.5, 1.0])


@pytest.mark.parametrize("args", [(1.0, 0.5, 1.0), (1.0, 0.5, 1.5), (1.0, np.nan, 0.2)])
def test_gap_closed_rejects_no_gap_and_non_finite(args):
    with pytest.raises(ValueError):
        gap_closed(*args)
