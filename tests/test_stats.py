"""Phase 11: paired statistics."""

import pytest

from t2sql.eval.stats import bootstrap_ci, mcnemar_p, paired_delta_ci


def test_bootstrap_ci_degenerate_on_constant():
    lo, hi = bootstrap_ci([1.0] * 20)
    assert lo == hi == 1.0


def test_bootstrap_ci_spreads_on_mixed():
    lo, hi = bootstrap_ci([1.0, 0.0, 1.0, 0.0] * 5)
    assert 0.0 <= lo <= 0.5 and 0.5 <= hi <= 1.0
    assert lo < hi


def test_paired_delta_ci_on_identical_inputs_is_zero():
    point, lo, hi = paired_delta_ci([1, 0, 1, 0], [1, 0, 1, 0])
    assert point == 0.0 and lo <= 0.0 <= hi


def test_paired_delta_ci_detects_improvement():
    a = [0] * 50
    b = [1] * 10 + [0] * 40
    point, lo, hi = paired_delta_ci(a, b)
    assert point == pytest.approx(0.2)
    assert lo > 0.0  # resamples of 0->1 flips never negative


def test_mcnemar_known_values():
    assert mcnemar_p([True] * 5, [False] * 5) == pytest.approx(0.0625)  # 5 flips
    assert mcnemar_p([1, 1, 0, 0], [1, 0, 1, 0]) == 1.0                  # 1 vs 1
    assert mcnemar_p([True, False], [True, False]) == 1.0                 # no disc.
    p = mcnemar_p([0] * 20 + [1] * 80, [1] * 10 + [0] * 90)              # +10 net
    assert 0.0 < p < 0.05


def test_mcnemar_shape_validation():
    with pytest.raises(ValueError):
        mcnemar_p([1], [1, 0])