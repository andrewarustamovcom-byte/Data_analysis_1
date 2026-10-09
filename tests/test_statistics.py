"""Проверки формул и кластеризации на маленьких контролируемых данных."""

import unittest

import numpy as np
import pandas as pd
from scipy import stats

from src.config import Config
from src.diagnostics import break_even_costs
from src.statistics import binary_effect, bootstrap_values, discrete_ks_permutation, mean_effect, ratio_effect


class StatisticalTests(unittest.TestCase):
    def test_welch_matches_scipy_with_unequal_variance(self):
        c = np.array([0., 0., 3., 5., 25.])
        t = np.array([0., 0., 0., 100., 250., 900.])
        result = mean_effect(c, t)
        reference = stats.ttest_ind(t, c, equal_var=False)
        self.assertAlmostEqual(result["p_value"], reference.pvalue)
        self.assertAlmostEqual(result["ci_low"], reference.confidence_interval().low)
        self.assertAlmostEqual(result["effect"], t.mean()-c.mean())

    def test_ratio_uncertainty_accounts_for_repeat_events(self):
        # У каждого пользователя 100 одинаковых исходов: события внутри него зависимы.
        c_n, c_d = np.array([0, 0, 100, 100]), np.full(4, 100)
        t_n, t_d = np.array([0, 100, 100, 100]), np.full(4, 100)
        result = ratio_effect(c_n, c_d, t_n, t_d)
        naive_se = np.sqrt(.5*.5/400 + .75*.25/400)
        self.assertGreater(result["se"], naive_se*8)
        self.assertAlmostEqual(result["effect"], .25)

    def test_no_orders_does_not_turn_undefined_ratio_into_zero(self):
        result = ratio_effect([0, 0], [0, 0], [0, 0], [0, 0])
        self.assertTrue(np.isnan(result["control"]))
        self.assertTrue(np.isnan(result["p_value"]))

    def test_zero_events_still_have_nonzero_uncertainty(self):
        result = binary_effect(np.zeros(100), np.zeros(100))
        self.assertEqual(result["p_value"], 1)
        self.assertLess(result["ci_low"], 0)
        self.assertGreater(result["ci_high"], 0)

    def test_fractional_binary_input_is_rejected(self):
        with self.assertRaises(ValueError):
            binary_effect([0, .5, 1], [0, 1, 1])

    def test_bootstrap_is_reproducible_and_resamples_users(self):
        values = np.array([[0, 10], [0, 10], [10, 10], [10, 10]])
        a = bootstrap_values(values, 500, np.random.default_rng(7), ratio=True)
        b = bootstrap_values(values, 500, np.random.default_rng(7), ratio=True)
        np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(a.mean(), .5, delta=.04)
        self.assertGreater(a.std(), .2)

    def test_ks_handles_tied_identical_distributions(self):
        result = discrete_ks_permutation([0, 0, 1, 1], [0, 0, 1, 1], 99, np.random.default_rng(7))
        self.assertEqual(result["statistic"], 0)
        self.assertEqual(result["p_value"], 1)

    def test_break_even_price_zeroes_additional_margin(self):
        mart = pd.DataFrame({"group": ["control"]*3 + ["test"]*3,
                             "margin": [10., 20., 30., 40., 50., 60.],
                             "tickets_observed": [0]*6, "impressions": [100]*6})
        row = break_even_costs(mart, Config()).iloc[0]
        self.assertAlmostEqual(row.ai_break_even_rub_per_impression, .3)
        adjusted = mart.margin-row.ai_break_even_rub_per_impression*mart.impressions*mart.group.eq("test")
        effect = mean_effect(adjusted[mart.group.eq("control")], adjusted[mart.group.eq("test")])
        self.assertAlmostEqual(effect["effect"], 0)
        self.assertLess(row.ci_low, .3)
        self.assertGreater(row.ci_high, .3)


if __name__ == "__main__":
    unittest.main()
