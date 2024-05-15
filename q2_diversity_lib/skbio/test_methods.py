import numpy as np
import numpy.testing as npt

from qiime2.plugin.testing import TestPluginBase

from q2_diversity_lib.skbio._methods import (_berger_parker, _brillouin_d,
                                             _simpsons_dominance, _esty_ci,
                                             _goods_coverage, _margalef,
                                             _mcintosh_d, _strong)


class SkbioTests(TestPluginBase):
    package = 'q2_diversity_lib.skbio'

# tests for passthrough metrics were sourced from skbio
    def test_berger_parker_d(self):
        self.assertEqual(_berger_parker(np.array([5, 5])), 0.5)
        self.assertEqual(_berger_parker(np.array([1, 1, 1, 1, 0])), 0.25)

    def test_brillouin_d(self):
        self.assertAlmostEqual(_brillouin_d(np.array([1, 2, 0, 0, 3, 1])),
                               0.86289353018248782)

    def test_esty_ci(self):
        def _diversity(indices, f):
            """Calculate diversity index for each window of size 1.
            indices: vector of indices of taxa
            f: f(counts) -> diversity measure
            """
            result = []
            max_size = max(indices) + 1
            freqs = np.zeros(max_size, dtype=int)
            for i in range(len(indices)):
                freqs += np.bincount(indices[i:i + 1], minlength=max_size)
                try:
                    curr = f(freqs)
                except (ZeroDivisionError, FloatingPointError):
                    curr = 0
                result.append(curr)
            return np.array(result)

        data = [1, 1, 2, 1, 1, 3, 2, 1, 3, 4]

        observed_lower, observed_upper = zip(*_diversity(data, _esty_ci))

        expected_lower = np.array([1, -1.38590382, -0.73353593, -0.17434465,
                                   -0.15060902, -0.04386191, -0.33042054,
                                   -0.29041008, -0.43554755, -0.33385652])
        expected_upper = np.array([1, 1.38590382, 1.40020259, 0.67434465,
                                   0.55060902, 0.71052858, 0.61613483,
                                   0.54041008, 0.43554755, 0.53385652])

        npt.assert_array_almost_equal(observed_lower, expected_lower)
        npt.assert_array_almost_equal(observed_upper, expected_upper)

    def test_simpson(self):
        self.assertAlmostEqual(_simpsons_dominance(np.array([1, 0, 2, 5, 2])),
                               0.66)
        self.assertAlmostEqual(_simpsons_dominance(np.array([5])), 0)

    def test_goods_coverage(self):
        counts = [1] * 75 + [2, 2, 2, 2, 2, 2, 3, 4, 4]
        obs = _goods_coverage(counts)
        self.assertAlmostEqual(obs, 0.23469387755)

    def test_margalef(self):

        self.assertEqual(_margalef(np.array([0, 1, 1, 4, 2, 5, 2, 4, 1, 2])),
                         8 / np.log(22))

    def test_mcintosh_d(self):
        self.assertAlmostEqual(_mcintosh_d(np.array([1, 2, 3])),
                               0.636061424871458)

    def test_strong(self):
        self.assertAlmostEqual(_strong(np.array([1, 2, 3, 1])), 0.214285714)
