# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from subprocess import CalledProcessError

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import biom
import os

from qiime2.plugin.testing import TestPluginBase
from qiime2 import Artifact

from ..alpha import (pielou_evenness, observed_features,
                     shannon_entropy, METRICS,
                     _berger_parker, _brillouin_d,
                     _simpsons_dominance, _esty_ci,
                     _goods_coverage, _margalef,
                     _mcintosh_d, _strong
                     )


class SmokeTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        empty_table = biom.Table(np.array([]), [], [])
        self.empty_table = Artifact.import_data('FeatureTable[Frequency]',
                                                empty_table)

    def test_non_phylogenetic_passed_empty_table(self):
        for metric in METRICS['NONPHYLO']['IMPL']:
            metric = METRICS['NAME_TRANSLATIONS'][metric]
            with self.assertRaisesRegex(ValueError, 'empty'):
                self.plugin.actions[metric](table=self.empty_table)


class FaithPDTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def artifact(self, semantic_type, fp):
        return Artifact.import_data(semantic_type, self.get_data_path(fp))

    @staticmethod
    def assert_vec_equal(actual_art, expected):
        actual = actual_art.view(pd.Series)
        pdt.assert_series_equal(actual, expected)

    def setUp(self):
        super().setUp()
        self.fn = self.plugin.actions['faith_pd']
        self.tbl = self.artifact('FeatureTable[Frequency]',
                                 'faith_test_table.biom')
        self.tre = self.artifact('Phylogeny[Rooted]', 'faith_test.tree')
        self.expected = pd.Series({'S1': 0.5, 'S2': 0.7, 'S3': 1.0,
                                   'S4': 100.5, 'S5': 101},
                                  name='faith_pd')

    def test_receives_empty_table(self):
        table = self.artifact('FeatureTable[Frequency]', 'empty_table.biom')
        with self.assertRaisesRegex(ValueError, 'empty'):
            self.fn(table=table, phylogeny=self.tre)

    def test_method(self):
        actual_art, = self.fn(table=self.tbl, phylogeny=self.tre)
        self.assert_vec_equal(actual_art, self.expected)

    def test_accepted_types_have_consistent_behavior(self):
        rf_tbl = self.artifact('FeatureTable[RelativeFrequency]',
                               'faith_test_table_rf.biom')
        pa_tbl = self.artifact('FeatureTable[PresenceAbsence]',
                               'faith_test_table_pa.biom')
        for table in [self.tbl, rf_tbl, pa_tbl]:
            actual_art, = self.fn(table=table, phylogeny=self.tre)
            self.assert_vec_equal(actual_art, self.expected)

    def test_passed_tree_missing_tip(self):
        tree = self.artifact('Phylogeny[Rooted]', 'missing_tip.tree')
        with self.assertRaises(CalledProcessError):
            obs = self.fn(table=self.tbl, phylogeny=tree)
            self.assertTrue('not a subset of the tree tips' in obs.stderr)

    def test_passed_rootonlytree(self):
        tree = self.artifact('Phylogeny[Rooted]', 'root_only.tree')
        with self.assertRaises(CalledProcessError):
            obs = self.fn(table=self.tbl, phylogeny=tree)
            self.assertTrue('not a subset of the tree tips' in obs.stderr)

    def test_no_index_name(self):
        res, = self.fn(table=self.tbl, phylogeny=self.tre)
        res_dirfmt = res.view(res.format)
        faithpd_fp = os.path.join(str(res_dirfmt), 'alpha-diversity.tsv')
        with open(faithpd_fp, 'r') as fh:
            data = fh.read()
            self.assertNotIn('#SampleID', data)


class ObservedFeaturesTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        self.input_table = biom.Table(np.array([[1, 0, .5, 999, 1],
                                                [0, 1, 2, 0, 5],
                                                [0, 0, 0, 1, 10]]),
                                      ['A', 'B', 'C'],
                                      ['S1', 'S2', 'S3', 'S4', 'S5'])
        # Calculated by hand:
        self.expected = pd.Series(
                {'S1': 1, 'S2': 1, 'S3': 2, 'S4': 2,
                 'S5': 3},
                name='observed_features')

    def test_method(self):
        actual = observed_features(table=self.input_table)
        pdt.assert_series_equal(actual, self.expected)

    def test_accepted_types_have_consistent_behavior(self):
        freq_table = self.input_table
        rel_freq_table = self.input_table.norm(axis='sample',
                                               inplace=False)
        p_a_table = self.input_table.pa(inplace=False)
        accepted_tables = [freq_table, rel_freq_table, p_a_table]
        for table in accepted_tables:
            actual = observed_features(table)
            pdt.assert_series_equal(actual, self.expected)


class PielouEvennessTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        self.input_table = biom.Table(np.array([[0, 1, 1, 1, 999, 1],
                                                [0, 0, 1, 1, 999, 1],
                                                [0, 0, 0, 1, 999, 2]]),
                                      ['A', 'B', 'C'],
                                      ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])
        # Calculated by hand:
        self.expected = pd.Series(
                {'S1': np.NaN, 'S2': np.NaN, 'S3': 1, 'S4': 1,
                 'S5': 1, 'S6': 0.946394630357186},
                name='pielou_evenness')

    def test_method(self):
        actual = pielou_evenness(table=self.input_table)
        pdt.assert_series_equal(actual, self.expected)

    def test_accepted_types_have_consistent_behavior(self):
        freq_table = self.input_table
        rel_freq_table = self.input_table.norm(axis='sample',
                                               inplace=False)
        accepted_tables = [freq_table, rel_freq_table]
        for table in accepted_tables:
            actual = pielou_evenness(table)
            pdt.assert_series_equal(actual, self.expected)

    def test_drop_undefined_samples(self):
        NaN_table = biom.Table(np.array([[0, 1, 0, 0, 1, 1],
                                         [0, 0, 1, 0, 1, 1],
                                         [0, 0, 0, 1, 0, 1]]),
                               ['A', 'B', 'C'],
                               ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])
        # pandas supports floating point correction for float dtype only,
        # these 1 ints were changed to 1.0 floats to get that support
        expected = pd.Series({'S5': 1.0, 'S6': 1.0}, name='pielou_evenness')
        actual = pielou_evenness(table=NaN_table, drop_undefined_samples=True)
        pdt.assert_series_equal(actual, expected, check_dtype=False)

    def test_do_not_drop_undefined_samples(self):
        NaN_table = biom.Table(np.array([[0, 1, 0, 0, 1, 1],
                                         [0, 0, 1, 0, 1, 1],
                                         [0, 0, 0, 1, 0, 1]]),
                               ['A', 'B', 'C'],
                               ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])
        expected = pd.Series({'S1': np.NaN, 'S2': np.NaN, 'S3': np.NaN,
                              'S4': np.NaN, 'S5': 1, 'S6': 1},
                             name='pielou_evenness')
        actual = pielou_evenness(table=NaN_table, drop_undefined_samples=False)
        pdt.assert_series_equal(actual, expected)


class ShannonEntropyTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        self.input_table = biom.Table(np.array([[0, 0, 0, 0, 1, 1],
                                                [0, 1, 0, 0, 1, 1],
                                                [0, 0, 1, 0, 1, 1],
                                                [0, 0, 1, 0, 0, 1]]),
                                      ['A', 'B', 'C', 'D'],
                                      ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])
        # Calculated by hand:
        self.expected = pd.Series(
                {'S1': np.NaN, 'S2': 0, 'S3': 1, 'S4': np.NaN,
                 'S5': 1.584962500721156, 'S6': 2},
                name='shannon_entropy')

    def test_method(self):
        actual = shannon_entropy(table=self.input_table)
        pdt.assert_series_equal(actual, self.expected)

    def test_accepted_types_have_consistent_behavior(self):
        freq_table = self.input_table
        rel_freq_table = self.input_table.norm(axis='sample',
                                               inplace=False)
        accepted_tables = [freq_table, rel_freq_table]
        for table in accepted_tables:
            actual = shannon_entropy(table)
            pdt.assert_series_equal(actual, self.expected)

    def test_drop_undefined_samples(self):
        expected = pd.Series({'S2': 0, 'S3': 1, 'S5': 1.584962500721156,
                              'S6': 2}, name='shannon_entropy')
        actual = shannon_entropy(table=self.input_table,
                                 drop_undefined_samples=True)
        pdt.assert_series_equal(actual, expected, check_dtype=False)

    def test_do_not_drop_undefined_samples(self):
        actual = shannon_entropy(table=self.input_table,
                                 drop_undefined_samples=False)
        pdt.assert_series_equal(actual, self.expected)


class AlphaPassthroughTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        self.method = self.plugin.actions['alpha_passthrough']
        empty_table = biom.Table(np.array([]), [], [])
        self.empty_table = Artifact.import_data('FeatureTable[Frequency]',
                                                empty_table)
        crawford_tbl = self.get_data_path('crawford.biom')
        self.crawford_tbl = Artifact.import_data('FeatureTable[Frequency]',
                                                 crawford_tbl)

    def test_method(self):
        for metric in METRICS['NONPHYLO']['UNIMPL']:
            # NOTE: crawford table used b/c input_table too minimal for `ace`
            self.method(table=self.crawford_tbl, metric=metric)
        # If we get here, then our methods ran without error
        self.assertTrue(True)

    def test_passed_empty_table(self):
        for metric in METRICS['NONPHYLO']['UNIMPL']:
            with self.assertRaisesRegex(ValueError, 'empty'):
                self.method(table=self.empty_table, metric=metric)

    def test_passed_bad_metric(self):
        with self.assertRaisesRegex(TypeError,
                                    'imaginary_metric.*incompatible'):
            self.method(table=self.crawford_tbl, metric='imaginary_metric')

    def test_passed_implemented_metric(self):
        # alpha_passthrough does not provide access to measures that have been
        # implemented locally
        for metric in METRICS['NONPHYLO']['IMPL']:
            with self.assertRaisesRegex(TypeError, f"{metric}.*incompatible"):
                self.method(table=self.crawford_tbl, metric=metric)

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
