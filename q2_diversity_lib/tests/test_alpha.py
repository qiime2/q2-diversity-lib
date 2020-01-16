# ----------------------------------------------------------------------------
# Copyright (c) 2018-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin.testing import TestPluginBase
from q2_diversity_lib import (faith_pd, pielou_evenness, observed_features,
                              shannon_entropy)

import io
import biom
import skbio
import numpy as np
import pandas as pd
import pandas.util.testing as pdt

import copy

nonphylogenetic_measures = [observed_features, pielou_evenness,
                            shannon_entropy]


class SmokeTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        self.empty_table = biom.Table(np.array([]), [], [])

    def test_non_phylogenetic_passed_empty_table(self):
        for measure in nonphylogenetic_measures:
            with self.assertRaisesRegex(ValueError, "empty"):
                measure(table=self.empty_table)


class FaithPDTests(TestPluginBase):

    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        self.input_table = biom.Table(np.array([[1, 0, .5, 999, 1],
                                                [0, 1, 2, 0, 1],
                                                [0, 0, 0, 1, 1]]),
                                      ['A', 'B', 'C'],
                                      ['S1', 'S2', 'S3', 'S4', 'S5'])
        self.input_tree = skbio.TreeNode.read(io.StringIO(
                '((A:0.3, B:0.50):0.2, C:100)root;'))
        self.faith_pd_expected = pd.Series({'S1': 0.5, 'S2': 0.7, 'S3': 1.0,
                                            'S4': 100.5, 'S5': 101},
                                           name='faith_pd')

    def test_receives_empty_table(self):
        empty_table = biom.Table(np.array([]), [], [])
        with self.assertRaisesRegex(ValueError, "empty"):
            faith_pd(table=empty_table, phylogeny=self.input_tree)

    def test_method(self):
        actual = faith_pd(table=self.input_table, phylogeny=self.input_tree)
        pdt.assert_series_equal(actual, self.faith_pd_expected)

    def test_accepted_types_have_consistent_behavior(self):
        freq_table = self.input_table
        rel_freq_table = copy.deepcopy(self.input_table).norm(axis='sample',
                                                              inplace=False)
        p_a_table = copy.deepcopy(self.input_table).pa()
        accepted_tables = [freq_table, rel_freq_table, p_a_table]
        for table in accepted_tables:
            actual = faith_pd(table=table, phylogeny=self.input_tree)
            pdt.assert_series_equal(actual, self.faith_pd_expected)

    def test_error_rewriting(self):
        tree = skbio.TreeNode.read(io.StringIO(
            '((A:0.3):0.2, C:100)root;'))
        with self.assertRaisesRegex(skbio.tree.MissingNodeError,
                                    'feature_ids.*phylogeny'):
            faith_pd(table=self.input_table, phylogeny=tree)


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
        self.observed_features_expected = pd.Series(
                {'S1': 1, 'S2': 1, 'S3': 2, 'S4': 2,
                 'S5': 3},
                name='observed_features')

    def test_method(self):
        actual = observed_features(table=self.input_table)
        pdt.assert_series_equal(actual, self.observed_features_expected)

    def test_accepted_types_have_consistent_behavior(self):
        freq_table = self.input_table
        rel_freq_table = copy.deepcopy(self.input_table).norm(axis='sample',
                                                              inplace=False)
        p_a_table = copy.deepcopy(self.input_table).pa()
        accepted_tables = [freq_table, rel_freq_table, p_a_table]
        for table in accepted_tables:
            actual = observed_features(table)
            pdt.assert_series_equal(actual, self.observed_features_expected)


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
        self.pielou_evenness_expected = pd.Series(
                {'S1': np.NaN, 'S2': np.NaN, 'S3': 1, 'S4': 1,
                 'S5': 1, 'S6': 0.946394630357186},
                name='pielou_evenness')

    def test_method(self):
        actual = pielou_evenness(table=self.input_table)
        pdt.assert_series_equal(actual, self.pielou_evenness_expected)

    def test_accepted_types_have_consistent_behavior(self):
        freq_table = self.input_table
        rel_freq_table = copy.deepcopy(self.input_table).norm(axis='sample',
                                                              inplace=False)
        accepted_tables = [freq_table, rel_freq_table]
        for table in accepted_tables:
            actual = pielou_evenness(table)
            pdt.assert_series_equal(actual, self.pielou_evenness_expected)

    def test_drop_undefined_samples(self):
        NaN_table = biom.Table(np.array([[0, 1, 0, 0, 1, 1],
                                         [0, 0, 1, 0, 1, 1],
                                         [0, 0, 0, 1, 0, 1]]),
                               ['A', 'B', 'C'],
                               ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])
        expected = pd.Series({'S5': 1, 'S6': 1}, name='pielou_evenness')
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
        self.shannon_entropy_expected = pd.Series(
                {'S1': np.NaN, 'S2': 0, 'S3': 1, 'S4': np.NaN,
                 'S5': 1.584962500721156, 'S6': 2},
                name='shannon_entropy')

    def test_method(self):
        actual = shannon_entropy(table=self.input_table)
        pdt.assert_series_equal(actual, self.shannon_entropy_expected)

    def test_accepted_types_have_consistent_behavior(self):
        freq_table = self.input_table
        rel_freq_table = copy.deepcopy(self.input_table).norm(axis='sample',
                                                              inplace=False)
        accepted_tables = [freq_table, rel_freq_table]
        for table in accepted_tables:
            actual = shannon_entropy(table)
            pdt.assert_series_equal(actual, self.shannon_entropy_expected)

    def test_drop_undefined_samples(self):
        expected = pd.Series({'S2': 0, 'S3': 1, 'S5': 1.584962500721156,
                              'S6': 2}, name='shannon_entropy')
        actual = shannon_entropy(table=self.input_table,
                                 drop_undefined_samples=True)
        pdt.assert_series_equal(actual, expected, check_dtype=False)

    def test_do_not_drop_undefined_samples(self):
        actual = shannon_entropy(table=self.input_table,
                                 drop_undefined_samples=False)
        pdt.assert_series_equal(actual, self.shannon_entropy_expected)
