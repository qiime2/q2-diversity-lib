# ----------------------------------------------------------------------------
# Copyright (c) 2018-2022, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from subprocess import CalledProcessError

import numpy as np
import pandas as pd
import pandas.testing as pdt
import biom

from qiime2.plugin.testing import TestPluginBase
from qiime2 import Artifact

from ..alpha import (pielou_evenness, observed_features,
                     shannon_entropy, METRICS)


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
