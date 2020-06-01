# ----------------------------------------------------------------------------
# Copyright (c) 2018-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import io

import numpy as np
import numpy.testing as npt
import biom
import skbio

from qiime2.plugin.testing import TestPluginBase
from q2_types.feature_table import BIOMV210Format
from q2_diversity_lib import (
        bray_curtis, jaccard, unweighted_unifrac, weighted_unifrac)

from qiime2 import Artifact

nonphylogenetic_measures = [bray_curtis, jaccard]
phylogenetic_measures = [unweighted_unifrac, weighted_unifrac]


class SmokeTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        self.valid_table_fp = self.get_data_path('two_feature_table.biom')
        self.valid_table = biom.load_table(self.valid_table_fp)
        self.valid_table_as_BIOMV210Format = \
            BIOMV210Format(self.valid_table_fp, mode='r')
        # empty table fp generated from self.empty_table with biom v2.1.7
        self.empty_table = biom.Table(np.array([]), [], [])
        self.empty_table_fp = self.get_data_path('empty_table.biom')
        self.empty_table_as_BIOMV210Format = \
            BIOMV210Format(self.empty_table_fp, mode='r')

        self.empty_tree_fp = self.get_data_path('empty.tree')
        self.root_only_tree_fp = self.get_data_path('root_only.tree')
        self.missing_tip_tree_fp = self.get_data_path('missing_tip.tree')
        self.two_feature_tree_fp = self.get_data_path('two_feature.tree')
        self.extra_tip_tree_fp = self.get_data_path('extra_tip.tree')
        self.input_tree = skbio.TreeNode.read(io.StringIO(
                '((A:0.3, B:0.50):0.2, C:100)root;'))
        self.valid_tree_fp = self.get_data_path('three_feature.tree')

    def test_nonphylogenetic_measures_passed_empty_table(self):
        for measure in nonphylogenetic_measures:
            with self.assertRaisesRegex(ValueError, "empty"):
                measure(table=self.empty_table)

    def test_phylogenetic_measures_passed_empty_table(self):
        for measure in phylogenetic_measures:
            with self.assertRaisesRegex(ValueError, "empty"):
                measure(table=self.empty_table_as_BIOMV210Format,
                        phylogeny=self.valid_tree_fp)

    def test_phylogenetic_measures_passed_emptytree_fp(self):
        # HACK: different regular expressions are used here for unweighted and
        # all other unifracs, because tree/table validation is not being
        # applied to the other unifracs. Once unifrac PR #106 is merged, this
        # change will need to be reverted.
        for measure in phylogenetic_measures:
            if (measure.__name__ == 'unweighted_unifrac'):
                with self.assertRaisesRegex(ValueError, "newick"):
                    measure(table=self.valid_table_as_BIOMV210Format,
                            phylogeny=self.empty_tree_fp)
            else:
                with self.assertRaisesRegex(
                        ValueError, 'table.*not.*completely represented'):
                    measure(table=self.valid_table_as_BIOMV210Format,
                            phylogeny=self.empty_tree_fp)

    def test_phylogenetic_measures_passed_rootonlytree_fp(self):
        for measure in phylogenetic_measures:
            with self.assertRaisesRegex(ValueError,
                                        "table.*not.*completely represented"):
                measure(table=self.valid_table_as_BIOMV210Format,
                        phylogeny=self.root_only_tree_fp)

    def test_phylogenetic_measures_passed_tree_missing_tip_fp(self):
        for measure in phylogenetic_measures:
            with self.assertRaisesRegex(ValueError,
                                        "table.*not.*completely represented"):
                measure(table=self.valid_table_as_BIOMV210Format,
                        phylogeny=self.missing_tip_tree_fp)

    def test_phylogenetic_measure_passed_tree_w_extra_tip_fp(self):
        for measure in phylogenetic_measures:
            matched_tree_output = measure(self.valid_table_as_BIOMV210Format,
                                          self.two_feature_tree_fp)
            extra_tip_tree_output = measure(self.valid_table_as_BIOMV210Format,
                                            self.valid_tree_fp)
            self.assertEqual(matched_tree_output, extra_tip_tree_output)


# ----------------------------Non-Phylogenetic---------------------------------
class BrayCurtisTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        self.input_table = biom.Table(np.array([[1, 0, 999, 1],
                                                [0, .5, 0, 1],
                                                [0, 0, 1, 1]]),
                                      ['A', 'B', 'C'],
                                      ['S1', 'S2', 'S3', 'S4'])

        self.expected = skbio.DistanceMatrix(
                [[0.0000000, 1, 0.998001998, 0.5],
                 [1, 0.0000000, 1, .714285714],
                 [0.998001998, 1, 0.0000000, .996011964],
                 [0.5, .714285714, .996011964, 0.0000000]],
                ids=['S1', 'S2', 'S3', 'S4'])

    def test_method(self):
        actual = bray_curtis(table=self.input_table, n_jobs=1)
        self.assertEqual(actual.ids, self.expected.ids)
        for id1 in actual.ids:
            for id2 in actual.ids:
                npt.assert_almost_equal(actual[id1, id2],
                                        self.expected[id1, id2])

    def test_accepted_types_have_consistent_behavior_rarefied(self):
        rarefied_table = biom.Table(np.array([[3, 1, 2, 6],
                                             [1, 3, 2, 0],
                                             [2, 2, 2, 0]]),
                                    ['A', 'B', 'C'],
                                    ['S1', 'S2', 'S3', 'S4'])

        normalized_rarefied_table = rarefied_table.norm(axis='sample',
                                                        inplace=False)
        self.expected = skbio.DistanceMatrix(
                [[0.0000000, 0.333333333333333, 0.1666666667, 0.5],
                 [0.333333333333333, 0.0000000, 0.1666666667, 0.833333333333],
                 [0.1666666667, 0.1666666667, 0.0000000, 0.66666666667],
                 [0.5, 0.833333333333, 0.66666666667, 0.0000000]],
                ids=['S1', 'S2', 'S3', 'S4'])

        accepted_tables = [rarefied_table, normalized_rarefied_table]
        for table in accepted_tables:
            actual = bray_curtis(table=table)
            self.assertEqual(actual.ids, self.expected.ids)
            for id1 in actual.ids:
                for id2 in actual.ids:
                    npt.assert_almost_equal(actual[id1, id2],
                                            self.expected[id1, id2])

    def test_accepted_types_have_inconsistent_behavior_unrarefied(self):
        freq_table = self.input_table
        relative_freq_table = self.input_table.norm(axis='sample',
                                                    inplace=False)
        # count data should pass, as in test_method
        actual = bray_curtis(table=freq_table)
        for id1 in actual.ids:
            for id2 in actual.ids:
                npt.assert_almost_equal(actual[id1, id2],
                                        self.expected[id1, id2])

        # normalized data should fail
        actual = bray_curtis(table=relative_freq_table)
        with self.assertRaisesRegex(AssertionError,
                                    "not almost equal"):
            for id1 in actual.ids:
                for id2 in actual.ids:
                    npt.assert_almost_equal(actual[id1, id2],
                                            self.expected[id1, id2])


class JaccardTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        self.freq_table = biom.Table(np.array([[1, 0, 999, 1],
                                               [0, .5, 0, 1],
                                               [0, 0, 1, 1]]),
                                     ['A', 'B', 'C'],
                                     ['S1', 'S2', 'S3', 'S4'])
        self.expected = skbio.DistanceMatrix(
                [[0.0000000, 1, 0.5, 0.6666667],
                 [1, 0.0000000, 1, 0.6666667],
                 [0.5, 1, 0.0000000, 0.3333333],
                 [0.6666667, 0.6666667, 0.3333333, 0.0000000]],
                ids=['S1', 'S2', 'S3', 'S4'])

        self.relative_freq_table = self.freq_table.norm(axis='sample',
                                                        inplace=False)
        self.p_a_table = self.freq_table.pa(inplace=False)

    def test_method(self):
        actual = jaccard(table=self.freq_table, n_jobs=1)
        self.assertEqual(actual.ids, self.expected.ids)
        for id1 in actual.ids:
            for id2 in actual.ids:
                npt.assert_almost_equal(actual[id1, id2],
                                        self.expected[id1, id2])

    def test_accepted_types_have_consistent_behavior(self):
        accepted_tables = [self.freq_table, self.relative_freq_table,
                           self.p_a_table]
        for table in accepted_tables:
            actual = jaccard(table=table)
            self.assertEqual(actual.ids, self.expected.ids)
            for id1 in actual.ids:
                for id2 in actual.ids:
                    npt.assert_almost_equal(actual[id1, id2],
                                            self.expected[id1, id2])


# ---------------------------------Phylogenetic-------------------------------
class UnweightedUnifrac(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        # expected computed with skbio.diversity.beta_diversity
        self.expected = skbio.DistanceMatrix([[0.00, 0.25, 0.25],
                                             [0.25, 0.00, 0.00],
                                             [0.25, 0.00, 0.00]],
                                             ids=['S1', 'S2', 'S3'])

        self.table_fp = self.get_data_path('two_feature_table.biom')
        self.table_as_BIOMV210Format = BIOMV210Format(self.table_fp, mode='r')
        self.rf_table_fp = self.get_data_path('two_feature_rf_table.biom')
        self.rf_table_as_BIOMV210Format = BIOMV210Format(self.rf_table_fp,
                                                         mode='r')
        self.p_a_table_fp = self.get_data_path('two_feature_p_a_table.biom')
        self.p_a_table_as_BIOMV210Format = BIOMV210Format(self.p_a_table_fp,
                                                          mode='r')

        self.tree_fp = self.get_data_path('three_feature.tree')

    def test_method(self):
        actual = unweighted_unifrac(self.table_as_BIOMV210Format, self.tree_fp)
        self.assertEqual(actual.ids, self.expected.ids)
        for id1 in actual.ids:
            for id2 in actual.ids:
                npt.assert_almost_equal(actual[id1, id2],
                                        self.expected[id1, id2])

    def test_accepted_types_have_consistent_behavior(self):
        freq_table = self.table_as_BIOMV210Format
        rel_freq_table = self.rf_table_as_BIOMV210Format
        p_a_table = self.p_a_table_as_BIOMV210Format
        accepted_tables = [freq_table, rel_freq_table, p_a_table]
        for table in accepted_tables:
            actual = unweighted_unifrac(table=table, phylogeny=self.tree_fp)
            self.assertEqual(actual.ids, self.expected.ids)
            for id1 in actual.ids:
                for id2 in actual.ids:
                    npt.assert_almost_equal(actual[id1, id2],
                                            self.expected[id1, id2])

    def test_does_it_run_through_framework(self):
        unweighted_unifrac_thru_framework = self.plugin.actions[
                    'unweighted_unifrac']
        table_as_artifact = Artifact.import_data(
                    'FeatureTable[Frequency]', self.table_fp)
        tree_as_artifact = Artifact.import_data(
                    'Phylogeny[Rooted]', self.tree_fp)
        unweighted_unifrac_thru_framework(table_as_artifact,
                                          tree_as_artifact)
        # If we get here, then it ran without error
        self.assertTrue(True)


class WeightedUnifrac(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        # expected computed with diversity.beta_phylogenetic (weighted_unifrac)
        self.expected = skbio.DistanceMatrix(
            np.array([0.44656238, 0.23771096, 0.30489123, 0.23446002,
                      0.65723575, 0.44911772, 0.381904, 0.69144829,
                      0.39611776, 0.36568012, 0.53377975, 0.48908025,
                      0.35155196, 0.28318669, 0.57376916, 0.23395746,
                      0.24658122, 0.60271637, 0.39802552, 0.36567394,
                      0.68062701, 0.36862049, 0.48350632, 0.33024631,
                      0.33266697, 0.53464744, 0.74605075, 0.53951035,
                      0.49680733, 0.79178838, 0.37109012, 0.52629343,
                      0.22118218, 0.32400805, 0.43189708, 0.59705893]),
            ids=('10084.PC.481', '10084.PC.593', '10084.PC.356',
                 '10084.PC.355', '10084.PC.354', '10084.PC.636',
                 '10084.PC.635', '10084.PC.607', '10084.PC.634'))

        self.table_fp = self.get_data_path('crawford.biom')
        self.table_as_BIOMV210Format = BIOMV210Format(self.table_fp, mode='r')
        self.rel_freq_table_fp = self.get_data_path('crawford_rf.biom')
        self.rf_table_as_BIOMV210Format = \
            BIOMV210Format(self.rel_freq_table_fp, mode='r')

        self.tree_fp = self.get_data_path('crawford.nwk')

    def test_method(self):
        actual = weighted_unifrac(self.table_as_BIOMV210Format, self.tree_fp)
        self.assertEqual(actual.ids, self.expected.ids)
        for id1 in actual.ids:
            for id2 in actual.ids:
                npt.assert_almost_equal(actual[id1, id2],
                                        self.expected[id1, id2])

    def test_accepted_types_have_consistent_behavior(self):
        freq_table = self.table_as_BIOMV210Format
        rel_freq_table = self.rf_table_as_BIOMV210Format
        accepted_tables = [freq_table, rel_freq_table]
        for table in accepted_tables:
            actual = weighted_unifrac(table=table, phylogeny=self.tree_fp)
            self.assertEqual(actual.ids, self.expected.ids)
            for id1 in actual.ids:
                for id2 in actual.ids:
                    npt.assert_almost_equal(actual[id1, id2],
                                            self.expected[id1, id2])
