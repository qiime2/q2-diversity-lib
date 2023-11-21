# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import os
from subprocess import CalledProcessError

import numpy as np
import numpy.testing as npt
import biom
import skbio
import pkg_resources

from qiime2.plugin.testing import TestPluginBase
from qiime2 import Artifact

from ..beta import bray_curtis, jaccard, METRICS


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

    def test_bray_curtis_relative_frequency(self):
        input_table = biom.Table(
            np.array([
                [0.3, 0, 0.77, 0.5],
                [0.1, 0, 0.15, 0.25],
                [0.6, 1, 0.08, 0.25]
            ]),
            ['A', 'B', 'C'],
            ['S1', 'S2', 'S3', 'S4']
        )
        expected = skbio.DistanceMatrix(
            [
                [0.0000000, 0.4, 0.52, 0.35],
                [0.4, 0.0000000, 0.92, 0.75],
                [0.52, 0.92, 0.0000000, 0.27],
                [0.35, 0.75, 0.27, 0.0000000]
            ],
            ids=['S1', 'S2', 'S3', 'S4']
        )
        actual = bray_curtis(table=input_table, n_jobs=1)
        self.assertEqual(actual.ids, self.expected.ids)
        for id1 in actual.ids:
            for id2 in actual.ids:
                npt.assert_almost_equal(
                    actual[id1, id2], expected[id1, id2]
                )


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

    def artifact(self, semantic_type, fp):
        return Artifact.import_data(semantic_type, self.get_data_path(fp))

    def setUp(self):
        super().setUp()

        self.fn = self.plugin.actions['unweighted_unifrac']

        # expected computed with skbio.diversity.beta_diversity
        self.expected = skbio.DistanceMatrix([[0.00, 0.25, 0.25],
                                             [0.25, 0.00, 0.00],
                                             [0.25, 0.00, 0.00]],
                                             ids=['S1', 'S2', 'S3'])

        self.tbl = self.artifact('FeatureTable[Frequency]',
                                 'two_feature_table.biom')
        self.tre = self.artifact('Phylogeny[Rooted]', 'three_feature.tree')

    def test_method(self):
        actual_art, = self.fn(self.tbl, self.tre)
        actual = actual_art.view(skbio.DistanceMatrix)
        self.assertEqual(actual.ids, self.expected.ids)
        for id1 in actual.ids:
            for id2 in actual.ids:
                npt.assert_almost_equal(actual[id1, id2],
                                        self.expected[id1, id2])

    def test_method_bypass_tips(self):
        actual_art, = self.fn(self.tbl, self.tre, bypass_tips=True)
        actual = actual_art.view(skbio.DistanceMatrix)
        self.assertEqual(actual.ids, self.expected.ids)

    def test_accepted_types_have_consistent_behavior(self):
        rf_tbl = self.artifact('FeatureTable[RelativeFrequency]',
                               'two_feature_rf_table.biom')
        pa_tbl = self.artifact('FeatureTable[PresenceAbsence]',
                               'two_feature_p_a_table.biom')

        for table in [self.tbl, rf_tbl, pa_tbl]:
            actual_art, = self.fn(table=table, phylogeny=self.tre)
            actual = actual_art.view(skbio.DistanceMatrix)
            self.assertEqual(actual.ids, self.expected.ids)
            for id1 in actual.ids:
                for id2 in actual.ids:
                    npt.assert_almost_equal(actual[id1, id2],
                                            self.expected[id1, id2])

    def test_missing_tips_tree(self):
        tre = self.artifact('Phylogeny[Rooted]', 'root_only.tree')
        with self.assertRaises(CalledProcessError):
            obs = self.fn(self.tbl, tre)
            self.assertTrue('not a subset of the tree tips', obs.stderr)

    def test_extra_tip_tree(self):
        tbl = self.artifact('FeatureTable[Frequency]',
                            'two_feature_table.biom')

        tre_2 = self.artifact('Phylogeny[Rooted]', 'two_feature.tree')
        obs_2_art, = self.fn(tbl, tre_2)
        obs_2 = obs_2_art.view(skbio.DistanceMatrix)

        tre_3 = self.artifact('Phylogeny[Rooted]', 'three_feature.tree')
        obs_3_art, = self.fn(tbl, tre_3)
        obs_3 = obs_3_art.view(skbio.DistanceMatrix)

        self.assertEqual(obs_2, obs_3)


class WeightedUnifrac(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def artifact(self, semantic_type, fp):
        return Artifact.import_data(semantic_type, self.get_data_path(fp))

    def setUp(self):
        super().setUp()

        self.fn = self.plugin.actions['weighted_unifrac']

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

        self.tbl = self.artifact('FeatureTable[Frequency]', 'crawford.biom')
        self.tre = self.artifact('Phylogeny[Rooted]', 'crawford.nwk')

    def test_method(self):
        actual_art, = self.fn(self.tbl, self.tre)
        actual = actual_art.view(skbio.DistanceMatrix)
        self.assertEqual(actual.ids, self.expected.ids)
        for id1 in actual.ids:
            for id2 in actual.ids:
                npt.assert_almost_equal(actual[id1, id2],
                                        self.expected[id1, id2])

    def test_method_bypass_tips(self):
        actual_art, = self.fn(self.tbl, self.tre, bypass_tips=True)
        actual = actual_art.view(skbio.DistanceMatrix)
        self.assertEqual(actual.ids, self.expected.ids)

    def test_accepted_types_have_consistent_behavior(self):
        tbl_rf = self.artifact('FeatureTable[RelativeFrequency]',
                               'crawford_rf.biom')

        for table in [self.tbl, tbl_rf]:
            actual_art, = self.fn(table=table, phylogeny=self.tre)
            actual = actual_art.view(skbio.DistanceMatrix)
            self.assertEqual(actual.ids, self.expected.ids)
            for id1 in actual.ids:
                for id2 in actual.ids:
                    npt.assert_almost_equal(actual[id1, id2],
                                            self.expected[id1, id2])

    def test_missing_tips_tree(self):
        tre = self.artifact('Phylogeny[Rooted]', 'root_only.tree')
        with self.assertRaises(CalledProcessError):
            obs = self.fn(self.tbl, tre)
            self.assertTrue('not a subset of the tree tips', obs.stderr)

    def test_extra_tip_tree(self):
        tbl = self.artifact('FeatureTable[Frequency]',
                            'two_feature_table.biom')

        tre_2 = self.artifact('Phylogeny[Rooted]', 'two_feature.tree')
        obs_2_art, = self.fn(tbl, tre_2)
        obs_2 = obs_2_art.view(skbio.DistanceMatrix)

        tre_3 = self.artifact('Phylogeny[Rooted]', 'three_feature.tree')
        obs_3_art, = self.fn(tbl, tre_3)
        obs_3 = obs_3_art.view(skbio.DistanceMatrix)

        self.assertEqual(obs_2, obs_3)


class BetaPhylogeneticMetaPassthroughTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        self.fn = self.plugin.actions['beta_phylogenetic_meta_passthrough']

        self.empty_tbl = Artifact.import_data(
            'FeatureTable[Frequency]', biom.Table(np.array([]), [], []))

        # checking parity with the unifrac.meta tests
        def unifrac_data(fn):
            path = os.path.join('data', fn)
            return pkg_resources.resource_filename('unifrac.tests', path)

        self.tables = [
            Artifact.import_data('FeatureTable[Frequency]',
                                 unifrac_data('e1.biom')),
            Artifact.import_data('FeatureTable[Frequency]',
                                 unifrac_data('e2.biom')),
        ]

        self.trees = [
            Artifact.import_data('Phylogeny[Rooted]',
                                 unifrac_data('t1.newick')),
            Artifact.import_data('Phylogeny[Rooted]',
                                 unifrac_data('t2.newick')),
        ]

    def test_method(self):
        for metric in METRICS['PHYLO']['UNIMPL']:
            obs_art, = self.fn(tables=self.tables, phylogenies=self.trees,
                               metric=metric)
            obs = obs_art.view(skbio.DistanceMatrix)
            self.assertEqual(('A', 'B', 'C'), obs.ids)
            self.assertEqual((3, 3), obs.shape)

    def test_passed_bad_metric(self):
        with self.assertRaisesRegex(TypeError,
                                    'imaginary_metric.*incompatible'):
            self.fn(tables=self.tables, phylogenies=self.trees,
                    metric='imaginary_metric')


class BetaPassthroughTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        self.method = self.plugin.actions['beta_passthrough']
        empty_table = biom.Table(np.array([]), [], [])
        self.empty_table = Artifact.import_data('FeatureTable[Frequency]',
                                                empty_table)
        crawford_tbl = self.get_data_path('crawford.biom')
        self.crawford_tbl = Artifact.import_data('FeatureTable[Frequency]',
                                                 crawford_tbl)
        table = biom.Table(np.array([[0, 1, 3], [1, 1, 2]]),
                           ['O1', 'O2'],
                           ['S1', 'S2', 'S3'])
        self.table = Artifact.import_data('FeatureTable[Frequency]', table)

    def test_method(self):
        for metric in METRICS['NONPHYLO']['UNIMPL']:
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
        # beta_passthrough does not provide access to measures that have been
        # implemented locally
        for metric in METRICS['NONPHYLO']['IMPL']:
            with self.assertRaisesRegex(TypeError, f"{metric}.*incompatible"):
                self.method(table=self.crawford_tbl, metric=metric)

    def test_aitchison(self):
        actual, = self.method(table=self.table, metric='aitchison')
        actual = actual.view(skbio.DistanceMatrix)
        expected = skbio.DistanceMatrix([[0.0000000, 0.4901290, 0.6935510],
                                         [0.4901290, 0.0000000, 0.2034219],
                                         [0.6935510, 0.2034219, 0.0000000]],
                                        ids=['S1', 'S2', 'S3'])

        self.assertEqual(actual.ids, expected.ids)
        for id1 in actual.ids:
            for id2 in actual.ids:
                npt.assert_almost_equal(actual[id1, id2], expected[id1, id2])

    def test_canberra_adkins(self):
        t = biom.Table(np.array([[0, 0], [0, 1], [1, 2]]),
                       ['O1', 'O2', 'O3'],
                       ['S1', 'S2'])
        t = Artifact.import_data('FeatureTable[Frequency]', t)
        # expected calculated by hand
        expected = skbio.DistanceMatrix(np.array([[0.0, 0.66666666666],
                                                  [0.66666666666, 0.0]]),
                                        ids=['S1', 'S2'])
        actual, = self.method(table=t, metric='canberra_adkins')
        actual = actual.view(skbio.DistanceMatrix)

        self.assertEqual(actual.ids, expected.ids)
        for id1 in actual.ids:
            for id2 in actual.ids:
                npt.assert_almost_equal(actual[id1, id2], expected[id1, id2])

    def test_beta_canberra_adkins_negative_values(self):
        t = biom.Table(np.array([[0, 0], [0, 1], [-1, -2]]),
                       ['O1', 'O2', 'O3'],
                       ['S1', 'S2'])
        t = Artifact.import_data('FeatureTable[Frequency]', t)

        with self.assertRaisesRegex(ValueError, '.*negative values'):
            self.method(table=t, metric='canberra_adkins')

    def test_jensenshannon(self):
        # expected computed with scipy.spatial.distance.jensenshannon
        expected = skbio.DistanceMatrix([[0.0000000, 0.4645014, 0.52379239],
                                         [0.4645014, 0.0000000, 0.07112939],
                                         [0.52379239, 0.07112939, 0.0000000]],
                                        ids=['S1', 'S2', 'S3'])

        actual, = self.method(table=self.table, metric='jensenshannon')
        actual = actual.view(skbio.DistanceMatrix)
        self.assertEqual(actual.ids, expected.ids)
        for id1 in actual.ids:
            for id2 in actual.ids:
                npt.assert_almost_equal(actual[id1, id2], expected[id1, id2])


class BetaPhylogeneticPassthroughTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def artifact(self, semantic_type, fn):
        return Artifact.import_data(semantic_type, self.get_data_path(fn))

    def setUp(self):
        super().setUp()

        self.fn = self.plugin.actions['beta_phylogenetic_passthrough']

        self.tbl = self.artifact('FeatureTable[Frequency]', 'crawford.biom')
        self.tre = self.artifact('Phylogeny[Rooted]', 'crawford.nwk')

    def test_method(self):
        for metric in METRICS['PHYLO']['UNIMPL']:
            obs_art, = self.fn(table=self.tbl, phylogeny=self.tre,
                               metric=metric)
            obs = obs_art.view(skbio.DistanceMatrix)
            self.assertEqual(9, len(obs.ids))
            self.assertEqual((9, 9), obs.shape)

        # variance adjusted
        for metric in METRICS['PHYLO']['UNIMPL']:
            obs_art, = self.fn(table=self.tbl, phylogeny=self.tre,
                               metric=metric, variance_adjusted=True)
            obs = obs_art.view(skbio.DistanceMatrix)
            self.assertEqual(9, len(obs.ids))
            self.assertEqual((9, 9), obs.shape)

        # bypass tips
        for metric in METRICS['PHYLO']['UNIMPL']:
            obs_art, = self.fn(table=self.tbl, phylogeny=self.tre,
                               metric=metric, bypass_tips=True)
            obs = obs_art.view(skbio.DistanceMatrix)
            self.assertEqual(9, len(obs.ids))
            self.assertEqual((9, 9), obs.shape)

    def test_passed_empty_table(self):
        tbl = Artifact.import_data(
            'FeatureTable[Frequency]', biom.Table(np.array([]), [], []))

        for metric in METRICS['PHYLO']['UNIMPL']:
            with self.assertRaisesRegex(ValueError, 'empty'):
                self.fn(table=tbl, phylogeny=self.tre, metric=metric)

    def test_passed_bad_metric(self):
        with self.assertRaisesRegex(TypeError, 'imaginary.*incompatible'):
            self.fn(table=self.tbl, phylogeny=self.tre, metric='imaginary')

    def test_beta_phylogenetic_alpha_on_non_generalized(self):
        with self.assertRaisesRegex(ValueError, 'The alpha parameter is only '
                                    'allowed when the selected metric is '
                                    '\'generalized_unifrac\''):
            self.fn(table=self.tbl, phylogeny=self.tre,
                    metric='unweighted_unifrac', alpha=0.11)
