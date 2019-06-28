# ----------------------------------------------------------------------------
# Copyright (c) 2018-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin.testing import TestPluginBase
from q2_diversity_lib import faith_pd

import io
import biom
import skbio
import numpy as np
import pandas as pd
import pandas.util.testing as pdt

import copy


class AlphaTests(TestPluginBase):

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

    def test_faith_pd(self):
        actual = faith_pd(table=self.input_table, phylogeny=self.input_tree)
        pdt.assert_series_equal(actual, self.faith_pd_expected)

    def test_faith_pd_error_rewriting(self):
        tree = skbio.TreeNode.read(io.StringIO(
            '((A:0.3):0.2, C:100)root;'))
        with self.assertRaisesRegex(skbio.tree.MissingNodeError,
                                    'feature_ids.*phylogeny'):
            faith_pd(table=self.input_table, phylogeny=tree)

    def test_all_accepted_types_have_consistent_behavior(self):
        freq_table = self.input_table
        rel_freq_table = copy.deepcopy(self.input_table).norm(axis='sample',
                                                              inplace=False)
        p_a_table = copy.deepcopy(self.input_table).pa()
        accepted_tables = [freq_table, rel_freq_table, p_a_table]
        for table in accepted_tables:
            actual = faith_pd(table=table, phylogeny=self.input_tree)
            pdt.assert_series_equal(actual, self.faith_pd_expected)

    def test_alpha_phylogenetic_empty_table(self):
        empty_table = biom.Table(np.array([]), [], [])
        with self.assertRaisesRegex(ValueError, "empty"):
            faith_pd(table=empty_table, phylogeny=self.input_tree)
