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


class AlphaTests(TestPluginBase):

    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        self.input_table = biom.Table(np.array([[0, 1, 3], [1, 1, 2]]),
                                      ['O1', 'O2'],
                                      ['S1', 'S2', 'S3'])

    def test_faith_pd(self):
        tree = skbio.TreeNode.read(io.StringIO(
            '((O1:0.25, O2:0.50):0.25, O3:0.75)root;'))
        actual = faith_pd(table=self.input_table, phylogeny=tree)
        # expected computed with skbio.diversity.alpha_diversity
        expected = pd.Series({'S1': 0.75, 'S2': 1.0, 'S3': 1.0},
                             name='faith_pd')
        pdt.assert_series_equal(actual, expected)

    def test_faith_pd_error_rewriting(self):
        tree = skbio.TreeNode.read(io.StringIO(
            '((O1:0.25):0.25, O3:0.75)root;'))
        # Verify through regex that there is a ``feature_ids`` substring
        # followed by a ``phylogeny``
        with self.assertRaisesRegex(skbio.tree.MissingNodeError,
                                    'feature_ids.*phylogeny'):
            faith_pd(table=self.input_table, phylogeny=tree)

    def test_alpha_phylogenetic_empty_table(self):
        empty_table = biom.Table(np.array([]), [], [])
        tree = skbio.TreeNode.read(io.StringIO(
            '((O1:0.25):0.25, O3:0.75)root;'))
        with self.assertRaisesRegex(ValueError, "empty"):
            faith_pd(table=empty_table, phylogeny=tree)
