# ----------------------------------------------------------------------------
# Copyright (c) 2018-2021, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import biom
import numpy as np
import pkg_resources

from qiime2 import Artifact

def get_test_data_path(filename):
    return pkg_resources.resource_filename('q2_diversity_lib.tests',
                                           f'data/{filename}')

s_ids_1 = ['S1', 'S2', 'S3', 'S4', 'S5']
def ft1_factory():
    return Artifact.import_data(
        'FeatureTable[Frequency]',
        biom.Table(np.array([[1, 0, .5, 999, 1],
                            [0, 1, 2, 0, 5],
                            [0, 0, 0, 1, 10]]),
                   ['A', 'B', 'C'],
                   s_ids_1)
    )

def tree_factory():
    input_tree_fp = get_test_data_path('faith_test.tree')
    return Artifact.import_data(
        'Phylogeny[Rooted]', input_tree_fp
    )

def faith_pd_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    tree = use.init_artifact('phylogeny', tree_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='faith_pd'),
        use.UsageInputs(table=ft, phylogeny=tree),
        use.UsageOutputNames(vector='faith_pd_vector'))

    result.assert_output_type('SampleData[AlphaDiversity]')

def observed_features_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='observed_features'),
        use.UsageInputs(table=ft),
        use.UsageOutputNames(vector='obs_feat_vector'))

    result.assert_output_type('SampleData[AlphaDiversity]')

    # AssertRegex for lines in output files
    exp = zip(s_ids_1, [1, 1, 2, 2, 3])
    for id, val in exp:
        result.assert_has_line_matching(
            path='alpha-diversity.tsv',
            expression=f'{id}\t{val}'
        )

def pielou_evenness_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='pielou_evenness'),
        use.UsageInputs(table=ft),
        use.UsageOutputNames(vector='pielou_vector')
    )

    result.assert_output_type('SampleData[AlphaDiversity]')

def pielou_drop_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='pielou_evenness'),
        use.UsageInputs(table=ft, drop_undefined_samples=True),
        use.UsageOutputNames(vector='pielou_vector')
    )

    result.assert_output_type('SampleData[AlphaDiversity]')

def shannon_entropy_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='shannon_entropy'),
        use.UsageInputs(table=ft),
        use.UsageOutputNames(vector='shannon_vector')
    )

    result.assert_output_type('SampleData[AlphaDiversity]')

def shannon_drop_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='shannon_entropy'),
        use.UsageInputs(table=ft, drop_undefined_samples=True),
        use.UsageOutputNames(vector='shannon_vector')
    )

    result.assert_output_type('SampleData[AlphaDiversity]')
