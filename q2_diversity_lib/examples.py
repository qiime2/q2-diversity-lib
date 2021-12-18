# ----------------------------------------------------------------------------
# Copyright (c) 2018-2021, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import biom
import numpy as np

from qiime2 import Artifact

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


def observed_features_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    result, = use.action(
    # This also works just fine, and is probably more idiomatic.
    # We're mutating the injected usage, and it's recording the process.
    # use.action(
        # NOTE: This uses the plugin ID, not the registered plugin name.
        # IDs are "normalized" by str.replace-ing dashes with underscores
        # This is likely done to reduce friction around imports
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='observed_features'),
        use.UsageInputs(table=ft),
        use.UsageOutputNames(vector='obs_feat_vector'))

    use.comment(f'Our result vector is named {result.name},'
                f' and it is a(n) {result.var_type}')

    # Check semantic type of output (and presumably that the example ran)
    # result.assert_output_type('SampleData[gerbil]')
    result.assert_output_type('SampleData[AlphaDiversity]')

    # AssertRegex for lines in output files
    exp = zip(s_ids_1, [1, 1, 2, 2, 3])
    for id, val in exp:
        result.assert_has_line_matching(
            path='alpha-diversity.tsv',
            expression=f'{id}\t{val}'
        )

