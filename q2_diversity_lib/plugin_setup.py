# ----------------------------------------------------------------------------
# Copyright (c) 2018-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import q2_diversity_lib
from qiime2.plugin import (Plugin, Citations, Bool)

from q2_types.feature_table import (FeatureTable, Frequency, RelativeFrequency,
                                    PresenceAbsence)
from q2_types.tree import Phylogeny, Rooted
from q2_types.sample_data import AlphaDiversity, SampleData

citations = Citations.load('citations.bib', package='q2_diversity_lib')
plugin = Plugin(
    name='diversity-lib',
    version=q2_diversity_lib.__version__,
    website='https://github.com/qiime2/q2-diversity-lib',
    short_description='Plugin for computing community diversity.',
    package='q2_diversity_lib',
    description='This QIIME 2 plugin computes individual metrics for '
    ' community alpha and beta diversity.',
)

plugin.methods.register_function(
    function=q2_diversity_lib.faith_pd,
    inputs={'table': FeatureTable[Frequency | RelativeFrequency
            | PresenceAbsence],
            'phylogeny': Phylogeny[Rooted]},
    parameters=None,
    outputs=[('vector',
              SampleData[AlphaDiversity])],
    input_descriptions={
        'table': 'The feature table containing the samples for which Faith\'s '
                 'phylogenetic diversity should be computed. Table values '
                 'will be converted to presence/absence.',
        'phylogeny': 'Phylogenetic tree containing tip identifiers that '
                     'correspond to the feature identifiers in the table. '
                     'This tree can contain tip ids that are not present in '
                     'the table, but all feature ids in the table must be '
                     'present in this tree.'},
    parameter_descriptions=None,
    output_descriptions={'vector': 'Vector containing per-sample values for '
                                   'Faith\'s Phylogenetic Diversity.'},
    name='Faith\'s Phylogenetic Diversity',
    description='Computes Faith\'s Phylogenetic Diversity for all samples in '
                'a feature table.',
    citations=[citations['faith1992conservation']]
)

plugin.methods.register_function(
    function=q2_diversity_lib.observed_features,
    inputs={'table': FeatureTable[Frequency | RelativeFrequency
            | PresenceAbsence]},
    parameters=None,
    outputs=[('vector',
             SampleData[AlphaDiversity])],
    input_descriptions={'table': 'The feature table containing the samples '
                        'for which the number of observed features should be '
                        'calculated. Table values will be converted to '
                        'presence/absence.'},
    parameter_descriptions=None,
    output_descriptions={'vector': 'Vector containing per-sample counts of '
                                   'observed features.'},
    name='Observed Features',
    description='Compute the number of observed features for each sample in a '
                'feature table',
)

plugin.methods.register_function(
    function=q2_diversity_lib.pielou_evenness,
    inputs={'table': FeatureTable[Frequency | RelativeFrequency]},
    parameters={'drop_undefined_samples': Bool},
    outputs=[('vector',
             SampleData[AlphaDiversity])],
    input_descriptions={'table': 'The feature table containing the samples '
                        'for which Pielou\'s evenness should be computed.'},
    parameter_descriptions={'drop_undefined_samples': 'Samples with fewer than'
                            ' two observed features produce undefined (NaN) '
                            'values. If true, these samples are dropped '
                            'from the output vector.'},
    output_descriptions={'vector': 'Vector containing per-sample values '
                                   'for Pielou\'s Evenness.'},
    name='Pielou\'s Evenness',
    description='Compute Pielou\'s Evenness for each sample in a '
                'feature table',
    citations=[citations['pielou1966measurement']]
)

plugin.methods.register_function(
    function=q2_diversity_lib.shannon_entropy,
    inputs={'table': FeatureTable[Frequency | RelativeFrequency]},
    parameters={'drop_undefined_samples': Bool},
    outputs=[('vector',
             SampleData[AlphaDiversity])],
    input_descriptions={'table': 'The feature table containing the samples '
                        'for which Shannon\'s Entropy should be computed.'},
    parameter_descriptions={'drop_undefined_samples': 'Samples with no '
                            'observed features produce undefined (NaN) values.'
                            ' If true, these samples are dropped from the '
                            'output vector.'},
    output_descriptions={'vector': 'Vector containing per-sample values '
                                   'for Shannon\'s Entropy.'},
    name='Shannon\'s Entropy',
    description='Compute Shannon\'s Entropy for each sample in a '
                'feature table',
    citations=[citations['shannon1948communication']]
)
