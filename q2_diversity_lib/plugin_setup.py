# ----------------------------------------------------------------------------
# Copyright (c) 2018-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import q2_diversity_lib
from qiime2.plugin import (Plugin, Citations, Bool, Int, Range, Choices, Str)

from q2_types.feature_table import (FeatureTable, Frequency, RelativeFrequency,
                                    PresenceAbsence)
from q2_types.tree import Phylogeny, Rooted
from q2_types.sample_data import AlphaDiversity, SampleData
from q2_types.distance_matrix import DistanceMatrix

citations = Citations.load('citations.bib', package='q2_diversity_lib')
plugin = Plugin(
    name="diversity-lib",
    version=q2_diversity_lib.__version__,
    website="https://github.com/qiime2/q2-diversity-lib",
    short_description="Plugin for computing community diversity.",
    package="q2_diversity_lib",
    description="This QIIME 2 plugin computes individual metrics for "
    " community alpha and beta diversity.",
)

# ------------------------ alpha-diversity -----------------------
plugin.methods.register_function(
    function=q2_diversity_lib.faith_pd,
    inputs={'table': FeatureTable[Frequency | RelativeFrequency
            | PresenceAbsence],
            'phylogeny': Phylogeny[Rooted]},
    parameters=None,
    outputs=[('vector',
              SampleData[AlphaDiversity])],
    input_descriptions={
        'table': "The feature table containing the samples for which Faith's "
                 "phylogenetic diversity should be computed. Table values "
                 "will be converted to presence/absence.",
        'phylogeny': "Phylogenetic tree containing tip identifiers that "
                     "correspond to the feature identifiers in the table. "
                     "This tree can contain tip ids that are not present in "
                     "the table, but all feature ids in the table must be "
                     "present in this tree."},
    parameter_descriptions=None,
    output_descriptions={'vector': "Vector containing per-sample values for "
                                   "Faith's Phylogenetic Diversity."},
    name="Faith's Phylogenetic Diversity",
    description="Computes Faith's Phylogenetic Diversity for all samples in "
                "a feature table.",
    citations=[citations['faith1992conservation']]
)

plugin.methods.register_function(
    function=q2_diversity_lib.observed_features,
    inputs={'table': FeatureTable[Frequency | RelativeFrequency
            | PresenceAbsence]},
    parameters=None,
    outputs=[('vector',
             SampleData[AlphaDiversity])],
    input_descriptions={'table': "The feature table containing the samples "
                        "for which the number of observed features should be "
                        "calculated. Table values will be converted to "
                        "presence/absence."},
    parameter_descriptions=None,
    output_descriptions={'vector': "Vector containing per-sample counts of "
                                   "observed features."},
    name="Observed Features",
    description="Compute the number of observed features for each sample in a "
                "feature table"
)

plugin.methods.register_function(
    function=q2_diversity_lib.pielou_evenness,
    inputs={'table': FeatureTable[Frequency | RelativeFrequency]},
    parameters={'drop_undefined_samples': Bool},
    outputs=[('vector',
             SampleData[AlphaDiversity])],
    input_descriptions={'table': "The feature table containing the samples "
                        "for which Pielou's evenness should be computed."},
    parameter_descriptions={'drop_undefined_samples': "Samples with fewer than"
                            " two observed features produce undefined (NaN) "
                            "values. If true, these samples are dropped "
                            "from the output vector."},
    output_descriptions={'vector': "Vector containing per-sample values "
                                   "for Pielou's Evenness."},
    name="Pielou's Evenness",
    description="Compute Pielou's Evenness for each sample in a "
                "feature table",
    citations=[citations['pielou1966measurement']]
)

# TODO: Augment citations as needed
plugin.methods.register_function(
    function=q2_diversity_lib.shannon_entropy,
    inputs={'table': FeatureTable[Frequency | RelativeFrequency]},
    parameters={'drop_undefined_samples': Bool},
    outputs=[('vector',
             SampleData[AlphaDiversity])],
    input_descriptions={'table': "The feature table containing the samples "
                        "for which Shannon's Entropy should be computed."},
    parameter_descriptions={'drop_undefined_samples': "Samples with no "
                            "observed features produce undefined (NaN) values."
                            " If true, these samples are dropped from the "
                            "output vector."},
    output_descriptions={'vector': "Vector containing per-sample values "
                                   "for Shannon's Entropy."},
    name="Shannon's Entropy",
    description="Compute Shannon's Entropy for each sample in a "
                "feature table",
    citations=[citations['shannon1948communication']]
)

# ------------------------ beta-diversity -----------------------
# TODO: Augment citations as needed
plugin.methods.register_function(
    function=q2_diversity_lib.bray_curtis,
    inputs={'table': FeatureTable[Frequency]},
    parameters={'n_jobs': Int % Range(1, None) | Str % Choices(['auto'])},
    outputs=[('distance_matrix', DistanceMatrix)],
    input_descriptions={
        'table': "The feature table containing the samples for which "
                 "Bray-Curtis dissimilarity should be computed."},
    parameter_descriptions={
        'n_jobs': "The number of concurrent jobs to use in performing this "
                  "calculation. May not exceed the number of available "
                  "physical cores. If n_jobs = 'auto', one job will be "
                  "launched for each identified CPU core on the host."},
    output_descriptions={
        'distance_matrix': "Distance matrix for Bray-Curtis dissimilarity"},
    name="Bray-Curtis Dissimilarity",
    description="Compute Bray-Curtis dissimilarity for each sample in a "
                "feature table. Note: Frequency and relative frequency data "
                "produce different results unless overall sample sizes are "
                "identical. Please consider the impact on your results if "
                "you use Bray-Curtis with count data that has not been "
                "adjusted (normalized).",
    citations=[citations['sorensen1948method']])

# TODO: Augment citations as needed
plugin.methods.register_function(
    function=q2_diversity_lib.jaccard,
    inputs={'table': FeatureTable[Frequency | RelativeFrequency
            | PresenceAbsence]},
    parameters={'n_jobs': Int % Range(1, None) | Str % Choices(['auto'])},
    outputs=[('distance_matrix', DistanceMatrix)],
    input_descriptions={
        'table': "The feature table containing the samples for which "
                 "Jaccard distance should be computed."},
    parameter_descriptions={
        'n_jobs': "The number of concurrent jobs to use in performing this "
                  "calculation. May not exceed the number of available "
                  "physical cores. If n_jobs = 'auto', one job will be "
                  "launched for each identified CPU core on the host."},
    output_descriptions={
        'distance_matrix': "Distance matrix for Jaccard index"},
    name="Jaccard Distance",
    description="Compute Jaccard distance for each sample "
                "in a feature table. Jaccard is calculated using"
                "presence/absence data. Data of type "
                "FeatureTable[Frequency | Relative Frequency] is reduced"
                "to presence/absence prior to calculation.",
    citations=[citations['jaccard1908nouvelles']])


plugin.methods.register_function(
    function=q2_diversity_lib.unweighted_unifrac,
    inputs={'table': FeatureTable[Frequency | RelativeFrequency
            | PresenceAbsence],
            'phylogeny': Phylogeny[Rooted]},
    parameters={'threads': Int % Range(1, None) | Str % Choices(['auto']),
                'bypass_tips': Bool},
    outputs=[('distance_matrix', DistanceMatrix)],
    input_descriptions={
        'table': "The feature table containing the samples for which "
                 "Unweighted Unifrac should be computed.",
        'phylogeny': "Phylogenetic tree containing tip identifiers that "
                     "correspond to the feature identifiers in the table. "
                     "This tree can contain tip ids that are not present in "
                     "the table, but all feature ids in the table must be "
                     "present in this tree."},
    parameter_descriptions={
        'threads': "The number of CPU threads to use in performing this "
                   "calculation. May not exceed the number of available "
                   "physical cores. If threads = 'auto', one thread will be "
                   "created for each identified CPU core on the host.",
        'bypass_tips':
            ("In a bifurcating tree, the tips make up about 50% of "
             "the nodes in a tree. By ignoring them, specificity "
             "can be traded for reduced compute time. This has the"
             " effect of collapsing the phylogeny, and is analogous"
             " (in concept) to moving from 99% to 97% OTUs")},
    output_descriptions={'distance_matrix': "Distance matrix for Unweighted "
                         "Unifrac."},
    name="Unweighted Unifrac",
    description="Compute Unweighted Unifrac for each sample in a "
                "feature table",
    citations=[
        citations['lozupone2005unifrac'],
        citations['lozupone2007unifrac'],
        citations['hamady2010unifrac'],
        citations['lozupone2011unifrac'],
        citations['mcdonald2018unifrac']]
)

plugin.methods.register_function(
    function=q2_diversity_lib.weighted_unifrac,
    inputs={'table': FeatureTable[Frequency | RelativeFrequency],
            'phylogeny': Phylogeny[Rooted]},
    parameters={'threads': Int % Range(1, None) | Str % Choices(['auto']),
                'bypass_tips': Bool},
    outputs=[('distance_matrix', DistanceMatrix)],
    input_descriptions={
        'table': "The feature table containing the samples for which "
                 "Weighted Unifrac should be computed.",
        'phylogeny': "Phylogenetic tree containing tip identifiers that "
                     "correspond to the feature identifiers in the table. "
                     "This tree can contain tip ids that are not present in "
                     "the table, but all feature ids in the table must be "
                     "present in this tree."},
    parameter_descriptions={
        'threads': "The number of CPU threads to use in performing this "
                   "calculation. May not exceed the number of available "
                   "physical cores. If threads = 'auto', one thread will be "
                   "created for each identified CPU core on the host.",
        'bypass_tips':
            ("In a bifurcating tree, the tips make up about 50% of "
             "the nodes in a tree. By ignoring them, specificity "
             "can be traded for reduced compute time. This has the"
             " effect of collapsing the phylogeny, and is analogous"
             " (in concept) to moving from 99% to 97% OTUs")},
    output_descriptions={'distance_matrix': "Distance matrix for Unweighted "
                         "Unifrac."},
    name="Weighted Unifrac",
    description="Compute Weighted Unifrac for each sample in a "
                "feature table",
    citations=[
        citations['lozupone2005unifrac'],
        citations['lozupone2007unifrac'],
        citations['hamady2010unifrac'],
        citations['lozupone2011unifrac'],
        citations['mcdonald2018unifrac']]
)
