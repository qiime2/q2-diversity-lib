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
s_ids_2 = ['S6', 'S7', 'S8', 'S9', 'S10']


def ft1_factory():
    return Artifact.import_data(
        'FeatureTable[Frequency]',
        biom.Table(np.array([[1, 0, 5, 999, 1],
                            [0, 1, 2, 0, 5],
                            [0, 0, 0, 1, 10]]),
                   ['A', 'B', 'C'],
                   s_ids_1)
    )


def ft2_factory():
    return Artifact.import_data(
        'FeatureTable[Frequency]',
        biom.Table(np.array([[1, 10, 5, 999, 1],
                            [5, 1, 2, 0, 5],
                            [1, 40, 30, 10, 0]]),
                   ['A', 'B', 'C'],
                   s_ids_2)
    )


def tree_factory():
    input_tree_fp = get_test_data_path('faith_test.tree')
    return Artifact.import_data(
        'Phylogeny[Rooted]', input_tree_fp
    )


# ------------------------ alpha-diversity -----------------------
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


# ------------------------ beta-diversity -----------------------
def bray_curtis_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='bray_curtis'),
        use.UsageInputs(table=ft),
        use.UsageOutputNames(distance_matrix='bray_curtis_dm')
    )
    result.assert_output_type('DistanceMatrix')


def bray_curtis_n_jobs_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='bray_curtis'),
        use.UsageInputs(table=ft, n_jobs=1),
        use.UsageOutputNames(distance_matrix='bray_curtis_dm')
    )
    result.assert_output_type('DistanceMatrix')


def bray_curtis_auto_jobs_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='bray_curtis'),
        use.UsageInputs(table=ft, n_jobs='auto'),
        use.UsageOutputNames(distance_matrix='bray_curtis_dm')
    )
    result.assert_output_type('DistanceMatrix')


def jaccard_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='jaccard'),
        use.UsageInputs(table=ft),
        use.UsageOutputNames(distance_matrix='jaccard_dm')
    )
    result.assert_output_type('DistanceMatrix')


def jaccard_n_jobs_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='jaccard'),
        use.UsageInputs(table=ft, n_jobs=1),
        use.UsageOutputNames(distance_matrix='jaccard_dm')
    )
    result.assert_output_type('DistanceMatrix')


def jaccard_auto_jobs_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='jaccard'),
        use.UsageInputs(table=ft, n_jobs='auto'),
        use.UsageOutputNames(distance_matrix='jaccard_dm')
    )
    result.assert_output_type('DistanceMatrix')


def u_u_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    tree = use.init_artifact('phylogeny', tree_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='unweighted_unifrac'),
        use.UsageInputs(table=ft, phylogeny=tree),
        use.UsageOutputNames(distance_matrix='unweighted_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')


def u_u_n_threads_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    tree = use.init_artifact('phylogeny', tree_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='unweighted_unifrac'),
        use.UsageInputs(table=ft, phylogeny=tree, threads=1),
        use.UsageOutputNames(distance_matrix='unweighted_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')


def u_u_auto_threads_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    tree = use.init_artifact('phylogeny', tree_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='unweighted_unifrac'),
        use.UsageInputs(table=ft, phylogeny=tree, threads='auto'),
        use.UsageOutputNames(distance_matrix='unweighted_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')


def u_u_bypass_tips_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    tree = use.init_artifact('phylogeny', tree_factory)
    use.comment("bypass_tips can be used with any threads setting, "
                "but auto may be a good choice if you're trimming run time.")
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='unweighted_unifrac'),
        use.UsageInputs(table=ft, phylogeny=tree,
                        threads='auto', bypass_tips=True),
        use.UsageOutputNames(distance_matrix='unweighted_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')


def w_u_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    tree = use.init_artifact('phylogeny', tree_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='weighted_unifrac'),
        use.UsageInputs(table=ft, phylogeny=tree),
        use.UsageOutputNames(distance_matrix='weighted_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')


def w_u_n_threads_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    tree = use.init_artifact('phylogeny', tree_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='weighted_unifrac'),
        use.UsageInputs(table=ft, phylogeny=tree, threads=1),
        use.UsageOutputNames(distance_matrix='weighted_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')


def w_u_auto_threads_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    tree = use.init_artifact('phylogeny', tree_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='weighted_unifrac'),
        use.UsageInputs(table=ft, phylogeny=tree, threads='auto'),
        use.UsageOutputNames(distance_matrix='weighted_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')


def w_u_bypass_tips_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    tree = use.init_artifact('phylogeny', tree_factory)
    use.comment("bypass_tips can be used with any threads setting, "
                "but auto may be a good choice if you're trimming run time.")
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='weighted_unifrac'),
        use.UsageInputs(table=ft, phylogeny=tree,
                        threads='auto', bypass_tips=True),
        use.UsageOutputNames(distance_matrix='weighted_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')


# ------------------------ Passthrough Methods ------------------------
def alpha_passthrough_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='alpha_passthrough'),
        use.UsageInputs(table=ft, metric='simpson'),
        use.UsageOutputNames(vector='simpson_vector')
    )
    result.assert_output_type('SampleData[AlphaDiversity]')


def beta_passthrough_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='beta_passthrough'),
        use.UsageInputs(table=ft, metric='euclidean'),
        use.UsageOutputNames(distance_matrix='euclidean_dm')
    )
    result.assert_output_type('DistanceMatrix')


def beta_passthrough_n_jobs_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='beta_passthrough'),
        use.UsageInputs(table=ft, metric='kulsinski', n_jobs=1),
        use.UsageOutputNames(distance_matrix='kulsinski_dm')
    )
    result.assert_output_type('DistanceMatrix')


def beta_passthrough_auto_jobs_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='beta_passthrough'),
        use.UsageInputs(table=ft, metric='aitchison', n_jobs='auto'),
        use.UsageOutputNames(distance_matrix='aitchison_dm')
    )
    use.comment("Here, a default pseudocount of 1 is added to feature counts. "
                "Pseudocount is ignored for non-compositional metrics.")
    result.assert_output_type('DistanceMatrix')


def beta_passthrough_pseudocount_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='beta_passthrough'),
        use.UsageInputs(table=ft, metric='aitchison', n_jobs='auto',
                        pseudocount=5),
        use.UsageOutputNames(distance_matrix='aitchison_dm')
    )
    result.assert_output_type('DistanceMatrix')


def beta_phylo_passthrough_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    tree = use.init_artifact('phylogeny', tree_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='beta_phylogenetic_passthrough'),
        use.UsageInputs(table=ft, phylogeny=tree,
                        metric='weighted_normalized_unifrac'),
        use.UsageOutputNames(distance_matrix='weighted_normalized_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')


def beta_phylo_passthrough_n_threads_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    tree = use.init_artifact('phylogeny', tree_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='beta_phylogenetic_passthrough'),
        use.UsageInputs(table=ft, phylogeny=tree, threads=1,
                        metric='weighted_normalized_unifrac'),
        use.UsageOutputNames(distance_matrix='weighted_normalized_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')


def beta_phylo_passthrough_auto_threads_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    tree = use.init_artifact('phylogeny', tree_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='beta_phylogenetic_passthrough'),
        use.UsageInputs(table=ft, phylogeny=tree, threads='auto',
                        metric='weighted_normalized_unifrac'),
        use.UsageOutputNames(distance_matrix='weighted_normalized_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')


def beta_phylo_passthrough_bypass_tips_example(use):
    ft = use.init_artifact('feature_table', ft1_factory)
    tree = use.init_artifact('phylogeny', tree_factory)
    use.comment("bypass_tips can be used with any threads setting, "
                "but auto may be a good choice if you're trimming run time.")
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='beta_phylogenetic_passthrough'),
        use.UsageInputs(table=ft, phylogeny=tree,
                        metric='weighted_normalized_unifrac',
                        threads='auto', bypass_tips=True),
        use.UsageOutputNames(distance_matrix='weighted_normalized_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')


def beta_phylo_passthrough_variance_adjusted_example(use):
    use.comment(
        "Chang et al's variance adjustment may be applied to any unifrac "
        "method by using this passthrough function.")
    ft = use.init_artifact('feature_table', ft1_factory)
    tree = use.init_artifact('phylogeny', tree_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='beta_phylogenetic_passthrough'),
        use.UsageInputs(table=ft, phylogeny=tree,
                        metric='weighted_unifrac',
                        threads='auto',
                        variance_adjusted=True),
        use.UsageOutputNames(distance_matrix='var_adj_weighted_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')


def beta_phylo_passthrough_min_generalized_unifrac_example(use):
    use.comment(
        "Generalized unifrac is passed alpha=1 by default. "
        "This is roughly equivalent to weighted normalized unifrac, "
        "which method will be used instead, because it is better optimized."
    )
    ft = use.init_artifact('feature_table', ft1_factory)
    tree = use.init_artifact('phylogeny', tree_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='beta_phylogenetic_passthrough'),
        use.UsageInputs(table=ft, phylogeny=tree,
                        metric='generalized_unifrac'),
        use.UsageOutputNames(distance_matrix='generalized_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')


def beta_phylo_passthrough_generalized_unifrac_example(use):
    use.comment("passing a float between 0 and 1 to 'alpha' gives you control "
                "over the importance of sample proportions.")
    ft = use.init_artifact('feature_table', ft1_factory)
    tree = use.init_artifact('phylogeny', tree_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='beta_phylogenetic_passthrough'),
        use.UsageInputs(table=ft, phylogeny=tree,
                        metric='generalized_unifrac',
                        alpha=0.75),
        use.UsageOutputNames(distance_matrix='generalized_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')


def beta_phylo_meta_passthrough_example(use):
    use.comment(
        "For brevity, these examples are focused on meta-specific parameters. "
        "See the documentation for beta_phylogenetic_passthrough for "
        "additional relevant information."
    )
    ft1 = use.init_artifact('feature_table1', ft1_factory)
    ft2 = use.init_artifact('feature_table2', ft2_factory)
    tree1 = use.init_artifact('phylogeny1', tree_factory)
    tree2 = use.init_artifact('phylogeny2', tree_factory)
    use.comment("NOTE: the number of trees and tables must match.")
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='beta_phylogenetic_meta_passthrough'),
        use.UsageInputs(tables=[ft1, ft2], phylogenies=[tree1, tree2],
                        metric='weighted_normalized_unifrac'),
        use.UsageOutputNames(distance_matrix='ft1_ft2_w_norm_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')


def beta_phylo_meta_weights_example(use):
    ft1 = use.init_artifact('feature_table1', ft1_factory)
    ft2 = use.init_artifact('feature_table2', ft2_factory)
    tree = use.init_artifact('phylogeny', tree_factory)
    use.comment("The number of weights must match the number of tables/trees.")
    use.comment("If meaningful, it is possible to pass the same phylogeny "
                "more than once.")
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='beta_phylogenetic_meta_passthrough'),
        use.UsageInputs(tables=[ft1, ft2], phylogenies=[tree, tree],
                        metric='weighted_normalized_unifrac',
                        weights=[3.0, 42.0]),
        use.UsageOutputNames(distance_matrix='ft1_ft2_w_norm_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')


def beta_phylo_meta_consolidation_example(use):
    ft1 = use.init_artifact('feature_table1', ft1_factory)
    ft2 = use.init_artifact('feature_table2', ft2_factory)
    tree1 = use.init_artifact('phylogeny1', tree_factory)
    tree2 = use.init_artifact('phylogeny2', tree_factory)
    result, = use.action(
        use.UsageAction(plugin_id='diversity_lib',
                        action_id='beta_phylogenetic_meta_passthrough'),
        use.UsageInputs(tables=[ft1, ft2], phylogenies=[tree1, tree2],
                        metric='weighted_normalized_unifrac',
                        weights=[0.4, 0.6],
                        consolidation='skipping_missing_values'),
        use.UsageOutputNames(distance_matrix='ft1_ft2_w_norm_unifrac_dm')
    )
    result.assert_output_type('DistanceMatrix')
