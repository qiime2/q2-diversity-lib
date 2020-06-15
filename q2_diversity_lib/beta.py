# ----------------------------------------------------------------------------
# Copyright (c) 2018-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from functools import partial

import biom
import skbio.diversity
import sklearn.metrics
import unifrac

from q2_types.feature_table import BIOMV210Format
from q2_types.tree import NewickFormat
from ._util import (_disallow_empty_tables,
                    _validate_requested_cpus)


# --------------------Non-Phylogenetic-----------------------
@_disallow_empty_tables
@_validate_requested_cpus
def bray_curtis(table: biom.Table, n_jobs: int = 1) -> skbio.DistanceMatrix:
    counts = table.matrix_data.toarray().T
    sample_ids = table.ids(axis='sample')
    return skbio.diversity.beta_diversity(
        metric='braycurtis',
        counts=counts,
        ids=sample_ids,
        validate=True,
        pairwise_func=sklearn.metrics.pairwise_distances,
        n_jobs=n_jobs
    )


@_disallow_empty_tables
@_validate_requested_cpus
def jaccard(table: biom.Table, n_jobs: int = 1) -> skbio.DistanceMatrix:
    counts = table.matrix_data.toarray().T
    sample_ids = table.ids(axis='sample')
    return skbio.diversity.beta_diversity(
        metric='jaccard',
        counts=counts,
        ids=sample_ids,
        validate=True,
        pairwise_func=sklearn.metrics.pairwise_distances,
        n_jobs=n_jobs
    )


# ------------------------Phylogenetic-----------------------
@_disallow_empty_tables
@_validate_requested_cpus
def unweighted_unifrac(table: BIOMV210Format,
                       phylogeny: NewickFormat,
                       threads: int = 1,
                       bypass_tips: bool = False) -> skbio.DistanceMatrix:
    return unifrac.unweighted(str(table), str(phylogeny), threads=threads,
                              variance_adjusted=False, bypass_tips=bypass_tips)


@_disallow_empty_tables
@_validate_requested_cpus
def weighted_unnormalized_unifrac(table: BIOMV210Format,
                                  phylogeny: NewickFormat,
                                  threads: int = 1,
                                  bypass_tips: bool = False
                                  ) -> skbio.DistanceMatrix:
    return unifrac.weighted_unnormalized(str(table), str(phylogeny),
                                         threads=threads,
                                         variance_adjusted=False,
                                         bypass_tips=bypass_tips)


def implemented_phylogenetic_metrics_dict():
    return {'unweighted_unifrac': unweighted_unifrac,
            'weighted_unnormalized_unifrac': weighted_unnormalized_unifrac}


def unimplemented_phylogenetic_metrics_dict():
    return {'unweighted_unifrac': unifrac.unweighted,
            'weighted_unifrac': unifrac.weighted_unnormalized,
            'weighted_normalized_unifrac': unifrac.weighted_normalized,
            'generalized_unifrac': unifrac.generalized}


def all_phylogenetic_metrics():
    return implemented_phylogenetic_metrics_dict().keys() \
            | unimplemented_phylogenetic_metrics_dict().keys()


@_disallow_empty_tables
@_validate_requested_cpus
def beta_phylogenetic_dispatch(table: BIOMV210Format, phylogeny: NewickFormat,
                               metric: str, threads: int = 1,
                               variance_adjusted: bool = False,
                               alpha: float = None,
                               bypass_tips: bool = False
                               ) -> skbio.DistanceMatrix:

    metrics = all_phylogenetic_metrics()
    generalized_unifrac = 'generalized_unifrac'

    if metric not in metrics:
        raise ValueError("Unknown metric: %s" % metric)

    if alpha is not None and metric != generalized_unifrac:
        raise ValueError('The alpha parameter is only allowed when the choice'
                         ' of metric is generalized_unifrac')

    if metric == generalized_unifrac:
        alpha = 1.0 if alpha is None else alpha
        func = partial(metrics[metric], alpha=alpha)
    else:
        func = metrics[metric]

    if metric in ('unweighted_unifrac', 'weighted_normalized_unifrac') \
            and not variance_adjusted:
        appropriate_metrics = implemented_phylogenetic_metrics_dict()
    else:
        appropriate_metrics = unimplemented_phylogenetic_metrics_dict()

    func = appropriate_metrics[metric]

    # unifrac processes tables and trees should be filenames
    return func(str(table), str(phylogeny), threads=threads,
                variance_adjusted=variance_adjusted, bypass_tips=bypass_tips)
