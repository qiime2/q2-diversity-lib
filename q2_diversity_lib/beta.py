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


# ----------Collections to simplify dispatch process------------------------
def implemented_nonphylogenetic_metrics_dict():
    return {'braycurtis': bray_curtis,
            'jaccard': jaccard}


def implemented_nonphylogenetic_metrics():
    return set(implemented_nonphylogenetic_metrics_dict())


def unimplemented_nonphylogenetic_metrics():
    return {'cityblock', 'euclidean', 'seuclidean', 'sqeuclidean', 'cosine',
            'correlation', 'hamming', 'chebyshev', 'canberra', 'mahalanobis',
            'yule', 'matching', 'dice', 'kulsinski', 'rogerstanimoto',
            'russellrao', 'sokalmichener', 'sokalsneath', 'wminkowski',
            'aitchison', 'canberra_adkins', 'jensenshannon'}


def all_nonphylogenetic_metrics():
    return implemented_nonphylogenetic_metrics() | \
           unimplemented_nonphylogenetic_metrics()


def implemented_phylogenetic_metrics_dict():
    return {'unweighted_unifrac': unweighted_unifrac,
            'weighted_unnormalized_unifrac': weighted_unnormalized_unifrac}


def unimplemented_phylogenetic_metrics_dict():
    return {'unweighted_unifrac': unifrac.unweighted,
            'weighted_unnormalized_unifrac': unifrac.weighted_unnormalized,
            'weighted_normalized_unifrac': unifrac.weighted_normalized,
            'generalized_unifrac': unifrac.generalized}


def all_phylogenetic_metrics_dict():
    return {**implemented_phylogenetic_metrics_dict(),
            **unimplemented_phylogenetic_metrics_dict()}


def all_phylogenetic_metrics():
    return set(all_phylogenetic_metrics_dict())


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


# TODO: can a pipeline function without any imported Actions? i.e. can it just
# implement its own behavior, as would happen if this pipeline ran skbio calcs?
# TODO: import these decorators
@_disallow_empty_tables
@_validate_requested_cpus
def beta_dispatch(table: BIOMV210Format, metric: str, pseudocount: int = 1,
                  n_jobs: int = 1) -> skbio.DistanceMatrix:

    all_metrics = all_nonphylogenetic_metrics()
    implemented_metrics = implemented_nonphylogenetic_metrics_dict()

    if metric not in all_metrics:
        raise ValueError("Unknown metric: %s" % metric)

    counts = table.matrix_data.toarray().T
    sample_ids = table.ids(axis='sample')

    if metric in implemented_metrics:
        func = implemented_nonphylogenetic_metrics_dict()[metric]
    else:
        func = skbio.diversity.beta_diversity

    # TODO: define and deal with locally-implemented metrics

    result = func(metric=metric, counts=counts, ids=sample_ids,
                  validate=True,
                  pairwise_func=sklearn.metrics.pairwise_distances,
                  n_jobs=n_jobs)
    # TODO: tuple-ize result, and adapt this as a pipeline
    return result


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


@_disallow_empty_tables
@_validate_requested_cpus
def beta_phylogenetic_dispatch(table: BIOMV210Format, phylogeny: NewickFormat,
                               metric: str, threads: int = 1,
                               variance_adjusted: bool = False,
                               alpha: float = None,
                               bypass_tips: bool = False
                               ) -> skbio.DistanceMatrix:

    metrics = all_phylogenetic_metrics_dict()
    generalized_unifrac = 'generalized_unifrac'

    if metric not in metrics:
        raise ValueError("Unknown metric: %s" % metric)

    if alpha is not None and metric != generalized_unifrac:
        raise ValueError('The alpha parameter is only allowed when the choice'
                         ' of metric is generalized_unifrac')

    # HACK: this logic will be simpler once the remaining unifracs are done
    if metric in ('unweighted_unifrac', 'weighted_normalized_unifrac') \
            and not variance_adjusted:
        func = implemented_phylogenetic_metrics_dict()[metric]
    else:
        if metric == generalized_unifrac:
            alpha = 1.0 if alpha is None else alpha
            func = partial(unimplemented_phylogenetic_metrics_dict()[metric],
                           alpha=alpha,
                           variance_adjusted=variance_adjusted)
        else:
            func = partial(unimplemented_phylogenetic_metrics_dict()[metric],
                           variance_adjusted=variance_adjusted)

    result = func(table, phylogeny, threads=threads,
                  bypass_tips=bypass_tips)
    return result
