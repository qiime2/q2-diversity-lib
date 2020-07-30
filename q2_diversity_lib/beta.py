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
from skbio.stats.composition import clr
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import jensenshannon
import numpy as np

from q2_types.feature_table import BIOMV210Format
from q2_types.tree import NewickFormat
from ._util import (_disallow_empty_tables,
                    _validate_requested_cpus)


# NOTE: a metric may be in both implemented and unimplemented collections,
# if it is only implemented with certain params (e.g. both 'vanilla' and
# Variance Adjusted weighted unifracs use unifrac.weighted_unnormalized, but
# only 'vanilla' is currently implemented)
METRICS = {
    'PHYLO': {
        'IMPL': {'unweighted_unifrac', 'weighted_unifrac'},
        'UNIMPL': {'unweighted_unifrac', 'weighted_unifrac',
                   'weighted_normalized_unifrac', 'generalized_unifrac'},
    },
    'NONPHYLO': {
        'IMPL': {'braycurtis', 'jaccard'},
        'UNIMPL': {'cityblock', 'euclidean', 'seuclidean', 'sqeuclidean',
                   'cosine', 'correlation', 'hamming', 'chebyshev', 'canberra',
                   'mahalanobis', 'yule', 'matching', 'dice', 'kulsinski',
                   'rogerstanimoto', 'russellrao', 'sokalmichener',
                   'sokalsneath', 'wminkowski', 'aitchison', 'canberra_adkins',
                   'jensenshannon'}
    }
}

_all_phylo_metrics = METRICS['PHYLO']['IMPL'] | METRICS['PHYLO']['UNIMPL']
_all_nonphylo_metrics = METRICS['NONPHYLO']['IMPL'] \
                        | METRICS['NONPHYLO']['UNIMPL']


def unimplemented_phylogenetic_metrics_dict():
    return {'unweighted_unifrac': unifrac.unweighted,
            'weighted_unifrac': unifrac.weighted_unnormalized,
            'weighted_normalized_unifrac': unifrac.weighted_normalized,
            'generalized_unifrac': unifrac.generalized}


def local_method_names_dict():
    return {'braycurtis': 'bray_curtis',
            'jaccard': 'jaccard',
            'unweighted_unifrac': 'unweighted_unifrac',
            'weighted_unifrac': 'weighted_unifrac'}


# -------------------- Method Dispatch -----------------------
def beta_dispatch(ctx, table, metric, pseudocount=1, n_jobs=1):
    all_metrics = _all_nonphylo_metrics
    implemented_metrics = METRICS['NONPHYLO']['IMPL']

    if metric not in all_metrics:
        raise ValueError("Unknown metric: %s" % metric)

    if metric in implemented_metrics:
        # TODO: Make this method_name translation consistent
        func = ctx.get_action(
                'diversity_lib', local_method_names_dict()[metric])
    else:
        func = ctx.get_action('diversity_lib', 'beta_passthrough')
        func = partial(func, metric=metric, pseudocount=pseudocount)

    # TODO: test dispatch to skbio and local measures to ensure partial works
    result = func(table=table, n_jobs=n_jobs)
    return tuple(result)


def beta_phylogenetic_dispatch(ctx, table, phylogeny, metric, threads=1,
                               variance_adjusted=False, alpha=None,
                               bypass_tips=False):
    all_metrics = _all_phylo_metrics
    generalized_unifrac = 'generalized_unifrac'

    if metric not in all_metrics:
        raise ValueError("Unknown metric: %s" % metric)

    if alpha is not None and metric != generalized_unifrac:
        raise ValueError('The alpha parameter is only allowed when the choice'
                         ' of metric is generalized_unifrac')

    # HACK: this logic will be simpler once the remaining unifracs are done
    if metric in ('unweighted_unifrac', 'weighted_unifrac') \
            and not variance_adjusted:
        func = ctx.get_action('diversity_lib', metric)
    else:
        # handle unimplemented unifracs
        func = ctx.get_action('diversity_lib', 'unifrac_beta_dispatch')
        func = partial(func, metric=metric, alpha=alpha,
                       variance_adjusted=variance_adjusted)

    result = func(table, phylogeny, threads=threads,
                  bypass_tips=bypass_tips)
    return tuple(result)


@_disallow_empty_tables
@_validate_requested_cpus
def beta_passthrough(table: biom.Table, metric: str, pseudocount: int = 1,
                     n_jobs: int = 1) -> skbio.DistanceMatrix:
    def aitchison(x, y, **kwds):
        return euclidean(clr(x), clr(y))

    def canberra_adkins(x, y, **kwds):
        if (x < 0).any() or (y < 0).any():
            raise ValueError("Canberra-Adkins is only defined over positive "
                             "values.")

        nz = ((x > 0) | (y > 0))
        x_ = x[nz]
        y_ = y[nz]
        nnz = nz.sum()

        return (1. / nnz) * np.sum(np.abs(x_ - y_) / (x_ + y_))

    def jensen_shannon(x, y, **kwds):
        return jensenshannon(x, y)

    counts = table.matrix_data.toarray().T
    sample_ids = table.ids(axis='sample')
    if metric == 'aitchison':
        counts += pseudocount
        metric = aitchison
    elif metric == 'canberra_adkins':
        metric = canberra_adkins
    elif metric == 'jensenshannon':
        metric = jensen_shannon
    return skbio.diversity.beta_diversity(
            metric=metric, counts=counts, ids=sample_ids, validate=True,
            pairwise_func=sklearn.metrics.pairwise_distances, n_jobs=n_jobs)


@_disallow_empty_tables
@_validate_requested_cpus
def unifrac_beta_dispatch(table: BIOMV210Format, phylogeny: NewickFormat,
                          metric: str, threads: int = 1,
                          variance_adjusted: bool = False,
                          alpha: float = None, bypass_tips: bool = False
                          ) -> skbio.DistanceMatrix:
    # TODO: Should these checks be duplicated in this method and the "parent"
    # pipeline, in case users use unifrac_beta_dispatch directly?
    if metric not in _all_phylo_metrics:
        raise ValueError("Unknown metric: %s" % metric)

    if alpha is not None and metric != 'generalized_unifrac':
        raise ValueError('The alpha parameter is only allowed when the choice'
                         ' of metric is generalized_unifrac')

    # handle unimplemented unifracs
    if metric == 'generalized_unifrac':
        alpha = 1.0 if alpha is None else alpha
        func = partial(unimplemented_phylogenetic_metrics_dict()[metric],
                       alpha=alpha,
                       variance_adjusted=variance_adjusted)
    else:
        func = partial(unimplemented_phylogenetic_metrics_dict()[metric],
                       variance_adjusted=variance_adjusted)

    return func(table, phylogeny, threads=threads, bypass_tips=bypass_tips)


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
def weighted_unifrac(table: BIOMV210Format, phylogeny: NewickFormat,
                     threads: int = 1, bypass_tips: bool = False
                     ) -> skbio.DistanceMatrix:
    return unifrac.weighted_unnormalized(str(table), str(phylogeny),
                                         threads=threads,
                                         variance_adjusted=False,
                                         bypass_tips=bypass_tips)

# TODO: How do we feel about registered methods named with underscores:
#  e.g. _unifrac_beta_dispatch?
# By factoring this and beta_passthrough into methods, we expose additional
# user-facing methods (kinda gross) in order to clean up and make consistent
# the behavior of the dispatch pipelines. Among other things, this saves us
# from having to manually make skbio and unifrac results into Result objects.
# Pro: better modularity, consistency       Con: UI and Provenance clutter
