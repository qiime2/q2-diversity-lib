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


def all_nonphylogenetic_measures_beta():
    return implemented_nonphylogenetic_metrics() | \
           unimplemented_nonphylogenetic_metrics()


def implemented_phylogenetic_metrics_dict():
    return {'unweighted_unifrac': unweighted_unifrac,
            'weighted_unifrac': weighted_unifrac}


# NOTE: a metric may be in both implemented and unimplemented collections,
# if it is only implemented with certain params (e.g. VA Weighted Unifrac is
# unimplemented, but uses the unifrac.weighted_unnormalized function)
def unimplemented_phylogenetic_metrics_dict():
    return {'unweighted_unifrac': unifrac.unweighted,
            'weighted_unifrac': unifrac.weighted_unnormalized,
            'weighted_normalized_unifrac': unifrac.weighted_normalized,
            'generalized_unifrac': unifrac.generalized}


def all_phylogenetic_measures_dict():
    return {**implemented_phylogenetic_metrics_dict(),
            **unimplemented_phylogenetic_metrics_dict()}


def all_phylogenetic_measures_beta():
    return set(all_phylogenetic_measures_dict())


def local_method_names_dict():
    return {'braycurtis': 'bray_curtis',
            'jaccard': 'jaccard',
            'unweighted_unifrac': 'unweighted_unifrac',
            'weighted_unifrac': 'weighted_unifrac'}


# -------------------- Method Dispatch -----------------------
def beta_dispatch(ctx, table, metric, pseudocount=1, n_jobs=1):
    all_metrics = all_nonphylogenetic_measures_beta()
    implemented_metrics = implemented_nonphylogenetic_metrics_dict()

    if metric not in all_metrics:
        raise ValueError("Unknown metric: %s" % metric)

    if metric in implemented_metrics:
        func = ctx.get_action(
                'diversity_lib', local_method_names_dict()[metric])
    else:
        func = ctx.get_action('diversity_lib', 'skbio_dispatch')
        func = partial(func, metric=metric, pseudocount=pseudocount)

    # TODO: test dispatch to skbio and local measures to ensure partial works
    result = func(table=table, n_jobs=n_jobs)
    return tuple(result)


@_disallow_empty_tables
@_validate_requested_cpus
def beta_phylogenetic_dispatch(ctx, table, phylogeny, metric, threads=1,
                               variance_adjusted=False, alpha=None,
                               bypass_tips=False):
    metrics = all_phylogenetic_measures_dict()
    generalized_unifrac = 'generalized_unifrac'

    if metric not in metrics:
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
        table = table.view(BIOMV210Format)
        phylogeny = phylogeny.view(NewickFormat)
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
    return tuple(result)


@_disallow_empty_tables
@_validate_requested_cpus
def skbio_dispatch(table: biom.Table, metric: str, pseudocount: int = 1,
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
    print(type(counts))
    sample_ids = table.ids(axis='sample')
    print(type(sample_ids))
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
