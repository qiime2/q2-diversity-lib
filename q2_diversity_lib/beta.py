# ----------------------------------------------------------------------------
# Copyright (c) 2018-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import biom
import skbio.diversity
import sklearn.metrics
import unifrac


from q2_types.feature_table import BIOMV210Format
from q2_types.tree import NewickFormat
from ._util import (_disallow_empty_tables,
                    _safely_constrain_n_jobs)


# --------------------Non-Phylogenetic-----------------------
@_disallow_empty_tables
@_safely_constrain_n_jobs
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
@_safely_constrain_n_jobs
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
@_safely_constrain_n_jobs
def unweighted_unifrac(table: BIOMV210Format,
                       phylogeny: NewickFormat,
                       n_jobs: int = 1,
                       bypass_tips: bool = False) -> skbio.DistanceMatrix:
    f = unifrac.unweighted

    # TODO: is there ever a scenario in which an str will actually reach
    # this function? If not, torch this logic, and remove str handling from
    # _disallow_empty_tables
    if type(table) == str:
        table_fp = table
    elif type(table) == BIOMV210Format:
        table_fp = table.open().filename
    else:
        raise TypeError("Invalid table: must be BIOMV210Format or str.")

    return f(table_fp, str(phylogeny), threads=n_jobs,
             variance_adjusted=False, bypass_tips=bypass_tips)


@_disallow_empty_tables
@_safely_constrain_n_jobs
def weighted_unifrac(table: BIOMV210Format,
                     phylogeny: NewickFormat,
                     n_jobs: int = 1,
                     bypass_tips: bool = False) -> skbio.DistanceMatrix:
    f = unifrac.weighted_unnormalized

    # TODO: is there ever a scenario in which an str will actually reach
    # this function? If not, torch this logic, and remove str handling from
    # _disallow_empty_tables, and revert return from table_fp to table.open...
    if type(table) == str:
        table_fp = table
    elif type(table) == BIOMV210Format:
        table_fp = table.open().filename
    else:
        raise TypeError("Invalid table: must be BIOMV210Format or str.")

    return f(table_fp, str(phylogeny), threads=n_jobs,
             variance_adjusted=False, bypass_tips=bypass_tips)
