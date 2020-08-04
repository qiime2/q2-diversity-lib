# ----------------------------------------------------------------------------
# Copyright (c) 2018-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from inspect import signature
from functools import partial
import warnings

import pandas as pd
import skbio.diversity
import biom
import unifrac

from q2_types.feature_table import BIOMV210Format
from q2_types.tree import NewickFormat
from ._util import (_drop_undefined_samples,
                    _disallow_empty_tables)


METRICS = {
    'PHYLO': {
        'IMPL': {'faith_pd'},
        'UNIMPL': set()
    },
    'NONPHYLO': {
        'IMPL': {'observed_features', 'pielou_e', 'shannon'},
        'UNIMPL': {'ace', 'chao1', 'chao1_ci', 'berger_parker_d',
                   'brillouin_d', 'dominance', 'doubles', 'enspie', 'esty_ci',
                   'fisher_alpha', 'goods_coverage', 'heip_e',
                   'kempton_taylor_q', 'margalef', 'mcintosh_d', 'mcintosh_e',
                   'menhinick', 'michaelis_menten_fit', 'osd', 'robbins',
                   'simpson', 'simpson_e', 'singles', 'strong', 'gini_index',
                   'lladser_pe', 'lladser_ci'
                   }
    },
    'METRIC_NAME_TRANSLATIONS': {'shannon': 'shannon_entropy',
                                 'pielou_e': 'pielou_evenness'}
}


_all_phylo_metrics = METRICS['PHYLO']['IMPL'] | METRICS['PHYLO']['UNIMPL']
_all_nonphylo_metrics = METRICS['NONPHYLO']['IMPL'] \
                       | METRICS['NONPHYLO']['UNIMPL']


# --------------------- Method Dispatch --------------------------------------
# TODO: test drop_undefined_samples logic (including test for warning)
# TODO: smoke test to confirm l.86-88 doesn't blow up with an empty table
def alpha_rarefaction_dispatch(table: biom.Table, metric: str,
                               drop_undefined_samples: bool = False
                               ) -> pd.Series:
    # HACK - metrics overly permissive to squash test failures
    # TODO: revert this to just _all_nonphylo_metrics on transfer
    metrics = (_all_nonphylo_metrics |
               set(METRICS['METRIC_NAME_TRANSLATIONS'].values()))
    if metric not in metrics:
        raise ValueError("Unknown metric: %s" % metric)

    implemented_funcs = {'observed_features': observed_features,
                         'pielou_evenness': pielou_evenness,
                         'shannon_entropy': shannon_entropy}
    if metric in implemented_funcs:
        func = implemented_funcs[metric]
        if 'drop_undefined_samples' in signature(func).parameters:
            func = partial(func, table=table,
                           drop_undefined_samples=drop_undefined_samples)
        else:
            if drop_undefined_samples:
                warnings.warn(f"The {metric} metric does not support dropping "
                              "undefined samples.")
            func = partial(func, table=table)
    else:
        counts = table.matrix_data.toarray().astype(int).T
        sample_ids = table.ids(axis='sample')
        func = partial(skbio.diversity.alpha_diversity, metric=metric,
                       counts=counts, ids=sample_ids)

    result = func()
    result.name = metric
    return result


# TODO: smoke test empty table
def alpha_rarefaction_phylogenetic_dispatch(table: BIOMV210Format,
                                            phylogeny: NewickFormat,
                                            metric: str) -> pd.Series:
    metrics = _all_phylo_metrics
    if metric not in metrics:
        raise ValueError("Unknown phylogenetic metric: %s" % metric)

    # NOTE: This collection must contain a function for each alpha phylo metric
    alpha_phylo_functions = {'faith_pd': unifrac.faith_pd}
    func = alpha_phylo_functions[metric]
    result = func(str(table), str(phylogeny))
    result.name = metric
    return result


# --------------------- Phylogenetic -----------------------------------------
@_disallow_empty_tables
def faith_pd(table: BIOMV210Format, phylogeny: NewickFormat) -> pd.Series:
    table_str = str(table)
    tree_str = str(phylogeny)
    result = unifrac.faith_pd(table_str, tree_str)
    result.name = 'faith_pd'
    return result


# --------------------- Non-Phylogenetic -------------------------------------
@_disallow_empty_tables
def observed_features(table: biom.Table) -> pd.Series:
    presence_absence_table = table.pa(inplace=False)
    counts = presence_absence_table.matrix_data.toarray().astype(int).T
    sample_ids = presence_absence_table.ids(axis='sample')
    result = skbio.diversity.alpha_diversity(metric='observed_otus',
                                             counts=counts, ids=sample_ids)
    result.name = 'observed_features'
    return result


@_disallow_empty_tables
def pielou_evenness(table: biom.Table,
                    drop_undefined_samples: bool = False) -> pd.Series:
    counts = table.matrix_data.toarray().T
    sample_ids = table.ids(axis='sample')
    if drop_undefined_samples:
        counts, sample_ids = _drop_undefined_samples(
                counts, sample_ids, minimum_nonzero_elements=2)

    result = skbio.diversity.alpha_diversity(metric='pielou_e', counts=counts,
                                             ids=sample_ids)
    result.name = 'pielou_evenness'
    return result


@_disallow_empty_tables
def shannon_entropy(table: biom.Table,
                    drop_undefined_samples: bool = False) -> pd.Series:
    counts = table.matrix_data.toarray().T
    sample_ids = table.ids(axis='sample')
    if drop_undefined_samples:
        counts, sample_ids = _drop_undefined_samples(
                counts, sample_ids, minimum_nonzero_elements=1)
    result = skbio.diversity.alpha_diversity(metric='shannon', counts=counts,
                                             ids=sample_ids)
    result.name = 'shannon_entropy'
    return result


@_disallow_empty_tables
def alpha_passthrough(table: biom.Table, metric: str) -> pd.Series:
    counts = table.matrix_data.toarray().T
    sample_ids = table.ids(axis='sample')

    result = skbio.diversity.alpha_diversity(metric=metric, counts=counts,
                                             ids=sample_ids)
    result.name = metric
    return result
