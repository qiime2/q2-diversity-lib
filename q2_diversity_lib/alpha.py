# ----------------------------------------------------------------------------
# Copyright (c) 2018-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

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
                   'lladser_pe'
                   }
    },
    'NAME_TRANSLATIONS': {'faith_pd': 'faith_pd',
                          'shannon': 'shannon_entropy',
                          'pielou_e': 'pielou_evenness',
                          'observed_features': 'observed_features'
                          }
}


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
    # Note: some metrics require ints, but biom.Table seems to default to float
    # (e.g. ace, lladser_pe, michaelis_menten_fit)
    counts = table.matrix_data.astype(int).toarray().T
    sample_ids = table.ids(axis='sample')

    result = skbio.diversity.alpha_diversity(metric=metric, counts=counts,
                                             ids=sample_ids)
    result.name = metric
    return result
