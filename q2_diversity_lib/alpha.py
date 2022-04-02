# ----------------------------------------------------------------------------
# Copyright (c) 2018-2022, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import pandas as pd
import skbio.diversity
import biom
import unifrac
import numpy as np

from q2_types.feature_table import BIOMV210Format
from q2_types.tree import NewickFormat
from ._util import _disallow_empty_tables


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
    results = []
    for v in presence_absence_table.iter_data(dense=True):
        result = skbio.diversity.alpha_diversity(metric='observed_otus',
                                                 counts=v.astype(int),
                                                 ids=['placeholder', ])
        results.append(result.iloc[0])
    results = pd.Series(results, index=table.ids(), name='observed_features')
    return results


@_disallow_empty_tables
def pielou_evenness(table: biom.Table,
                    drop_undefined_samples: bool = False) -> pd.Series:
    if drop_undefined_samples:
        def transform_(v, i, m):
            if (v > 0).sum() < 2:
                return np.zeros(len(v))
            else:
                return v

        table = table.transform(transform_, inplace=False).remove_empty()

    results = []
    for v, i, m in table.iter(dense=True):
        result = skbio.diversity.alpha_diversity(metric='pielou_e',
                                                 counts=v,
                                                 ids=['placeholder', ])
        results.append(result.iloc[0])
    results = pd.Series(results, index=table.ids(), name='pielou_evenness')
    return results


@_disallow_empty_tables
def shannon_entropy(table: biom.Table,
                    drop_undefined_samples: bool = False) -> pd.Series:
    if drop_undefined_samples:
        table = table.remove_empty(inplace=False)

    results = []
    for v, i, m in table.iter(dense=True):
        result = skbio.diversity.alpha_diversity(metric='shannon',
                                                 counts=v,
                                                 ids=['placeholder', ])
        results.append(result.iloc[0])
    results = pd.Series(results, index=table.ids(), name='shannon_entropy')
    return results


@_disallow_empty_tables
def alpha_passthrough(table: biom.Table, metric: str) -> pd.Series:
    results = []
    for v, i, m in table.iter(dense=True):
        result = skbio.diversity.alpha_diversity(metric=metric,
                                                 counts=v.astype(int),
                                                 ids=['placeholder', ])
        results.append(result.iloc[0])
    results = pd.Series(results, index=table.ids(), name=metric)
    return results
