# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import pandas as pd
import skbio.diversity
import biom

from q2_types.feature_table import BIOMV210Format
from q2_types.sample_data import AlphaDiversityFormat
from q2_types.tree import NewickFormat
from ._util import (_drop_undefined_samples, _partition,
                    _disallow_empty_tables,
                    _validate_requested_cpus,
                    _omp_cmd_wrapper)


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
@_validate_requested_cpus
def faith_pd(table: BIOMV210Format, phylogeny: NewickFormat,
             threads: int = 1) -> AlphaDiversityFormat:
    vec = AlphaDiversityFormat()
    cmd = ['faithpd', '-i', str(table), '-t', str(phylogeny), '-o', str(vec)]
    _omp_cmd_wrapper(threads, cmd)
    return vec


# --------------------- Non-Phylogenetic -------------------------------------
@_disallow_empty_tables
def observed_features(table: biom.Table) -> pd.Series:
    presence_absence_table = table.pa(inplace=False)
    return pd.Series(presence_absence_table.sum('sample').astype(int),
                     index=table.ids(), name='observed_features')


@_disallow_empty_tables
def pielou_evenness(table: biom.Table,
                    drop_undefined_samples: bool = False) -> pd.Series:
    if drop_undefined_samples:
        table = _drop_undefined_samples(table, minimum_nonzero_elements=2)

    results = []
    for partition in _partition(table):
        counts = partition.matrix_data.T.toarray()
        sample_ids = partition.ids(axis='sample')
        results.append(skbio.diversity.alpha_diversity(metric='pielou_e',
                                                       counts=counts,
                                                       ids=sample_ids))
    result = pd.concat(results)
    result.name = 'pielou_evenness'
    return result


@_disallow_empty_tables
def shannon_entropy(table: biom.Table,
                    drop_undefined_samples: bool = False) -> pd.Series:
    if drop_undefined_samples:
        table = _drop_undefined_samples(table, minimum_nonzero_elements=1)

    results = []
    for partition in _partition(table):
        counts = partition.matrix_data.T.toarray()
        sample_ids = partition.ids(axis='sample')
        results.append(skbio.diversity.alpha_diversity(metric='shannon',
                                                       counts=counts,
                                                       ids=sample_ids))
    result = pd.concat(results)
    result.name = 'shannon_entropy'
    return result


@_disallow_empty_tables
def alpha_passthrough(table: biom.Table, metric: str) -> pd.Series:
    results = []
    for partition in _partition(table):
        counts = partition.matrix_data.astype(int).T.toarray()
        sample_ids = partition.ids(axis='sample')

        results.append(skbio.diversity.alpha_diversity(metric=metric,
                                                       counts=counts,
                                                       ids=sample_ids))
    result = pd.concat(results)
    result.name = metric
    return result
