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
import numpy as np

from q2_types.feature_table import BIOMV210Format
from q2_types.sample_data import AlphaDiversityFormat
from q2_types.tree import NewickFormat

from ._util import (_validate_tables,
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
@_validate_tables
@_validate_requested_cpus
def faith_pd(table: BIOMV210Format, phylogeny: NewickFormat,
             threads: int = 1) -> AlphaDiversityFormat:
    vec = AlphaDiversityFormat()
    cmd = ['faithpd', '-i', str(table), '-t', str(phylogeny), '-o', str(vec)]
    _omp_cmd_wrapper(threads, cmd)
    return vec


# --------------------- Non-Phylogenetic -------------------------------------
def _skbio_alpha_diversity_from_1d(v, metric):
    # alpha_diversity expects a 2d structure
    v = np.reshape(v, (1, len(v)))
    result = skbio.diversity.alpha_diversity(metric=metric,
                                             counts=v,
                                             ids=['placeholder', ],
                                             validate=False)
    return result.iloc[0]


@_validate_tables
def observed_features(table: biom.Table) -> pd.Series:
    presence_absence_table = table.pa(inplace=False)
    results = []
    for v in presence_absence_table.iter_data(dense=True):
        results.append(_skbio_alpha_diversity_from_1d(v.astype(int),
                                                      'observed_otus'))
    results = pd.Series(results, index=table.ids(), name='observed_features')
    return results


@_validate_tables
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
    for v in table.iter_data(dense=True):
        results.append(_skbio_alpha_diversity_from_1d(v, 'pielou_e'))
    results = pd.Series(results, index=table.ids(), name='pielou_evenness')
    return results


@_validate_tables
def shannon_entropy(table: biom.Table,
                    drop_undefined_samples: bool = False) -> pd.Series:
    if drop_undefined_samples:
        table = table.remove_empty(inplace=False)

    results = []
    for v in table.iter_data(dense=True):
        results.append(_skbio_alpha_diversity_from_1d(v, 'shannon'))
    results = pd.Series(results, index=table.ids(), name='shannon_entropy')
    return results


@_validate_tables
def alpha_passthrough(table: biom.Table, metric: str) -> pd.Series:
    results = []

    for v in table.iter_data(dense=True):
        results.append(_skbio_alpha_diversity_from_1d(v.astype(int), metric))
    results = pd.Series(results, index=table.ids(), name=metric)
    return results
