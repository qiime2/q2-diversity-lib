# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import functools

import skbio.diversity
import skbio.diversity.alpha

import biom

from q2_types.feature_table import BIOMV210Format
from q2_types.sample_data import AlphaDiversityFormat
from q2_types.tree import NewickFormat

from ._util import (_validate_tables,
                    _validate_requested_cpus,
                    _omp_cmd_wrapper)

from q2_diversity_lib.skbio._methods import (_berger_parker, _brillouin_d,
                                             _simpsons_dominance, _esty_ci,
                                             _goods_coverage, _margalef,
                                             _mcintosh_d, _strong, _shannon,
                                             _p_evenness)

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

    # this is needed to prevent #SampleID from being retained as the
    # index name in the faith PD vector
    # (consistent with the other diversity outputs)
    df = pd.read_csv(str(vec), sep='\t')
    df.set_index('#SampleID', inplace=True)
    df.index.name = None
    df.to_csv(str(vec), sep='\t', header=True)

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
        # using in-house metrics temporarily
        # results.append(_skbio_alpha_diversity_from_1d(v, 'pielou_e'))
        v = np.reshape(v, (1, len(v)))
        results.extend([_p_evenness(c)for c in v])
    results = pd.Series(results, index=table.ids(), name='pielou_evenness')
    return results


@_validate_tables
def shannon_entropy(table: biom.Table,
                    drop_undefined_samples: bool = False) -> pd.Series:
    if drop_undefined_samples:
        table = table.remove_empty(inplace=False)

    results = []
    for v in table.iter_data(dense=True):
        # using in-house metrics temporarily
        # results.append(_skbio_alpha_diversity_from_1d(v, 'shannon'))
        v = np.reshape(v, (1, len(v)))
        results.extend([_shannon(c)for c in v])
    results = pd.Series(results, index=table.ids(), name='shannon_entropy')
    return results


@_validate_tables
def alpha_passthrough(table: biom.Table, metric: str) -> pd.Series:
    results = []
    method_map = {"berger_parker_d": _berger_parker,
                  "brillouin_d": _brillouin_d,
                  "simpson": _simpsons_dominance,
                  "esty_ci": _esty_ci,
                  "goods_coverage": _goods_coverage,
                  "margalef": _margalef,
                  "mcintosh_d": _mcintosh_d,
                  "strong": _strong}

    if metric in method_map:
        metric = functools.partial(method_map[metric])
        for v in table.iter_data(dense=True):
            v = np.reshape(v, (1, len(v)))
            results.extend([metric(c)for c in v])
    else:
        for v in table.iter_data(dense=True):
            results.append(_skbio_alpha_diversity_from_1d(v.astype(int),
                                                          metric))
    results = pd.Series(results, index=table.ids(), name=metric)
    return results
