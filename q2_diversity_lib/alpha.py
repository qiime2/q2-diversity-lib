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


# ---------- Collections to simplify dispatch process ------------------------
# TODO: should this comment be dropped? Should I match it in beta.py?
# must contain an entry for every metric in phylogenetic_metrics
def implemented_phylogenetic_measures_dict():
    return {'faith_pd': unifrac.faith_pd}


def implemented_phylogenetic_measures():
    return set(implemented_phylogenetic_measures_dict())


# TODO: should any of these be _private?
# TODO: handle observed_features->observed_otus transition cleanly
# TODO: This should probably include a breaking change to the name in q2-div
def implemented_non_phylogenetic_measures_dict():
    return {'observed_otus': observed_features, 'pielou_e': pielou_evenness,
            'shannon': shannon_entropy}


def implemented_nonphylogenetic_measures():
    return set(implemented_non_phylogenetic_measures_dict())


def unimplemented_nonphylogenetic_measures():
    return {'ace', 'chao1', 'chao1_ci', 'berger_parker_d', 'brillouin_d',
            'dominance', 'doubles', 'enspie', 'esty_ci', 'fisher_alpha',
            'goods_coverage', 'heip_e', 'kempton_taylor_q', 'margalef',
            'mcintosh_d', 'mcintosh_e', 'menhinick', 'michaelis_menten_fit',
            'osd', 'robbins', 'simpson', 'simpson_e', 'singles', 'strong',
            'gini_index', 'lladser_pe', 'lladser_ci'}


def all_nonphylogenetic_measures():
    return implemented_nonphylogenetic_measures() | \
           unimplemented_nonphylogenetic_measures()


# --------------------- Method Dispatch --------------------------------------
@_disallow_empty_tables
def alpha_dispatch(table: biom.Table, metric: str) -> pd.Series:
    pass


@_disallow_empty_tables
def alpha_phylogenetic_dispatch(table: BIOMV210Format, phylogeny: NewickFormat,
                                metric: str) -> pd.Series:
    metrics = implemented_phylogenetic_measures()
    if metric not in metrics:
        raise ValueError("Unknown phylogenetic metric: %s" % metric)

    func = implemented_phylogenetic_measures_dict()[metric]
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
    presence_absence_table = table.pa()
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
