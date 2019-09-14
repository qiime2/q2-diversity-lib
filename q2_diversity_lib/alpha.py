# ----------------------------------------------------------------------------
# Copyright (c) 2018-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import biom
import pandas as pd
import skbio.diversity

from ._util import _drop_undefined_samples, _disallow_empty_tables


@_disallow_empty_tables
def faith_pd(table: biom.Table, phylogeny: skbio.TreeNode) -> pd.Series:
    presence_absence_table = table.pa()
    counts = presence_absence_table.matrix_data.toarray().astype(int).T
    sample_ids = presence_absence_table.ids(axis='sample')
    feature_ids = presence_absence_table.ids(axis='observation')

    try:
        result = skbio.diversity.alpha_diversity(metric='faith_pd',
                                                 counts=counts,
                                                 ids=sample_ids,
                                                 otu_ids=feature_ids,
                                                 tree=phylogeny)
    except skbio.tree.MissingNodeError as e:
        message = str(e).replace('otu_ids', 'feature_ids')
        message = message.replace('tree', 'phylogeny')
        raise skbio.tree.MissingNodeError(message)

    result.name = 'faith_pd'
    return result


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
