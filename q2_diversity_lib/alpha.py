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
from unifrac import faith_pd as f_pd

from q2_types.feature_table import BIOMV210Format
from q2_types.tree import NewickFormat
from ._util import (_drop_undefined_samples,
                    _disallow_empty_tables)


@_disallow_empty_tables
def faith_pd(table: BIOMV210Format, phylogeny: NewickFormat) -> pd.Series:
    table_str = str(table)
    tree_str = str(phylogeny)
    result = f_pd(table_str, tree_str)
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
