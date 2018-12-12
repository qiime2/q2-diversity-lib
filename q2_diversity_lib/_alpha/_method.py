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


def faith_pd(table: biom.Table, phylogeny: skbio.TreeNode) -> pd.Series:
    if table.is_empty():
        raise ValueError("The provided table object is empty")

    counts = table.matrix_data.toarray().astype(int).T
    sample_ids = table.ids(axis='sample')
    feature_ids = table.ids(axis='observation')

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
