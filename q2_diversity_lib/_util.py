# ----------------------------------------------------------------------------
# Copyright (c) 2018-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np


def _drop_undefined_samples(counts: np.ndarray, sample_ids: np.ndarray,
                            minimum_nonzero_elements: int) -> (np.ndarray,
                                                               np.ndarray):
    nonzero_elements_per_sample = (counts != 0).sum(1)
    fancy_index = np.where(
            nonzero_elements_per_sample < minimum_nonzero_elements)
    filtered_counts = np.delete(counts, fancy_index, 0)
    filtered_sample_ids = np.delete(sample_ids, fancy_index)
    return (filtered_counts, filtered_sample_ids)
