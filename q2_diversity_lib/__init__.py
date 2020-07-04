# ----------------------------------------------------------------------------
# Copyright (c) 2018-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from .alpha import (faith_pd, observed_features, pielou_evenness,
                    shannon_entropy, alpha_dispatch,
                    all_nonphylogenetic_measures)
from .beta import (bray_curtis, jaccard, unweighted_unifrac,
                   weighted_unnormalized_unifrac,
                   beta_dispatch, beta_phylogenetic_dispatch,
                   all_nonphylogenetic_metrics)
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

# TODO: sort out measures/metrics, and provide unique names for
# alpha. and beta.all_nonphylogenetic...
__all__ = ['faith_pd', 'observed_features', 'pielou_evenness',
           'shannon_entropy', 'bray_curtis', 'jaccard', 'unweighted_unifrac',
           'weighted_unnormalized_unifrac', 'alpha_dispatch', 'beta_dispatch',
           'beta_phylogenetic_dispatch', 'all_nonphylogenetic_measures',
           'all_nonphylogenetic_metrics']
