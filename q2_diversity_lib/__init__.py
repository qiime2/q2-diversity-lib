# ----------------------------------------------------------------------------
# Copyright (c) 2018-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from .alpha import (faith_pd, observed_features, pielou_evenness,
                    shannon_entropy, alpha_passthrough,
                    alpha_phylogenetic_dispatch,
                    alpha_rarefaction_dispatch,
                    alpha_rarefaction_phylogenetic_dispatch)
from .beta import (bray_curtis, jaccard, unweighted_unifrac,
                   weighted_unifrac,
                   beta_dispatch, beta_phylogenetic_dispatch,
                   beta_passthrough, beta_phylogenetic_passthrough)
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

__all__ = ['faith_pd', 'observed_features', 'pielou_evenness',
           'shannon_entropy', 'alpha_passthrough', 'bray_curtis', 'jaccard',
           'unweighted_unifrac', 'weighted_unifrac',
           'alpha_phylogenetic_dispatch', 'alpha_rarefaction_dispatch',
           'alpha_rarefaction_phylogenetic_dispatch', 'beta_dispatch',
           'beta_phylogenetic_dispatch', 'beta_passthrough',
           'beta_phylogenetic_passthrough',
           ]
