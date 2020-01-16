# ----------------------------------------------------------------------------
# Copyright (c) 2018-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from .alpha import (faith_pd, observed_features, pielou_evenness,
                    shannon_entropy)
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions


__all__ = ['faith_pd', 'observed_features', 'pielou_evenness',
           'shannon_entropy']
