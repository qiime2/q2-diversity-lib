# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from .skbio._methods import (_berger_parker, _brillouin_d, _simpsons_dominance,
                             _esty_ci, _goods_coverage, _margalef, _mcintosh_d,
                             _strong, _shannon, _p_evenness)
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

__all__ = ['_berger_parker', '_brillouin_d', '_simpsons_dominance',
           '_esty_ci', '_goods_coverage', '_margalef', '_mcintosh_d',
           '_strong', '_shannon', '_p_evenness']
