# ----------------------------------------------------------------------------
# Copyright (c) 2018-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase, main
from q2_diversity_lib.plugin_setup import plugin


class PluginTests(TestCase):
    def test_plugin(self):
        self.assertEqual(plugin.name, 'diversity-lib')


if __name__ == "__main__":
    main()
