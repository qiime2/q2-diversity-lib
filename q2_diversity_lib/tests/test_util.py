# ----------------------------------------------------------------------------
# Copyright (c) 2018-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin.testing import TestPluginBase
from .._util import _disallow_empty_tables
import biom
import numpy as np


class DisallowEmptyTablesTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        self.empty_table = biom.Table(np.array([]), [], [])

        @_disallow_empty_tables
        def f1(table: biom.Table):
            pass
        self.function_with_table_param = f1

        @_disallow_empty_tables
        def f2():
            pass
        self.function_without_table_param = f2

    def test_pass_empty_table_positionally(self):
        with self.assertRaisesRegex(ValueError, "table.*is empty"):
            self.function_with_table_param(self.empty_table)

    def test_pass_empty_table_as_kwarg(self):
        with self.assertRaisesRegex(ValueError, "table.*is empty"):
            self.function_with_table_param(table=self.empty_table)

    def test_decorated_lambda_with_table_param(self):
        with self.assertRaisesRegex(ValueError, "table.*is empty"):
            decorated_lambda = _disallow_empty_tables(lambda table: None)
            decorated_lambda(self.empty_table)

    def test_wrapped_function_has_no_table_param(self):
        with self.assertRaisesRegex(TypeError, "no parameter.*table"):
            self.function_without_table_param()
