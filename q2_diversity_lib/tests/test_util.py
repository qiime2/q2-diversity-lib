# ----------------------------------------------------------------------------
# Copyright (c) 2018-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin.testing import TestPluginBase
from .._util import _disallow_empty_tables, _safely_count_cpus
import biom
import numpy as np
import unittest.mock as mock
import psutil


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


class SafelyCountCPUSTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()

        @_safely_count_cpus
        def function_no_args():
            pass
        self.function_no_args = function_no_args

        @_safely_count_cpus
        def function_w_arg(system_cpus):
            return system_cpus
        self.cpu_counter_w_system_cpus_arg = function_w_arg

    def test_wrapped_function_does_not_take_a_cpu_count_arg(self):
        with self.assertRaisesRegex(AttributeError, 'without \'system_cpus'):
            self.function_no_args()

    @mock.patch("q2_diversity_lib._util.psutil.Process")
    def test_function_takes_a_cpu_count_arg(self, mock_cpu_affinity):
        mock_cpu_affinity = psutil.Process()
        mock_cpu_affinity.cpu_affinity = mock.MagicMock(return_value=[0, 1, 2])
        # Is the length of our mocked cpu_affinity 3?
        self.assertEqual(self.cpu_counter_w_system_cpus_arg(), 3)

    @mock.patch('psutil.Process.cpu_affinity', side_effect=AttributeError(
                "Here's your error"))
    @mock.patch('psutil.cpu_count', return_value=999)
    def test_system_has_no_cpu_affinity(self, mock_cpu_count, mock_cpu_affin):
        self.assertEqual(self.cpu_counter_w_system_cpus_arg(), 999)
