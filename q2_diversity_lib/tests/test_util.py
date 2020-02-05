# ----------------------------------------------------------------------------
# Copyright (c) 2018-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin.testing import TestPluginBase
from .._util import (_disallow_empty_tables_passed_object,
                     _safely_constrain_n_jobs,
                     _disallow_empty_tables_passed_filepath)
import biom
import numpy as np
import unittest.mock as mock
import psutil


class DisallowEmptyTablesPassedObjectTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        self.empty_table = biom.Table(np.array([]), [], [])
        # empty table generated from self.empty_table with biom v2.1.7
        self.empty_table_fp = self.get_data_path('empty_table.biom')

        @_disallow_empty_tables_passed_object
        def f1(table: biom.Table):
            pass
        self.function_with_table_param = f1

        @_disallow_empty_tables_passed_object
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
            decorated_lambda = _disallow_empty_tables_passed_object(
                        lambda table: None)
            decorated_lambda(self.empty_table)

    def test_wrapped_function_has_no_table_param(self):
        with self.assertRaisesRegex(TypeError, "no parameter.*table"):
            self.function_without_table_param()

    def test_passed_filepath_not_table_object(self):
        with self.assertRaisesRegex(
                    AttributeError, "no attribute \'is_empty\'"):
            self.function_with_table_param(table=self.empty_table_fp)


class DisallowEmptyTablesPassedFilePathTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        self.empty_table = biom.Table(np.array([]), [], [])
        # empty table generated from self.empty_table with biom v2.1.7
        self.empty_table_fp = self.get_data_path('empty_table.biom')

        @_disallow_empty_tables_passed_filepath
        def f1(table: biom.Table):
            pass
        self.function_with_table_param = f1
        self.test_function = f1

        @_disallow_empty_tables_passed_filepath
        def f2():
            pass
        self.function_without_table_param = f2

    def test_pass_empty_table_positionally(self):
        with self.assertRaisesRegex(ValueError, "table.*is empty"):
            self.function_with_table_param(self.empty_table_fp)

    def test_pass_empty_table_as_kwarg(self):
        with self.assertRaisesRegex(ValueError, "table.*is empty"):
            self.function_with_table_param(table=self.empty_table_fp)

    def test_decorated_lambda_with_table_param(self):
        with self.assertRaisesRegex(ValueError, "table.*is empty"):
            decorated_lambda = _disallow_empty_tables_passed_filepath(
                        lambda table: None)
            decorated_lambda(self.empty_table_fp)

    def test_wrapped_function_has_no_table_param(self):
        with self.assertRaisesRegex(TypeError, "no parameter.*table"):
            self.function_without_table_param()

    def test_passed_object_not_filepath(self):
        with self.assertRaisesRegex(TypeError, "path should be string"):
            self.test_function(table=self.empty_table)


class SafelyCountCPUSTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()

        @_safely_constrain_n_jobs
        def function_no_params():
            pass
        self.function_no_params = function_no_params

        @_safely_constrain_n_jobs
        def function_w_param(n_jobs):
            return n_jobs
        self.function_w_n_jobs_param = function_w_param

    def smoke_test_function_has_no_n_jobs_param(self):
        with self.assertRaisesRegex(TypeError, 'without \'n_jobs'):
            self.function_no_params()

    @mock.patch("q2_diversity_lib._util.psutil.Process")
    def smoke_test_function_with_an_n_jobs_param(self, mock_cpu_affinity):
        mock_cpu_affinity = psutil.Process()
        mock_cpu_affinity.cpu_affinity = mock.MagicMock(return_value=[0, 1, 2])
        self.assertEqual(self.function_w_n_jobs_param(3), 3)

    @mock.patch('psutil.Process.cpu_affinity', side_effect=AttributeError)
    @mock.patch('psutil.cpu_count', return_value=999)
    def test_system_has_no_cpu_affinity(self, mock_cpu_count, mock_cpu_affin):
        self.assertEqual(self.function_w_n_jobs_param(999), 999)

    @mock.patch("q2_diversity_lib._util.psutil.Process")
    def test_n_jobs_greater_than_system_cpus(self, mock_cpu_affinity):
        mock_cpu_affinity = psutil.Process()
        mock_cpu_affinity.cpu_affinity = mock.MagicMock(return_value=[0, 1, 2])
        with self.assertRaisesRegex(ValueError, "n_jobs cannot exceed"):
            self.function_w_n_jobs_param(4)
