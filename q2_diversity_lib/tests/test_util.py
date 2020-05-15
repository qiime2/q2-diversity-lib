# ----------------------------------------------------------------------------
# Copyright (c) 2018-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import unittest.mock as mock

import numpy as np
import biom
import psutil

from qiime2.plugin.testing import TestPluginBase
from .._util import (_disallow_empty_tables,
                     _safely_constrain_n_jobs)
from .test_beta import phylogenetic_measures, nonphylogenetic_measures


class DisallowEmptyTablesTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        self.empty_table = biom.Table(np.array([]), [], [])
        # empty table generated from self.empty_table with biom v2.1.7
        self.empty_table_fp = self.get_data_path('empty_table.biom')
        self.valid_table_fp = self.get_data_path('crawford.biom')
        self.invalid_table_fp = 'invalid_path_name.baaad'
        self.invalid_view_type = 9999999

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

    def test_passed_invalid_file_path(self):
        with self.assertRaisesRegex(
                    ValueError, "file path.*invalid_path_name.baaad"):
            self.function_with_table_param(table=self.invalid_table_fp)

    def test_passed_invalid_view_type(self):
        with self.assertRaisesRegex(
                    ValueError, "Invalid view type"):
            self.function_with_table_param(table=self.invalid_view_type)


class SafelyConstrainNJobsTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()

        @_safely_constrain_n_jobs
        def function_no_params():
            pass
        self.function_no_params = function_no_params

        @_safely_constrain_n_jobs
        def function_w_param(n_jobs=3):
            return n_jobs
        self.function_w_n_jobs_param = function_w_param

        self.valid_table_fp = self.get_data_path('two_feature_table.biom')
        self.valid_tree_fp = self.get_data_path('three_feature.tree')
        self.valid_table = biom.load_table(self.valid_table_fp)

    def test_function_without_n_jobs_param(self):
        with self.assertRaisesRegex(TypeError, 'without \'n_jobs'):
            self.function_no_params()

    @mock.patch("q2_diversity_lib._util.psutil.Process")
    def test_function_with_an_n_jobs_param(self, mock_cpu_affinity):
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

    @mock.patch("q2_diversity_lib._util.psutil.Process")
    def test_n_jobs_passed_as_kwarg(self, mock_cpu_affinity):
        mock_cpu_affinity = psutil.Process()
        mock_cpu_affinity.cpu_affinity = mock.MagicMock(return_value=[0, 1, 2])
        self.assertEqual(self.function_w_n_jobs_param(n_jobs=3), 3)

    @mock.patch("q2_diversity_lib._util.psutil.Process")
    def test_n_jobs_passed_as_default(self, mock_cpu_affinity):
        mock_cpu_affinity = psutil.Process()
        mock_cpu_affinity.cpu_affinity = mock.MagicMock(return_value=[0, 1, 2])
        self.assertEqual(self.function_w_n_jobs_param(), 3)

    # This test confirms appropriate handling of dependency-specific behaviors,
    # so is coupled to methods from those dependencies
    def test_pass_n_jobs_edge_cases_phylogenetic(self):
        for measure in phylogenetic_measures:
            with self.assertRaisesRegex(ValueError, "0.*invalid arg.*n_jobs"):
                measure(table=self.valid_table_fp,
                        phylogeny=self.valid_tree_fp, n_jobs=0)
            with self.assertRaisesRegex(ValueError, "pos.*integer.*n_jobs"):
                measure(table=self.valid_table_fp,
                        phylogeny=self.valid_tree_fp, n_jobs=-1)
            with self.assertRaisesRegex(ValueError, "pos.*integer.*n_jobs"):
                measure(table=self.valid_table_fp,
                        phylogeny=self.valid_tree_fp, n_jobs=-2)

    # This test confirms appropriate handling of dependency-specific behaviors,
    # so is coupled to methods from those dependencies
    def test_pass_n_jobs_edge_cases_nonphylogenetic(self):
        for measure in nonphylogenetic_measures:
            with self.assertRaisesRegex(ValueError, "0.*invalid arg.*n_jobs"):
                measure(table=self.valid_table, n_jobs=0)
            with self.assertRaisesRegex(ValueError,
                                        "Invalid.*n_jobs.*avail.*requested"):
                measure(table=self.valid_table, n_jobs=-9999999)