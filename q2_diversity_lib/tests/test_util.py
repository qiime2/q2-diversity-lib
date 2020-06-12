# ----------------------------------------------------------------------------
# Copyright (c) 2018-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import mock

import numpy as np
import biom
import psutil

from qiime2 import Artifact
from qiime2.plugin.testing import TestPluginBase
from q2_types.feature_table import BIOMV210Format
from q2_types.tree import NewickFormat
from .._util import (_disallow_empty_tables,
                     _validate_requested_cpus)


class DisallowEmptyTablesTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()
        self.empty_table = biom.Table(np.array([]), [], [])
        # empty table generated from self.empty_table with biom v2.1.7
        empty_table_fp = self.get_data_path('empty_table.biom')
        self.empty_table_as_BIOMV210Format = BIOMV210Format(empty_table_fp,
                                                            mode='r')
        valid_table_fp = self.get_data_path('crawford.biom')
        self.valid_table_as_BIOMV210Format = BIOMV210Format(valid_table_fp,
                                                            mode='r')
        not_a_table_fp = self.get_data_path('crawford.nwk')
        self.invalid_view_type = NewickFormat(not_a_table_fp, mode='r')

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
            self.function_with_table_param(self.empty_table_as_BIOMV210Format)

    def test_pass_empty_table_as_kwarg(self):
        with self.assertRaisesRegex(ValueError, "table.*is empty"):
            self.function_with_table_param(
                table=self.empty_table_as_BIOMV210Format)

    def test_decorated_lambda_with_table_param(self):
        with self.assertRaisesRegex(ValueError, "table.*is empty"):
            decorated_lambda = _disallow_empty_tables(lambda table: None)
            decorated_lambda(self.empty_table_as_BIOMV210Format)

    def test_wrapped_function_has_no_table_param(self):
        with self.assertRaisesRegex(TypeError, "no parameter.*table"):
            self.function_without_table_param()

    def test_passed_invalid_view_type(self):
        with self.assertRaisesRegex(
                    ValueError, "Invalid view type.*Newick"):
            self.function_with_table_param(table=self.invalid_view_type)


class ValidateRequestedCPUsTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def setUp(self):
        super().setUp()

        @_validate_requested_cpus
        def function_no_params():
            pass
        self.function_no_params = function_no_params

        @_validate_requested_cpus
        def function_w_param(n_jobs=3):
            return n_jobs
        self.function_w_n_jobs_param = function_w_param

        @_validate_requested_cpus
        def function_w_threads(threads=2):
            return threads
        self.function_w_threads_param = function_w_threads

        @_validate_requested_cpus
        def function_w_duplicate_params(n_jobs=3, threads=2):
            pass
        self.function_w_both = function_w_duplicate_params

        self.jaccard_thru_framework = self.plugin.actions['jaccard']
        self.unweighted_unifrac_thru_framework = self.plugin.actions[
                    'unweighted_unifrac']

        two_feature_table_fp = self.get_data_path('two_feature_table.biom')
        self.two_feature_table = biom.load_table(two_feature_table_fp)
        self.two_feature_table_as_BIOMV210Format = BIOMV210Format(
                two_feature_table_fp, mode='r')
        self.two_feature_table_as_artifact = Artifact.import_data(
                    'FeatureTable[Frequency]', two_feature_table_fp)

        larger_table_fp = self.get_data_path('crawford.biom')
        self.larger_table_as_artifact = Artifact.import_data(
                'FeatureTable[Frequency]', larger_table_fp)

        valid_tree_fp = self.get_data_path('three_feature.tree')
        self.valid_tree_as_NewickFormat = NewickFormat(valid_tree_fp, mode='r')
        self.valid_tree_as_artifact = Artifact.import_data(
                'Phylogeny[Rooted]', valid_tree_fp)

        larger_tree_fp = self.get_data_path('crawford.nwk')
        self.larger_tree_as_artifact = Artifact.import_data(
                'Phylogeny[Rooted]', larger_tree_fp)

    def test_function_without_cpu_request_param(self):
        with self.assertRaisesRegex(TypeError, 'without.*n_jobs.*threads'):
            self.function_no_params()

    @mock.patch("q2_diversity_lib._util.psutil.Process")
    def test_function_with_appropriate_param(self, mock_process):
        mock_process = psutil.Process()
        mock_process.cpu_affinity = mock.MagicMock(return_value=[0, 1, 2])
        self.assertEqual(self.function_w_n_jobs_param(3), 3)

        mock_process.cpu_affinity = mock.MagicMock(return_value=[4, 19])
        self.assertEqual(self.function_w_threads_param(2), 2)

    def test_function_with_duplicate_cpu_allocation_params(self):
        with self.assertRaisesRegex(TypeError, 'Duplicate parameters'):
            self.function_w_both()

    @mock.patch("q2_diversity_lib._util.psutil.Process")
    @mock.patch('psutil.cpu_count', return_value=999)
    def test_system_has_no_cpu_affinity(self, mock_cpu_count, mock_process):
        mock_process = psutil.Process()
        mock_process.cpu_affinity = mock.MagicMock(side_effect=AttributeError)
        self.assertEqual(self.function_w_n_jobs_param(999), 999)
        assert mock_process.cpu_affinity.called

    @mock.patch("q2_diversity_lib._util.psutil.Process")
    def test_requested_more_than_system_cpus(self, mock_process):
        mock_process = psutil.Process()
        mock_process.cpu_affinity = mock.MagicMock(return_value=[0, 1, 2])
        with self.assertRaisesRegex(ValueError, "\'n_jobs\' cannot exceed"):
            self.function_w_n_jobs_param(999)
        with self.assertRaisesRegex(ValueError, "\'threads\' cannot exceed"):
            self.function_w_threads_param(999)

    @mock.patch("q2_diversity_lib._util.psutil.Process")
    def test_requested_cpus_passed_as_kwarg(self, mock_process):
        mock_process = psutil.Process()
        mock_process.cpu_affinity = mock.MagicMock(return_value=[0, 1, 2])
        self.assertEqual(self.function_w_n_jobs_param(n_jobs=3), 3)

    @mock.patch("q2_diversity_lib._util.psutil.Process")
    def test_n_jobs_passed_as_default(self, mock_process):
        mock_process = psutil.Process()
        mock_process.cpu_affinity = mock.MagicMock(return_value=[0, 1, 2])
        self.assertEqual(self.function_w_n_jobs_param(), 3)

    @mock.patch("q2_diversity_lib._util.psutil.Process")
    def test_auto_passed_to_cpu_request(self, mock_process):
        mock_process = psutil.Process()
        mock_process.cpu_affinity = mock.MagicMock(return_value=[0, 1, 2])
        self.assertEqual(self.function_w_n_jobs_param('auto'), 3)
        self.assertEqual(self.function_w_n_jobs_param(n_jobs='auto'), 3)
        self.assertEqual(self.function_w_threads_param('auto'), 3)
        self.assertEqual(self.function_w_threads_param(threads='auto'), 3)

    @mock.patch("q2_diversity_lib._util.psutil.Process")
    def test_cpu_request_through_framework(self, mock_process):
        mock_process = psutil.Process()
        mock_process.cpu_affinity = mock.MagicMock(return_value=[0])

        self.jaccard_thru_framework(self.larger_table_as_artifact, n_jobs=1)
        self.jaccard_thru_framework(self.larger_table_as_artifact,
                                    n_jobs='auto')
        self.unweighted_unifrac_thru_framework(self.larger_table_as_artifact,
                                               self.larger_tree_as_artifact,
                                               threads=1)
        self.unweighted_unifrac_thru_framework(self.larger_table_as_artifact,
                                               self.larger_tree_as_artifact,
                                               threads='auto')
        # If we get here, then it ran without error
        self.assertTrue(True)

    @mock.patch("q2_diversity_lib._util.psutil.Process")
    def test_more_threads_than_max_stripes(self, mock_process):
        mock_process = psutil.Process()
        mock_process.cpu_affinity = mock.MagicMock(return_value=[0])

        # The two_feature_table used here has only three samples, meaning
        # that it has a max of (3+1)/2 = 2 stripes. Unifrac may report
        # requests of more-threads-than-stripes to stderror, but should handle
        # that situation gracefully.
        self.unweighted_unifrac_thru_framework(
                self.two_feature_table_as_artifact,
                self.valid_tree_as_artifact, threads=1)
        self.unweighted_unifrac_thru_framework(
                self.two_feature_table_as_artifact,
                self.valid_tree_as_artifact, threads='auto')
        # If we get here, then it ran without error
        self.assertTrue(True)
