import pandas as pd

from qiime2.plugin.testing import TestPluginBase
from qiime2.sdk.usage import DiagnosticUsage
from ..examples import observed_features_example

class UsageExampleTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def test_usage_examples(self):
        self.execute_examples()
