from qiime2.plugin.testing import TestPluginBase


class UsageExampleTests(TestPluginBase):
    package = 'q2_diversity_lib.tests'

    def test_usage_examples(self):
        self.execute_examples()
