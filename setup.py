# ----------------------------------------------------------------------------
# Copyright (c) 2018-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from setuptools import setup, find_packages
import versioneer

setup(
    name='q2-diversity-lib',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    author="Chris Keefe",
    author_email="crk239@nau.edu",
    description="Utility exposing diversity metrics/measures as Actions",
    entry_points={
        "qiime2.plugins":
        ["q2-diversity-lib=q2_diversity_lib.plugin_setup:plugin"]
    },
    url="https://qiime2.org",
    license='BSD-3-Clause',
    package_data={
        'q2_diversity_lib': ['citations.bib']
    },
    zip_safe=False,
)
