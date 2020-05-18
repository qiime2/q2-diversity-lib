# ----------------------------------------------------------------------------
# Copyright (c) 2018-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from inspect import signature
import os

import numpy as np
from decorator import decorator
import psutil
from biom import Table as bTable
from biom import load_table

from q2_types.feature_table import BIOMV210Format

skbio_methods = ["bray_curtis", "jaccard"]
unifrac_methods = ["unweighted_unifrac", "weighted_unifrac",
                   "faith_pd"]


def _drop_undefined_samples(counts: np.ndarray, sample_ids: np.ndarray,
                            minimum_nonzero_elements: int) -> (np.ndarray,
                                                               np.ndarray):
    nonzero_elements_per_sample = (counts != 0).sum(1)
    fancy_index = np.where(
            nonzero_elements_per_sample < minimum_nonzero_elements)
    filtered_counts = np.delete(counts, fancy_index, 0)
    filtered_sample_ids = np.delete(sample_ids, fancy_index)
    return (filtered_counts, filtered_sample_ids)


@decorator
def _disallow_empty_tables(some_function, *args, **kwargs):
    bound_signature = signature(some_function).bind(*args, **kwargs)
    table = bound_signature.arguments.get('table')
    if table is None:
        raise TypeError("The wrapped function has no parameter 'table'")

# TODO: It is not possible for a string passed to the framework to make it here
# TODO: if table is an instance of BIOM, is it possible for it to have an
# invalid filepath?
    if isinstance(table, str) or isinstance(table, BIOMV210Format):
        table = str(table)
        if not os.path.exists(table):
            raise ValueError(f'Invalid file path: {table} does not exist')
        table_obj = load_table(table)
    elif isinstance(table, bTable):
        table_obj = table
    else:
        raise ValueError("Invalid view type: table passed as "
                         f"{type(table)}")

    if table_obj.is_empty():
        raise ValueError('The provided table is empty')

    return some_function(*args, **kwargs)


@decorator
def _safely_constrain_n_jobs(some_function, *args, **kwargs):
    # If `Process.cpu_affinity` unavailable on system, fall back
    # https://psutil.readthedocs.io/en/latest/index.html#psutil.cpu_count
    bound_signature = signature(some_function).bind(*args, **kwargs)
    bound_signature.apply_defaults()
    try:
        n_jobs = bound_signature.arguments['n_jobs']
    except KeyError:
        raise TypeError("The _safely_constrain_n_jobs decorator may not be"
                        " applied to callables without 'n_jobs' parameter")
    try:
        cpus = len(psutil.Process().cpu_affinity())
    except AttributeError:
        cpus = psutil.cpu_count(logical=False)
    if n_jobs > cpus:
        raise ValueError('The value of n_jobs cannot exceed the'
                         f' number of processors ({cpus}) available in'
                         ' this system.')

    # skbio and unifrac handle n_jobs args differently:
    if n_jobs == 0:
        raise ValueError("0 is an invalid argument for n_jobs")

    if some_function.__name__ in unifrac_methods and n_jobs <= 0:
        raise ValueError("Unifrac methods must be assigned a positive "
                         "integer value for n_jobs")

    if some_function.__name__ in skbio_methods and (n_jobs < 0)\
            and (cpus + n_jobs + 1) < 1:
        n_jobs_plus_one = n_jobs + 1
        cpus_requested = (cpus + n_jobs_plus_one)
        raise ValueError(f"Invalid argument to n_jobs: {cpus} cpus "
                         f"available, {cpus} - {-n_jobs_plus_one} = "
                         f"{cpus_requested} requested")

    return some_function(*args, **kwargs)
