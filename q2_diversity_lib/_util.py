# ----------------------------------------------------------------------------
# Copyright (c) 2018-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from inspect import signature

import numpy as np
from decorator import decorator
import psutil
import biom

from q2_types.feature_table import BIOMV210Format


def _drop_undefined_samples(counts: np.ndarray, sample_ids: np.ndarray,
                            minimum_nonzero_elements: int) -> (np.ndarray,
                                                               np.ndarray):
    nonzero_elements_per_sample = (counts != 0).sum(axis=1)
    fancy_index = np.where(
            nonzero_elements_per_sample < minimum_nonzero_elements)
    filtered_counts = np.delete(counts, fancy_index, axis=0)
    filtered_sample_ids = np.delete(sample_ids, fancy_index)
    return (filtered_counts, filtered_sample_ids)


@decorator
def _disallow_empty_tables(wrapped_function, *args, **kwargs):
    bound_arguments = signature(wrapped_function).bind(*args, **kwargs)
    table = bound_arguments.arguments.get('table')
    if table is None:
        table = bound_arguments.arguments.get('tables')

    if table is None:
        raise TypeError("The wrapped function is missing argument 'table' or "
                        "'tables'")

    if not isinstance(table, (tuple, list, set)):
        table = [table]

    for tab in table:
        if isinstance(tab, BIOMV210Format):
            tab = str(tab)
            tab_obj = biom.load_table(tab)
        elif isinstance(tab, biom.Table):
            tab_obj = tab
        else:
            raise ValueError("Invalid view type: table passed as "
                             f"{type(tab)}")

        if tab_obj.is_empty():
            raise ValueError("The provided table is empty")

    return wrapped_function(*args, **kwargs)


@decorator
def _validate_requested_cpus(wrapped_function, *args, **kwargs):
    bound_arguments = signature(wrapped_function).bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    b_a_arguments = bound_arguments.arguments

    if 'n_jobs' in b_a_arguments and 'threads' in b_a_arguments:
        raise TypeError("Duplicate parameters: The _validate_requested_cpus "
                        "decorator may not be applied to callables with both "
                        "'n_jobs' and 'threads' parameters. Do you really need"
                        " both?")
    elif 'n_jobs' in b_a_arguments:
        param_name = 'n_jobs'
    elif 'threads' in b_a_arguments:
        param_name = 'threads'
    else:
        raise TypeError("The _validate_requested_cpus decorator may not be"
                        " applied to callables without an 'n_jobs' or "
                        "'threads' parameter.")

    # If `Process.cpu_affinity` unavailable on system, fall back
    # https://psutil.readthedocs.io/en/latest/index.html#psutil.cpu_count
    try:
        cpus_available = len(psutil.Process().cpu_affinity())
    except AttributeError:
        cpus_available = psutil.cpu_count(logical=False)

    cpus_requested = b_a_arguments[param_name]

    if cpus_requested == 'auto':
        # mutate bound_arguments.arguments 'auto' to the requested # of cpus...
        b_a_arguments[param_name] = cpus_available
        # ...and update cpus requested to prevent TypeError
        cpus_requested = cpus_available

    if cpus_requested > cpus_available:
        raise ValueError(f"The value passed to '{param_name}' cannot exceed "
                         f"the number of processors ({cpus_available}) "
                         "available to the system.")

    return wrapped_function(*bound_arguments.args, **bound_arguments.kwargs)
