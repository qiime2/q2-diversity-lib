# ----------------------------------------------------------------------------
# Copyright (c) 2018-2021, QIIME 2 development team.
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


def _drop_undefined_samples(table, minimum_nonzero_elements):
    def f(v, i, m):
        return (v > 0).sum() >= minimum_nonzero_elements
    return table.filter(f, inplace=False)


def _partition(table, block_size=100):
    number_of_splits = max(1, np.ceil(len(table.ids()) / block_size))
    splits = np.array_split(table.ids(), number_of_splits)
    split_map = {}
    for idx, split in enumerate(splits):
        for id_ in split:
            split_map[id_] = idx

    def part_f(i, m):
        return split_map[i]

    for _, block in table.partition(part_f):
        yield block


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
