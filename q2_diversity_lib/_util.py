# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from inspect import signature
from os import environ
import subprocess

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
def _validate_tables(wrapped_function, *args, **kwargs):
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

        if np.isnan(tab_obj.matrix_data.data).sum() > 0:
            raise ValueError("The provided table contains NaN")

        if (tab_obj.matrix_data.data < 0).sum() > 0:
            raise ValueError("The provided table contains negative values")

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


def _run_external_cmd(cmd, verbose=True, env=None):
    if verbose:
        print("Running external command line application. This may print"
              " messages to stdout and/or stderr.\nThe command being run is"
              " below. This command cannot be manually re-run as it will"
              " depend on temporary files that no longer exist.\n\nCommand:\n")
        print(" ".join(cmd), end='\n\n')
    return subprocess.run(cmd, check=True, env=env)


def _omp_cmd_wrapper(threads, cmd, verbose=True):
    env = environ.copy()
    env.update({'OMP_NUM_THREADS': str(threads)})
    return _run_external_cmd(cmd, verbose=verbose, env=env)
