# ----------------------------------------------------------------------------
# Copyright (c) 2018-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
from functools import wraps
from inspect import signature
import psutil


def _drop_undefined_samples(counts: np.ndarray, sample_ids: np.ndarray,
                            minimum_nonzero_elements: int) -> (np.ndarray,
                                                               np.ndarray):
    nonzero_elements_per_sample = (counts != 0).sum(1)
    fancy_index = np.where(
            nonzero_elements_per_sample < minimum_nonzero_elements)
    filtered_counts = np.delete(counts, fancy_index, 0)
    filtered_sample_ids = np.delete(sample_ids, fancy_index)
    return (filtered_counts, filtered_sample_ids)


def _disallow_empty_tables(some_function):
    @wraps(some_function)
    def wrapper(*args, **kwargs):
        try:
            bound_signature = signature(wrapper).bind(*args, **kwargs)
            table = bound_signature.arguments['table']
        except KeyError as ex:
            raise TypeError("The wrapped function has no parameter "
                            + str(ex) + ".")
        else:
            if table.is_empty():
                raise ValueError("The provided table object is empty")
        return some_function(*args, **kwargs)
    return wrapper


def _safely_count_cpus(some_function):
    @wraps(some_function)
    def wrapper(*args, **kwargs):
        # https://psutil.readthedocs.io/en/latest/index.html#psutil.cpu_count
        # `Process.cpu_affinity` may not be available on all systems. If not,
        # fall back to the original cpu counting mechanism.
        sig = signature(wrapper)
        try:
            cpus = len(psutil.Process().cpu_affinity())
            print(sig.parameters)
            if 'system_cpus' in sig.parameters:
                kwargs['system_cpus'] = cpus
            else:
                raise AttributeError("The _safely_count_cpus decorator may "
                                     "not be applied to callables without "
                                     "'system_cpus' parameter.")
        except AttributeError:
            cpus = psutil.cpu_count(logical=False)
            if 'system_cpus' in sig.parameters:
                kwargs['system_cpus'] = cpus
            else:
                raise AttributeError("The _safely_count_cpus decorator may "
                                     "not be applied to callables without "
                                     "'system_cpus' parameter.")
        return some_function(*args, **kwargs)
    return wrapper
