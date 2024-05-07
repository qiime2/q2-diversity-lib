import numpy as np

from skbio.diversity._util import _validate_counts_vector
import skbio.diversity.alpha

from scipy.special import gammaln


# c&p methods from skbio
def _berger_parker(counts):
    counts = _validate_counts_vector(counts)
    return counts.max() / counts.sum()


def _brillouin_d(counts):
    counts = _validate_counts_vector(counts)
    nz = counts[counts.nonzero()]
    n = nz.sum()
    return (gammaln(n + 1) - gammaln(nz + 1).sum()) / n


def _simpsons_dominance(counts):
    counts = _validate_counts_vector(counts)
    return 1 - skbio.diversity.alpha.dominance(counts)


def _esty_ci(counts):
    counts = _validate_counts_vector(counts)

    f1 = skbio.diversity.alpha.singles(counts)
    f2 = skbio.diversity.alpha.doubles(counts)
    n = counts.sum()
    z = 1.959963985
    W = (f1 * (n - f1) + 2 * n * f2) / (n ** 3)

    return f1 / n - z * np.sqrt(W), f1 / n + z * np.sqrt(W)


def _goods_coverage(counts):
    counts = _validate_counts_vector(counts)
    f1 = skbio.diversity.alpha.singles(counts)
    N = counts.sum()
    return 1 - (f1 / N)


def _margalef(counts):
    counts = _validate_counts_vector(counts)
    # replaced observed_otu call to sobs
    return (skbio.diversity.alpha.sobs(counts) - 1) / np.log(counts.sum())


def _mcintosh_d(counts):
    counts = _validate_counts_vector(counts)
    u = np.sqrt((counts * counts).sum())
    n = counts.sum()
    return (n - u) / (n - np.sqrt(n))


def _strong(counts):
    counts = _validate_counts_vector(counts)
    n = counts.sum()
    # replaced observed_otu call to sobs
    s = skbio.diversity.alpha.sobs(counts)
    i = np.arange(1, len(counts) + 1)
    sorted_sum = np.sort(counts)[::-1].cumsum()
    return (sorted_sum / n - (i / s)).max()


def _p_evenness(counts):
    counts = _validate_counts_vector(counts)
    return _shannon(counts, base=np.e) / np.log(
        skbio.diversity.alpha.sobs(counts=counts))


def _shannon(counts, base=2):
    counts = _validate_counts_vector(counts)
    freqs = counts / counts.sum()
    nonzero_freqs = freqs[freqs.nonzero()]
    return -(nonzero_freqs * np.log(nonzero_freqs)).sum() / np.log(base)
