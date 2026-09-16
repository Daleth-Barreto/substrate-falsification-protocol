"""Paired-null generators for Gate B2.

Each generator takes the REAL count matrix C (N x N_WIN) of a substrate run and
returns K surrogate count matrices.  Every surrogate is "paired": derived from
the exact run it is meant to falsify, never from a fresh simulation.  The three
constructions destroy different structure while preserving different marginals:

- circshift     : per-neuron circular shift with an independent random lag.
                  Preserves each neuron's exact count total, histogram and
                  autocorrelation (the sequence is rotated, not resampled).
                  Destroys cross-neuron task-locked alignment and phase.
- blockshuffle  : per-neuron, split the count sequence into B-window blocks and
                  shuffle the block order.  Preserves per-neuron marginals and
                  within-block short-range structure; destroys long-range
                  temporal and cross-neuron alignment.
- identityswap  : per-window random permutation of WHICH neuron produced each
                  count.  Preserves the per-window population count envelope
                  exactly (sum over N fixed per window, hence the task-driven
                  mean-activity envelope survives) and destroys the neuron-to-
                  readout alignment that wr ~ g exploits.

All three are strictly stronger nulls than Gate B's Poisson rate-match (which is
input-independent by construction).  If the substrate separates from ALL of
them, the task-carried information cannot be attributed to single-neuron
marginals, short-range dynamics, or the population firing envelope.
"""

import numpy as np
from numpy.random import default_rng

NULL_BASE = 4242
BLOCK_B = 5


def circshift(counts, k, seed):
    n, w = counts.shape
    rng = default_rng(NULL_BASE + seed)
    out = np.empty((k, n, w), dtype=counts.dtype)
    for i in range(k):
        lags = rng.integers(0, w, size=(n, 1))
        idx = (np.arange(w)[None, :] - lags) % w
        out[i] = counts[np.arange(n)[:, None], idx]
    return out


def blockshuffle(counts, k, seed):
    n, w = counts.shape
    b = BLOCK_B
    nblk = w // b
    rng = default_rng(NULL_BASE + 101 + seed)
    out = np.empty((k, n, nblk * b), dtype=counts.dtype)
    seq = counts[:, : nblk * b].reshape(n, nblk, b)
    for i in range(k):
        perm = np.empty((n, nblk), dtype=np.int64)
        for nn in range(n):
            perm[nn] = rng.permutation(nblk)
        out[i] = seq[np.arange(n)[:, None], perm].reshape(n, nblk * b)
    return out


def identityswap(counts, k, seed):
    n, w = counts.shape
    rng = default_rng(NULL_BASE + 202 + seed)
    out = np.empty((k, n, w), dtype=counts.dtype)
    for i in range(k):
        for ww in range(w):
            out[i, :, ww] = counts[rng.permutation(n), ww]
    return out