"""Information metrics (mini analogues of Gate A) on the command series.

- MI(x; u): mutual information between decoded command x and task u, binned,
  bias-corrected by permutation of u.
- TE(s -> x): transfer entropy from the channel-62 style binary event series s
  to the quantized command x, conditioning on the past of x, with a
  permutation null and a two-sided p-value.
"""

import numpy as np

RNG_SEED = 2026


def _discretize(x, n):
    lo, hi = np.percentile(x, [2.0, 98.0])
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    return np.clip(((x - lo) / (hi - lo) * n).astype(int), 0, n - 1)


def _mi_counts(a, b, n_a, n_b):
    h = np.zeros((n_a, n_b))
    for i in range(len(a)):
        h[a[i], b[i]] += 1
    h = h / h.sum()
    pa = h.sum(axis=1)
    pb = h.sum(axis=0)
    mi = 0.0
    for i in range(n_a):
        for j in range(n_b):
            if h[i, j] > 0 and pa[i] > 0 and pb[j] > 0:
                mi += h[i, j] * np.log(h[i, j] / (pa[i] * pb[j]))
    return mi


def mi_bias_corrected(x, u, n_perm=100):
    """Bias-corrected MI(x;u) in nats and two-sided permutation p."""
    rng = np.random.default_rng(RNG_SEED)
    n_a = n_b = 12
    a = _discretize(x, n_a)
    b = _discretize(u, n_b)
    obs = _mi_counts(a, b, n_a, n_b)
    perm = np.empty(n_perm)
    for k in range(n_perm):
        rng.shuffle(b)
        perm[k] = _mi_counts(a, b, n_a, n_b)
    bs = max(obs - perm.mean(), 0.0)
    p_one = float((perm >= obs).mean())
    if p_one >= 1.0 and np.std(perm) < 1e-12:
        p_two = 1.0          # degenerate null: observed equals every permutation
    else:
        p_two = 2.0 * min(p_one, 1.0 - p_one)
    return bs, p_two


def _te_from_hist(hist):
    """TE(s -> x | x) from joint histogram (x', s, x): log[ p*p(x) / (p(s,x)*p(x',x)) ]."""
    p = hist / hist.sum()
    p_xt = p.sum(axis=(0, 1))       # p(x)
    p_sx = p.sum(axis=0)            # p(s, x)
    p_xx = p.sum(axis=1)            # p(x', x)
    te = 0.0
    for xp in range(p.shape[0]):
        for si in range(1, p.shape[1]):
            for xt in range(p.shape[2]):
                px = p[xp, si, xt]
                if px <= 0:
                    continue
                num = px * p_xt[xt]
                den = p_sx[si, xt] * p_xx[xp, xt]
                if num > 0 and den > 0:
                    te += px * np.log(num / den)
    return te


def te_bias_corrected(s, x, n_perm=100):
    """Bias-corrected TE(s -> x | past x) in nats and two-sided permutation p.

    s: binary event series (length m). x: continuous command (length m+1);
    x[t] conditions the past and x[t+1] is predicted.
    """
    rng = np.random.default_rng(RNG_SEED)
    n_q = 6
    xq = _discretize(x, n_q)
    m = len(xq) - 1
    assert len(s) == m, (len(s), m)

    def hist_of(sb):
        h = np.zeros((n_q, 2, n_q))
        for t in range(m):
            h[xq[t + 1], int(sb[t]), xq[t]] += 1
        return h

    obs = _te_from_hist(hist_of(s))
    perm = np.empty(n_perm)
    for k in range(n_perm):
        perm[k] = _te_from_hist(hist_of(rng.permutation(s).copy()))
    bs = max(obs - perm.mean(), 0.0)
    p_one = float((perm >= obs).mean())
    if p_one >= 1.0 and np.std(perm) < 1e-12:
        p_two = 1.0          # degenerate null: observed equals every permutation
    else:
        p_two = 2.0 * min(p_one, 1.0 - p_one)
    return bs, p_two