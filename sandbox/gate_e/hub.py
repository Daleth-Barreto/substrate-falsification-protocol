"""Gate D harness: reuses Gate B substrates/wiring; decoders are the variable.

Four decoders are compared on the SAME spike trains (counts per control
window), all under the canonical vref = 0.75*u contract:

- canon    : k_ro * (wr @ rates) -> decide()            (Gate B reference)
- ridge    : best L2-regularized LINEAR readout of the  spike counts
             (upper bound of what ANY fixed linear decode could extract),
             fit on train windows, evaluated on a held-out test split
- passthru : xhat = 0.75*u directly (no substrate at all). By construction
             RMSE = 0 here; it is the ceiling anchor for the open-loop mini
             and the honest statement that pure command tracking does not
             NEED a substrate in a clean feed-forward setting.

Recurrence contribution is isolated separately: "nohub" runs LIF/IZH with
rec_scale = 0 (wiring ignored), same canon readout.
"""

import numpy as np
from numpy.random import default_rng

DT_MS = 0.5
STEPS = 4000              # 2 s of simulation
CW = 20                   # control window = 10 ms (rate bin)
N_WIN = STEPS // CW       # 200 control windows
N = 2000
P_RE = 0.01
GATE = 1.0
SLOW = 1.0
TH0, K = 0.0615, 0.277


def decide(thsig):
    """Canonical calibrated decision map (verbatim)."""
    return np.clip((np.asarray(thsig, dtype=float) - TH0) / K, 0.0, 0.75) * GATE * SLOW


def make_wiring(seed):
    rng = default_rng(seed)
    g = rng.normal(0.0, 1.0, size=(N, 1)).astype(np.float64)
    nz = int(N * N * P_RE)
    rows = rng.integers(0, N, size=nz)
    cols = rng.integers(0, N, size=nz)
    vals = rng.normal(0.0, 1.0 / np.sqrt(N * P_RE), size=nz).astype(np.float64)
    w = np.zeros((N, N), dtype=np.float64)
    np.add.at(w, (rows, cols), vals)
    wr = g[:, 0].copy()
    return g, w, wr


class LIF:
    name = "lif"

    def reset(self, seed):
        self.v = np.zeros(N, dtype=np.float64)
        self._refr = np.zeros(N, dtype=int)
        self._f = np.zeros(N, dtype=bool)

    def step(self, i_cur):
        self._refr -= 1
        active = self._refr <= 0
        dv = (i_cur[active] - self.v[active]) / (20.0 / DT_MS)
        self.v[active] += dv
        self._f = np.zeros(N, dtype=bool)
        self._f[active] = self.v[active] >= 1.0
        self.v[self._f] = 0.0
        self._refr[self._f] = int(2.0 / DT_MS)

    @property
    def fired(self):
        return self._f


class IZH:
    name = "izh"

    def reset(self, seed):
        self.v = np.full(N, -65.0, dtype=np.float64)
        self.u = np.full(N, 0.2 * -65.0, dtype=np.float64)
        self._f = np.zeros(N, dtype=bool)

    def step(self, i_cur):
        v, u = self.v, self.u
        dv = 0.5 * DT_MS * (0.04 * v * v + 5.0 * v + 140.0 - u + i_cur)
        du = 0.5 * DT_MS * (0.02 * (0.2 * v - u))
        vn = v + dv
        fired = vn >= 30.0
        vn[fired] = 30.0
        u += du
        u[fired] += 8.0
        vn[fired] = -65.0
        self.v = vn
        self.u = u
        self._f = fired

    @property
    def fired(self):
        return self._f


class PoissonNull:
    name = "poisson"

    def __init__(self, rate):
        self.rate = np.asarray(rate, dtype=np.float64)

    def reset(self, seed):
        self._rng = default_rng(seed)
        self._f = np.zeros(N, dtype=bool)

    def step(self, i_cur):
        self._f = self._rng.random(N) < self.rate

    @property
    def fired(self):
        return self._f


def run_profile(sub, wiring, u_win, base_i, drive_gain, rec_scale, k_ro):
    """Same as Gate B. Returns (thsig, counts)."""
    g, w, wr = wiring
    counts = np.zeros((N, N_WIN), dtype=np.int32)
    use_rec = rec_scale != 0.0
    for wi in range(N_WIN):
        rec = w @ counts[:, max(wi - 1, 0)] if use_rec else 0.0
        i0 = base_i + drive_gain * (g[:, 0] * u_win[wi]) + rec_scale * rec
        for _ in range(CW):
            sub.step(i0)
            counts[sub.fired, wi] += 1
    c0 = wr @ (counts.astype(np.float64) / CW)
    thsig = k_ro * c0
    return thsig, counts


def ridge_readout(counts, vref, lam_grid, frac_test=0.3, tau=2):
    """Fit ridge linear readout x ~ [counts(w-t..w):1] -> vref(w).

    Windows are split by strict interleaving (even=train, odd=val) so train
    and test see the same profile phases (no extrapolation). Returns the held
    test predictor mean/std RMSE, the train RMSE (overfit marker), and the
    best lambda. counts: (N, W), vref: (W,).
    """
    W = counts.shape[1]
    n_keep = W - tau
    X = np.empty((n_keep, tau * N))
    y = np.empty(n_keep)
    for w in range(tau, W):
        X[w - tau] = counts[:, w - tau: w].ravel()
        y[w - tau] = vref[w]
    n_tr = int(n_keep * (1.0 - frac_test))
    # interleave: train = even indices, test = odd indices (same phase coverage)
    idx = np.arange(n_keep)
    tr, te = idx[::2][:n_tr], idx[1::2][:n_tr + (n_keep - len(idx[::2]))]
    te = te[:n_keep - len(tr)]
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    # standardize features to make lambda meaningful
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xts = (Xtr - mu) / sd
    Xvs = (Xte - mu) / sd
    ymu, ysd = ytr.mean(), ytr.std() + 1e-9
    ytc = (ytr - ymu) / ysd
    best = (None, np.inf)
    for lam in lam_grid:
        A = Xts.T @ Xts + lam * np.eye(Xts.shape[1])
        beta = np.linalg.solve(A, Xts.T @ ytc)
        pred = Xvs @ beta * ysd + ymu
        r = float(np.sqrt(np.mean((pred - yte) ** 2)))
        if r < best[1]:
            best = (lam, r)
    lam, rmse_te = best
    # train RMSE with the best lambda for the overfit marker
    A = Xts.T @ Xts + lam * np.eye(Xts.shape[1])
    beta = np.linalg.solve(A, Xts.T @ ytc)
    pred_tr = Xts @ beta * ysd + ymu
    rmse_tr = float(np.sqrt(np.mean((pred_tr - ytr) ** 2)))
    return {"rmse_te": rmse_te, "rmse_tr": rmse_tr, "lam": float(lam),
            "n_train": int(len(tr)), "n_test": int(len(te))}