"""Gate B decisive probe: does the substrate compute, holding wiring and decode fixed?

Three substrates share the SAME input projection, the SAME sparse recurrent
wiring and the SAME readout (channel -> thsig -> canonical decide()). Only the
neuron dynamics differ:

- LIF: leaky integrate-and-fire (numpy Euler, dt = 0.5 ms)
- IZH: Izhikevich regular-spiking population (numpy Euler, dt = 0.5 ms)
- POISSON: dead-substrate null whose per-neuron firing rates are matched to the
  LIF mean rates but are INDEPENDENT of the task input.

The canonical calibrated decision map is reused verbatim from the research line
(AGENTS.md):  vx = clamp((thsig - 0.0615) / 0.277, 0, 0.75) * gate * slow.
"""

import numpy as np
from numpy.random import default_rng

DT_MS = 0.5
STEPS = 4000              # 2 s of simulation
CW = 20                   # control window = 10 ms (rate bin)
N_WIN = STEPS // CW       # 200 control windows
N = 2000
P_RE = 0.01               # recurrent wiring density
GATE = 1.0
SLOW = 1.0
TH0, K = 0.0615, 0.277


def decide(thsig):
    """Canonical calibrated decision map (verbatim from AGENTS.md)."""
    return np.clip((np.asarray(thsig, dtype=float) - TH0) / K, 0.0, 0.75) * GATE * SLOW


def make_wiring(seed):
    """Input projection g (N,1), sparse recurrent w (N,N), readout wr ~ g.

    wr is aligned with the input projection so the readout channel reads the
    neurons modulated by the task (analogue of the real channel-62 decode); a
    zero-mean random readout would decorrelate every substrate identically.
    """
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


class Substrate:
    name = "abstract"

    def reset(self, seed):
        raise NotImplementedError

    def step(self, i_cur):
        raise NotImplementedError

    @property
    def fired(self):
        raise NotImplementedError


class LIF(Substrate):
    """Leaky integrate-and-fire. Spike at V >= 1, reset to 0, 2 ms refractory."""
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


class IZH(Substrate):
    """Izhikevich regular spiking (a=0.02, b=0.2, c=-65, d=8)."""
    name = "izh"

    def reset(self, seed):
        self.v = np.full(N, -65.0, dtype=np.float64)
        self.u = np.full(N, 0.2 * -65.0, dtype=np.float64)
        self._f = np.zeros(N, dtype=bool)

    def step(self, i_cur):
        v, u = self.v, self.u
        fired = np.zeros(N, dtype=bool)
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


class PoissonNull(Substrate):
    """Dead substrate: Poisson spikes with LIF-matched rates, input-independent."""
    name = "poisson"

    def __init__(self, rate):
        self.rate = np.asarray(rate, dtype=np.float64)  # per-neuron per-tick prob

    def reset(self, seed):
        self._rng = default_rng(seed)
        self._f = np.zeros(N, dtype=bool)

    def step(self, i_cur):
        self._f = self._rng.random(N) < self.rate

    @property
    def fired(self):
        return self._f


def run_profile(sub, wiring, u_win, base_i, drive_gain, rec_scale, k_ro):
    """Run one substrate on one task profile u_win (N_WIN,).

    Returns (thsig (N_WIN,), counts (N, N_WIN)). thsig is the readout channel
    scaled so the calibrated decide() lands in the canonical range.
    """
    g, w, wr = wiring
    counts = np.zeros((N, N_WIN), dtype=np.int32)
    for wi in range(N_WIN):
        rec = w @ counts[:, max(wi - 1, 0)]
        i0 = base_i + drive_gain * (g[:, 0] * u_win[wi]) + rec_scale * rec
        for _ in range(CW):
            sub.step(i0)
            counts[sub.fired, wi] += 1
    c0 = wr @ (counts.astype(np.float64) / CW)          # (N_WIN,) raw channel
    thsig = k_ro * c0
    return thsig, counts