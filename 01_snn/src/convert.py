import numpy as np
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RLGYM = REPO_ROOT.parent / "third_party" / "unitree_rl_gym"


def load_policy_params():
    m = torch.jit.load(str(RLGYM / "deploy" / "pre_train" / "g1" / "motion.pt"),
                       map_location="cpu")
    mods = {name: mod for name, mod in m.named_modules()}
    lin1 = mods["actor.0"]
    lin2 = mods["actor.2"]
    w1 = lin1.weight.detach().numpy()
    b1 = lin1.bias.detach().numpy()
    w2 = lin2.weight.detach().numpy()
    b2 = lin2.bias.detach().numpy()
    return w1, b1, w2, b2


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1.0))


class MLPHead:
    def __init__(self):
        self.w1, self.b1, self.w2, self.b2 = load_policy_params()

    def act(self, h):
        h = np.asarray(h)
        z = self.w1 @ h.T
        a = elu(z + self.b1[:, None])
        return (self.w2 @ a + self.b2[:, None]).T


def build_nengo_mlp_head(neurons_l1=1200, neurons_l2=800, radius1=3.0, radius2=3.0):
    import nengo
    w1, b1, w2, b2 = load_policy_params()
    d_in, h1 = w1.shape[1], w1.shape[0]
    net = nengo.Network(seed=0)
    with net:
        inp = nengo.Node(size_in=d_in)
        a1 = nengo.Ensemble(neurons_l1, d_in, radius=radius1, neuron_type=nengo.LIF())
        a2 = nengo.Ensemble(neurons_l2, h1, radius=radius2, neuron_type=nengo.LIF())
        out = nengo.Node(size_in=w2.shape[0])
        p = nengo.Probe(out, "output", synapse=None)
        nengo.Connection(inp, a1, synapse=None)
        nengo.Connection(a1, a2, function=lambda x: elu(w1 @ x + b1), synapse=0.005)
        nengo.Connection(a2, out, transform=w2, synapse=0.005)
        nengo.Connection(inp, out, transform=np.zeros((w2.shape[0], d_in)), synapse=0.005)
    return net, inp, out, p


class SpikeMLPHead:
    def __init__(self, neurons_l1=1200, neurons_l2=800, radius1=3.0, radius2=3.0):
        self.net, self.inp, self.out, self.p = build_nengo_mlp_head(neurons_l1, neurons_l2,
                                                                    radius1, radius2)
        import nengo
        self.sim = nengo.Simulator(self.net, dt=0.002)
        self._state = {"x": np.zeros(self.inp.size_out)}

        def input_fn(t):
            return self._state["x"]

        self.inp_function = None
        with self.net:
            pass
        self._sim_fast = self.sim

    def act_batch(self, H):
        out = np.zeros((H.shape[0], 12))
        for i, h in enumerate(H):
            self._state["x"] = np.asarray(h, dtype=float)
            self.sim.run_steps(15)
            out[i] = self.sim.data[self.p][-1]
        return out

    def reset(self):
        self.sim.reset()


def build_esn_reader(neurons=1500, spectral_radius=0.6, input_scale=1.0, dt=0.002, seed=0):
    import nengo
    rng = np.random.default_rng(seed)
    d_obs = 47
    d_h = 64
    net = nengo.Network(seed=seed)
    with net:
        inp = nengo.Node(size_in=d_obs)
        reservoir = nengo.Ensemble(neurons, d_obs + d_h, radius=1.0,
                                   neuron_type=nengo.LIF(), seed=seed,
                                   encoders=rng.normal(0, 1, (neurons, d_obs + d_h)),
                                   intercepts=rng.uniform(-0.8, 0.8, neurons))
        rec = rng.normal(0, 1, (neurons, d_obs + d_h)).astype(np.float32)
        rec *= spectral_radius
        hout = nengo.Node(size_in=d_h)
        o = nengo.Node(output=lambda t: 0.0, size_out=d_obs)
        nengo.Connection(inp, reservoir, synapse=None)
        nengo.Connection(reservoir, reservoir, transform=rec, synapse=0.02)
        nengo.Connection(0.0, reservoir, synapse=None)
        p = nengo.Probe(reservoir.neurons, "output")
    return net, inp, hout, p


class ESNReader:
    def __init__(self, neurons=1500, spectral_radius=0.6, input_scale=1.0):
        import nengo
        rng = np.random.default_rng(0)
        d_obs, d_h = 47, 64
        self.n = neurons
        self.Win = rng.normal(0, 1, (neurons, d_obs)) * input_scale
        Wrec = rng.normal(0, 1, (neurons, neurons))
        Wrec *= spectral_radius / max(np.abs(np.linalg.eigvals(Wrec)).max(), 1e-8)
        self.Wrec = Wrec
        self.tau = 0.02
        self.dt = 0.002
        self.x = np.zeros((neurons, 47 + 64))
        self.state = np.zeros(neurons)

    def _step(self, obs, h_target):
        f = np.tanh(self.Win @ obs + self.Wrec @ self.state)
        self.state = self.state + (self.dt / self.tau) * (f - self.state)
        return self.state

    def fit(self, obs_seq, h_seq, alpha_reg=1e-5):
        X = []
        self.state = np.zeros(self.n)
        for o, h in zip(obs_seq, h_seq):
            X.append(self.state.copy())
            self._step(o, h)
        X = np.array(X)
        H = np.asarray(h_seq)
        G = X.T @ X
        D = np.linalg.lstsq(G + alpha_reg * np.eye(self.n) * G.max(), X.T @ H, rcond=None)[0]
        self.D = D
        return D, X, H

    def rollout(self, obs_seq):
        self.state = np.zeros(self.n)
        hs = []
        for o in obs_seq:
            hs.append(self.D.T @ self.state)
            self._step_obs_only(o)
        return np.array(hs)

    def _step_obs_only(self, obs):
        f = np.tanh(self.Win @ obs + self.Wrec @ self.state)
        self.state = self.state + (self.dt / self.tau) * (f - self.state)


class SpikeHeadManifold:
    """Two-stage spiking NEF head with decoders fitted on the real activity
    manifold (ESN-style readout), not on uniform eval points."""

    def __init__(self, neurons=2000, settle=8):
        import nengo
        w1, b1, w2, b2 = load_policy_params()
        d_in, h1 = w1.shape[1], w1.shape[0]
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2
        self.neurons, self.settle = neurons, settle
        rng = np.random.default_rng(0)
        self._enc1 = rng.normal(0, 1, (neurons, d_in))
        self._enc2 = rng.normal(0, 1, (neurons, h1))
        self._h = {"x": np.zeros(d_in)}
        self._z = {"x": np.zeros(h1)}

        net = nengo.Network(seed=0)
        with net:
            hin = nengo.Node(lambda t: self._h["x"], size_in=0, size_out=d_in)
            zin = nengo.Node(lambda t: self._z["x"], size_in=0, size_out=h1)
            a1 = nengo.Ensemble(neurons, d_in, radius=1.1, neuron_type=nengo.LIF(),
                                encoders=self._enc1)
            a2 = nengo.Ensemble(neurons, h1, radius=1.1, neuron_type=nengo.LIF(),
                                encoders=self._enc2)
            p1 = nengo.Probe(a1.neurons, "output")
            p2 = nengo.Probe(a2.neurons, "output")
            nengo.Connection(hin, a1, synapse=None)
            nengo.Connection(zin, a2, synapse=0.005)
        self.sim = nengo.Simulator(net, dt=0.002)
        self._p1, self._p2 = p1, p2
        self.d1 = self.d2 = None
        self.net = net

    def _collect(self, H, Z):
        A1, A2 = [], []
        self.sim.reset()
        for hh, zz in zip(H, Z):
            self._h["x"] = np.asarray(hh, float)
            self._z["x"] = np.asarray(zz, float)
            self.sim.run_steps(self.settle)
            A1.append(self.sim.data[self._p1][-1].copy())
            A2.append(self.sim.data[self._p2][-1].copy())
        return np.array(A1), np.array(A2)

    def fit(self, H, TZ1, TACT):
        import nengo
        A1, A2 = self._collect(H, TZ1)
        self.d1 = np.linalg.lstsq(A1, np.asarray(TZ1), rcond=None)[0].T
        if TACT is not None:
            self.d2 = np.linalg.lstsq(A2, np.asarray(TACT), rcond=None)[0].T
        net = nengo.Network(seed=0)
        d_in, h1 = self.w1.shape[1], self.w1.shape[0]
        with net:
            hin = nengo.Node(lambda t: self._h["x"], size_in=0, size_out=d_in)
            a1 = nengo.Ensemble(self.neurons, d_in, radius=1.1, neuron_type=nengo.LIF(),
                                encoders=self._enc1)
            a2 = nengo.Ensemble(self.neurons, h1, radius=1.1, neuron_type=nengo.LIF(),
                                encoders=self._enc2)
            out = nengo.Node(size_in=self.w2.shape[0])
            q = nengo.Probe(out, "output", synapse=None)
            e2 = nengo.utils.ensemble.get_activities if False else None
            nengo.Connection(hin, a1, synapse=None)
            nengo.Connection(a1.neurons, a2.neurons,
                             transform=a2.encoders @ self.d1, synapse=0.005)
            if self.d2 is not None:
                nengo.Connection(a2.neurons, out, transform=self.d2, synapse=0.005)
        self.sim.close()
        self.sim = nengo.Simulator(net, dt=0.002)
        self._q = q
        self.net = net

    def act_batch(self, H):
        out = np.zeros((H.shape[0], self.w2.shape[0]))
        self.sim.reset()
        for i, x in enumerate(H):
            self._h["x"] = np.asarray(x, float)
            self.sim.run_steps(self.settle)
            out[i] = self.sim.data[self._q][-1]
        return out

    def reset(self):
        self.sim.reset()


def main():
    data = np.load(REPO_ROOT / "exported" / "lstm_walk_data.npz")
    obs, h, c, act = data["obs"], data["h"], data["c"], data["act"]
    w1, b1, _, _ = load_policy_params()
    z1 = elu(w1 @ h.T + b1[:, None]).T
    r1 = float(np.abs(h).max())
    r2 = float(np.abs(z1).max())
    print(f"[scale] max|h|={r1:.3f} max|z1|={r2:.3f}")

    mh = MLPHead()
    pred = mh.act(h)
    nrmse = np.linalg.norm(pred - act) / np.linalg.norm(act)
    print(f"[MLP-ann] head fidelity NRMSE={nrmse:.4f}")

    np.random.seed(0)
    imp = SpikeMLPHead(neurons_l1=1500, neurons_l2=1000, radius1=r1 * 1.3, radius2=r2 * 1.3)
    pred_n = imp.act_batch(h)
    nrmse_n = np.linalg.norm(pred_n - act) / np.linalg.norm(act)
    print(f"[MLP-snn] nengo head fidelity NRMSE={nrmse_n:.4f}")
    print(f"[MLP-snn] mean|act|={np.mean(np.abs(act)):.3f} mean|err|={np.mean(np.abs(pred_n-act)):.3f}")

    split = int(len(obs) * 0.7)
    mh2 = SpikeHeadManifold(neurons=2000, settle=8)
    mh2.fit(h[:split], z1[:split], act[:split])
    pred_m = mh2.act_batch(h[split:])
    nrmse_m = np.linalg.norm(pred_m - act[split:]) / np.linalg.norm(act[split:])
    print(f"[MLP-manifold] head fidelity NRMSE={nrmse_m:.4f} mean|err|={np.mean(np.abs(pred_m-act[split:])):.3f}")

    split = int(len(obs) * 0.7)
    esn = ESNReader(neurons=1500, spectral_radius=0.7)
    D, Xtr, Htr = esn.fit(obs[:split], h[:split])
    hp = esn.rollout(obs[split:])
    hell = np.linalg.norm(hp - h[split:], axis=1) / (np.linalg.norm(h[split:], axis=1) + 1e-6)
    print(f"[esn] h rollout rel err median={np.median(hell):.3f} mean={np.mean(hell):.3f}")

    a_from_h = mh.act(hp)
    err = np.linalg.norm(a_from_h - act[split:]) / (np.linalg.norm(act[split:]) + 1e-9)
    print(f"[esn->head] action NRMSE={err:.3f}")


if __name__ == "__main__":
    main()