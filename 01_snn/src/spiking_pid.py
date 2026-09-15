import numpy as np
import nengo


class SpikingPD:
    def __init__(self, n_joints, kp, kd, neurons_per_joint=100, r_p=0.5, r_d=5.0,
                 tau_rc=0.02, tau_ref=0.002, dt=0.002, seed=0, synapse=0.005):
        self.n = int(n_joints)
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.r_p = r_p
        self.r_d = r_d
        per = int(neurons_per_joint)
        self.total_neurons = self.n * per
        self.dt = dt
        state = {"acc": 0.0, "x": np.zeros(2 * self.n)}
        n_input = 2 * self.n

        def input_fn(t):
            return state["x"]

        def counter(t, x):
            state["acc"] += float(np.sum(np.abs(x)))
            return 0.0

        self.net = nengo.Network(seed=seed)
        with self.net:
            self.inp = nengo.Node(input_fn, size_in=0, size_out=n_input)
            self.out = nengo.Node(size_in=self.n)
            self._out_probe = nengo.Probe(self.out, "output", synapse=None)
            counter_node = nengo.Node(counter, size_in=self.total_neurons, size_out=0)
            for i in range(self.n):
                ens = nengo.Ensemble(per, 2, radius=1.0,
                                     neuron_type=nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref),
                                     label=f"joint_{i}")
                nengo.Connection(self.inp[[2 * i, 2 * i + 1]], ens,
                                 transform=np.array([[1.0 / self.r_p, 0.0],
                                                     [0.0, 1.0 / self.r_d]]),
                                 synapse=None)
                nengo.Connection(ens, self.out[i],
                                 transform=np.array([[self.kp[i] * self.r_p,
                                                      self.kd[i] * self.r_d]]),
                                 synapse=synapse)
                nengo.Connection(ens.neurons, counter_node[i * per:(i + 1) * per], synapse=None)
        self._state = state
        self.sim = nengo.Simulator(self.net, dt=dt)

    def step(self, err_pos, err_vel):
        self._state["x"] = np.concatenate([np.asarray(err_pos, dtype=float),
                                           np.asarray(err_vel, dtype=float)])
        self.sim.run_steps(1)
        return np.asarray(self.sim.data[self._out_probe][-1], dtype=float).ravel()

    @property
    def spikes(self):
        return int(round(self._state["acc"] * self.dt))

    @property
    def spikes_per_sec(self):
        return self.spikes / (self.sim.n_steps * self.dt) if self.sim.n_steps else 0.0

    def calibrate(self, n_samples=400, rng=None, r_p=None, r_d=None, settle=20):
        rng = rng or np.random.default_rng(0)
        r_p = self.r_p if r_p is None else r_p
        r_d = self.r_d if r_d is None else r_d
        e_p = rng.uniform(-r_p, r_p, (n_samples, self.n))
        e_d = rng.uniform(-r_d, r_d, (n_samples, self.n))
        want = e_p * self.kp + e_d * self.kd
        got = np.zeros_like(want)
        for i in range(n_samples):
            self._state["x"] = np.concatenate([e_p[i], e_d[i]])
            self.sim.run_steps(settle)
            got[i] = np.asarray(self.sim.data[self._out_probe][-1], dtype=float).ravel()
        err = np.linalg.norm(got - want, axis=0)
        den = np.linalg.norm(want, axis=0)
        nrmse = np.where(den > 1e-8, err / np.maximum(den, 1e-8), 0.0)
        return {
            "nrmse": float(np.nanmean(nrmse)),
            "nrmse_max": float(np.nanmax(nrmse)),
            "rmse_abs": float(np.nanmean(err)),
            "signal_abs": float(np.nanmean(np.abs(want))),
        }