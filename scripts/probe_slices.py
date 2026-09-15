import numpy as np
import nengo

n = 3
kp = np.array([100.0, 150.0, 40.0])
kd = np.array([2.0, 4.0, 2.0])
r_p, r_d = 0.5, 5.0
state = {"x": np.zeros(2 * n)}


def input_fn(t):
    return state["x"]


net = nengo.Network(seed=0)
with net:
    inp = nengo.Node(input_fn, size_in=0, size_out=2 * n)
    out = nengo.Node(size_in=n)
    pout = nengo.Probe(out, "output", synapse=None)
    ens_list = []
    for i in range(n):
        ens = nengo.Ensemble(150, 2, radius=1.0, neuron_type=nengo.LIF())
        nengo.Connection(inp[[2 * i, 2 * i + 1]], ens,
                         transform=np.array([[1.0 / r_p, 0.0], [0.0, 1.0 / r_d]]),
                         synapse=None)
        nengo.Connection(ens, out[i],
                         transform=np.array([[kp[i] * r_p, kd[i] * r_d]]),
                         synapse=0.005)
        ens_list.append(ens)

sim = nengo.Simulator(net, dt=0.002)
print("ens radius:", [e.radius for e in ens_list])
rng = np.random.default_rng(3)
for _ in range(5):
    e_p = rng.uniform(-0.3, 0.3, n)
    e_d = rng.uniform(-3.0, 3.0, n)
    state["x"] = np.concatenate([e_p, e_d])
    sim.run_steps(40)
    got = sim.data[pout][-1]
    want = e_p * kp + e_d * kd
    nrmse = np.linalg.norm(got - want) / np.linalg.norm(want)
    print("want", np.round(want, 1), "got", np.round(got, 1), "nrmse %.2f" % nrmse)