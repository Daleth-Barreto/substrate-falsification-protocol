import sys
import numpy as np
import nengo

state = {"x": np.array([0.0, 0.0])}


def input_fn(t):
    return state["x"]


net = nengo.Network(seed=0)
with net:
    inp = nengo.Node(input_fn, size_in=0, size_out=2)
    ens = nengo.Ensemble(200, 2, radius=1.0, neuron_type=nengo.LIF())
    out = nengo.Node(size_in=1)
    nengo.Connection(inp, ens, synapse=None)
    nengo.Connection(ens, out, synapse=0.005, transform=np.array([[10.0, 5.0]]))
    p = nengo.Probe(out, "output", synapse=None)

sim = nengo.Simulator(net, dt=0.002)
samples = [(0.1, -0.3), (0.5, 0.2), (-0.4, 0.6), (0.0, 0.0), (0.2, 0.1)]
for x, y in samples:
    state["x"] = np.array([x, y])
    sim.run_steps(30)
    got = sim.data[p][-1][0]
    want = 10 * x + 5 * y
    print(f"x=({x},{y}) want={want:7.2f} got={got:7.2f}")