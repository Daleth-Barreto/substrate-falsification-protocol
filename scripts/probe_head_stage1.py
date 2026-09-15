import sys
import numpy as np
import nengo
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from convert import load_policy_params, elu

data = np.load(Path(__file__).resolve().parents[1] / "exported" / "lstm_walk_data.npz")
h = data["h"][:50]
w1, b1, w2, b2 = load_policy_params()
t1 = elu(w1 @ h.T + b1[:, None]).T
t2 = (w2 @ t1.T + b2[:, None]).T

state = {"x": np.zeros(64)}


def ifn(t):
    return state["x"]


net = nengo.Network()
with net:
    inp = nengo.Node(ifn, size_in=0, size_out=64)
    a1 = nengo.Ensemble(1500, 64, radius=0.83 * 1.3, neuron_type=nengo.LIF())
    z1 = nengo.Node(size_in=32)
    p1 = nengo.Probe(z1, "output", synapse=None)
    nengo.Connection(inp, a1, synapse=None)
    nengo.Connection(a1, z1, function=lambda x: elu(w1 @ x + b1), synapse=0.005)

sim = nengo.Simulator(net, dt=0.002)
e1 = []
for i in range(50):
    state["x"] = h[i]
    sim.run_steps(20)
    e1.append(sim.data[p1][-1] - t1[i])
e1 = np.array(e1)
print("stage1 z1: mean|t1|=%.3f mean|err|=%.3f per-dim err/std:" % (np.mean(np.abs(t1)), np.mean(np.abs(e1))))
print("   dim errs", np.round(np.mean(np.abs(e1), axis=0), 3))