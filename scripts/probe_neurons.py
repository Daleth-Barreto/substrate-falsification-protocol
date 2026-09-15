import sys
import numpy as np
import nengo
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from convert import load_policy_params, elu

data = np.load(Path(__file__).resolve().parents[1] / "exported" / "lstm_walk_data.npz")
h = data["h"][:200]
w1, b1, w2, b2 = load_policy_params()
t1 = elu(w1 @ h.T + b1[:, None]).T
z2_base = np.abs(t1).mean()


def build(nens):
    state = {"x": np.zeros(64)}

    def ifn(t):
        return state["x"]

    net = nengo.Network()
    with net:
        inp = nengo.Node(ifn, size_in=0, size_out=64)
        a1 = nengo.Ensemble(nens, 64, radius=1.1, neuron_type=nengo.LIF())
        z1 = nengo.Node(size_in=32)
        p = nengo.Probe(z1, "output", synapse=None)
        nengo.Connection(inp, a1, synapse=None)
        nengo.Connection(a1, z1, function=lambda x: elu(w1 @ x + b1), synapse=0.005)
    return nengo.Simulator(net, dt=0.002), p, state


for nens in (1500, 3000, 6000, 12000):
    sim, p, state = build(nens)
    e1 = []
    for i in range(len(h)):
        state["x"] = h[i]
        sim.run_steps(15)
        e1.append(sim.data[p][-1] - t1[i])
    e1 = np.array(e1)
    nrmse = np.linalg.norm(e1) / np.linalg.norm(t1)
    print(f"neurons={nens:5d}  NRMSE={nrmse:.3f}  mean|err|={np.abs(e1).mean():.4f}")
    sim.close()