import numpy as np
import nengo

state = {"x": np.zeros(2)}


def input_fn(t):
    return state["x"]


kp, kd = 100.0, 2.0
r_p, r_d = 0.5, 5.0
net = nengo.Network(seed=0)
with net:
    inp = nengo.Node(input_fn, size_in=0, size_out=2)
    ens = nengo.Ensemble(200, 2, radius=1.0, neuron_type=nengo.LIF())
    out = nengo.Node(size_in=1)
    p = nengo.Probe(out, "output", synapse=None)
    nengo.Connection(inp, ens, synapse=None, transform=np.array([[1/r_p, 0], [0, 1/r_d]]))
    nengo.Connection(ens, out[0], transform=np.array([[kp*r_p, kd*r_d]]), synapse=0.005)

sim = nengo.Simulator(net, dt=0.002)
rng = np.random.default_rng(1)
pairs = []
for _ in range(400):
    e_p, e_d = rng.uniform(-0.3, 0.3), rng.uniform(-3, 3)
    state["x"] = np.array([e_p, e_d])
    sim.run_steps(20)
    got = sim.data[p][-1][0]
    want = kp * e_p + kd * e_d
    pairs.append((want, got))
pairs = np.array(pairs)
strong = np.abs(pairs[:, 0]) > 20.0
print("samples strong(>20):", int(strong.sum()))
sw = pairs[strong]
corr = np.corrcoef(sw[:, 0], sw[:, 1])[0, 1]
print("median ratio:", np.median(sw[:, 1] / sw[:, 0]), "corr:", round(corr, 3))
print("sample strong rows (want,got):")
for w, g in sw[:8]:
    print(f"  {w:8.1f}  {g:8.1f}  ratio {g/w:6.2f}")