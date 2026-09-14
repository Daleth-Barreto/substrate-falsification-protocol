import numpy as np
import nengo


def run(fn, x, n=600, radius=2.0, neurons=1000, syn=0.03, synout=0.05, dt=0.01):
    net = nengo.Network(seed=1)
    with net:
        inp = nengo.Node(lambda t: holder["x"], size_in=0, size_out=len(x))
        ens = nengo.Ensemble(neurons, len(x), radius=radius, neuron_type=nengo.LIF())
        out = nengo.Node(size_in=len(fn(x)))
        p = nengo.Probe(out, "output", synapse=None)
        nengo.Connection(inp, ens, synapse=syn)
        nengo.Connection(ens, out, function=fn, synapse=synout)
    sim = nengo.Simulator(net, dt=dt)
    holder = {"x": np.asarray(x, float)}
    sim.run_steps(n)
    v = sim.data[p][-1]
    sim.close()
    return v


holder = {"x": None}

# identity on dim3: feed x3=2.0
v = run(lambda x: [x[3]], [0.2, 0.1, 0.0, 2.0, 1.0])
print("identity dim3 x=2 ->", np.round(v, 3))

v = run(lambda x: [x[3]], [0.2, 0.1, 0.0, 1.0, 1.0])
print("identity dim3 x=1 ->", np.round(v, 3))

# constant 0.5 regardless
v = run(lambda x: [0.5], [0.2, 0.1, 0.0, 1.0, 1.0])
print("const 0.5 ->", np.round(v, 3))

v = run(lambda x: [0.5], [0.2, 0.1, 0.0, 5.0, 1.0])
print("const 0.5 nspk5 ->", np.round(v, 3))


def passthrough(x):
    net = nengo.Network(seed=1)
    with net:
        inp = nengo.Node(lambda t: holder["x"], size_in=0, size_out=5)
        out = nengo.Node(size_in=1)
        p = nengo.Probe(out, "output", synapse=None)
        nengo.Connection(inp[x], out, synapse=0.05)
    sim = nengo.Simulator(net, dt=0.01)
    holder["x"] = np.array([0.2, 0.1, 0.0, 2.0, 1.0])
    sim.run_steps(300)
    v = float(sim.data[p][-1][0])
    sim.close()
    return v


def static_identity():
    net = nengo.Network(seed=1)
    with net:
        inp = nengo.Node(np.array([0.2, 0.1, 0.0, 2.0, 1.0]))
        ens = nengo.Ensemble(1000, 5, radius=2.0, neuron_type=nengo.LIF())
        out = nengo.Node(size_in=1)
        p = nengo.Probe(out, "output", synapse=None)
        nengo.Connection(inp, ens, synapse=0.03)
        nengo.Connection(ens, out, function=lambda x: [x[3]], synapse=0.05)
    sim = nengo.Simulator(net, dt=0.01)
    sim.run_steps(600)
    v = float(sim.data[p][-1][0])
    sim.close()
    return v


print("passthrough dim3 x=2 ->", np.round(passthrough(3), 3))
print("static node->ens identity dim3 x=2 ->", np.round(static_identity(), 3))

import inspect

def probe(**kw):
    net = nengo.Network(seed=1)
    with net:
        inp = nengo.Node(np.array([0.2, 0.1, 0.0, 2.0, 1.0]))
        ens = nengo.Ensemble(kw.get("neurons", 1000), 5, radius=kw.get("radius", 2.0),
                             neuron_type=nengo.LIF())
        out = nengo.Node(size_in=1)
        p = nengo.Probe(out, "output", synapse=None)
        nengo.Connection(inp, ens, synapse=0.03)
        nengo.Connection(ens, out, function=lambda x: [x[3]], synapse=0.05)
    sim = nengo.Simulator(net, dt=kw.get("dt", 0.01), seed=kw.get("seed", 1))
    sim.run_steps(kw.get("n", 2000))
    v = float(sim.data[p][-1][0])
    sim.close()
    return v

print("dt=0.001 ->", round(probe(dt=0.001, n=2000), 3))
print("neurons=4000 ->", round(probe(neurons=4000), 3))
print("radius=10 ->", round(probe(radius=10.0), 3))
print("dt=0.004 n=20000 ->", round(probe(dt=0.004, n=20000), 3))