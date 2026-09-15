import numpy as np
import nengo

state = {"x": np.zeros(2)}


def input_fn(t):
    return state["x"]


def build(kp, kd, use_slice=True):
    r_p, r_d = 0.5, 5.0
    net = nengo.Network(seed=0)
    with net:
        inp = nengo.Node(input_fn, size_in=0, size_out=2)
        out = nengo.Node(size_in=1)
        p = nengo.Probe(out, "output", synapse=None)
        ens = nengo.Ensemble(150, 2, radius=1.0, neuron_type=nengo.LIF())
        nengo.Connection(inp if not use_slice else inp[[0, 1]], ens, synapse=None)
        nengo.Connection(ens, out if not use_slice else out[0],
                         transform=np.array([[kp * r_p, kd * r_d]]), synapse=0.005)
    sim = nengo.Simulator(net, dt=0.002)
    return sim, p


for kp, kd in [(10.0, 5.0), (100.0, 2.0), (150.0, 4.0)]:
    for use_slice in (False, True):
        sim, p = build(kp, kd, use_slice)
        rng = np.random.default_rng(1)
        errs = []
        for _ in range(80):
            e_p, e_d = rng.uniform(-0.3, 0.3), rng.uniform(-3, 3)
            state["x"] = np.array([e_p, e_d])
            sim.run_steps(30)
            got = sim.data[p][-1][0]
            want = kp * e_p + kd * e_d
            errs.append(abs(got - want) / (abs(want) + 1e-6))
        print(f"kp={kp:5.0f} kd={kd:4.0f} slice={int(use_slice)} meanNRMSE={np.mean(errs):.2f} med={np.median(errs):.2f}")