import sys
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from deploy12 import Deploy12
from spiking_pid import SpikingPD

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
FIGURES = RESULTS / "figures"
VIDEOS = RESULTS / "videos"


def sprint_cmd(t):
    if t < 2.0:
        return [0.5, 0.0, 0.0]
    if t < 5.0:
        return [1.0, 0.0, 0.0]
    if t < 10.0:
        return [1.5, 0.0, 0.0]
    return [2.0, 0.0, 0.0]


def constant_cmd(vx=0.8):
    def f(t):
        return [vx, 0.0, 0.0]
    return f


def run_one(controller, duration, cmd_fn, seed=0, obs_noise=0.0, push=None,
            record_video=False, n_neur=80, spd_ctrl=5):
    dep = Deploy12()
    dep.reset()
    dep.cmd_fn = cmd_fn
    dep.obs_noise_std = obs_noise
    dep.rng = np.random.default_rng(seed)
    spd = None
    tau = np.zeros(12)
    steps = 0
    energy_tau2 = 0.0
    rec_t, rec_h, rec_v, rec_p, rec_tau = [], [], [], [], []
    frames = []
    t0 = time.time()
    fallen = False
    while dep.data.time < duration:
        if steps % dep.dec == 0:
            dep.refresh_cmd()
            dep.policy_step()
        if controller == "pd":
            tau = dep.analog_pd_tau()
        elif controller == "spd":
            if spd is None:
                spd = SpikingPD(12, dep.kps, dep.kds, neurons_per_joint=n_neur)
            if steps % (dep.dec * spd_ctrl) == 0:
                tau = spd.step(dep.target - dep.data.qpos[7:], 0.0 - dep.data.qvel[6:])
        tau = np.clip(tau, -150, 150)
        if push is not None:
            if 3.5 <= dep.data.time <= 3.5 + push[1]:
                dep.data.xfrc_applied[dep.model.body("pelvis").id, :3] = np.asarray(push[0])
            else:
                dep.data.xfrc_applied[dep.model.body("pelvis").id, :3] = 0.0
        dep.apply_ctrl(tau)
        energy_tau2 += float(np.sum(tau ** 2))
        steps += 1
        if steps % 25 == 0:
            q = dep.data.qpos[3:7]
            pitch = np.arctan2(2 * (q[3]*q[2] + q[0]*q[1]), 1 - 2*(q[1]*q[1] + q[2]*q[2]))
            rec_t.append(dep.data.time)
            rec_h.append(dep.data.qpos[2])
            rec_v.append(dep.data.qvel[0])
            rec_p.append(float(pitch))
            rec_tau.append(float(np.linalg.norm(tau)))
            if record_video and steps % 25 == 0:
                frames.append(dep.render_frame())
        if dep.data.qpos[2] < 0.35:
            fallen = True
            break
    wall = time.time() - t0
    res = {
        "controller": controller,
        "sim_time": float(dep.data.time),
        "wall_time": float(wall),
        "fallen": fallen,
        "mean_vx": float(np.mean(rec_v)) if rec_v else float("nan"),
        "max_vx": float(np.max(np.abs(rec_v))) if rec_v else float("nan"),
        "max_forward": float(np.max(rec_v)) if rec_v else float("nan"),
        "mean_h": float(np.mean(rec_h)) if rec_h else float("nan"),
        "min_h": float(np.min(rec_h)) if rec_h else float("nan"),
        "mean_|pitch|": float(np.mean(np.abs(rec_p))) if rec_p else float("nan"),
        "tau2": energy_tau2,
        "tau2_s": energy_tau2 / max(dep.data.time, 1e-9) if dep.data.time else 0.0,
        "distance": np.asarray(rec_v, dtype=float).cumsum()[-1] / 50.0 if rec_v else 0.0,
        "spikes": int(spd.spikes) if spd is not None else 0,
        "sps": float(spd.spikes_per_sec) if spd is not None else 0.0,
        "series": {"t": rec_t, "h": rec_h, "v": rec_v, "p": rec_p, "tau": rec_tau},
        "frames": frames,
    }
    dep.close()
    return res


def summarize(res):
    return {k: v for k, v in res.items() if k in
            ("controller", "sim_time", "wall_time", "fallen", "mean_vx", "max_vx",
             "mean_h", "min_h", "mean_|pitch|", "tau2", "tau2_s", "distance",
             "spikes", "sps", "series")}


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    VIDEOS.mkdir(exist_ok=True)
    out = {}

    print("== E1 sprint (PD vs SpikingPD) ==")
    e1 = {}
    for ctrl in ("pd", "spd"):
        res = run_one(ctrl, duration=18.0, cmd_fn=sprint_cmd, record_video=True)
        e1[ctrl] = summarize(res)
        if res["frames"]:
            import imageio as iio
            writer = iio.get_writer(str(VIDEOS / f"e1_{ctrl}.mp4"), fps=40)
            for f in res["frames"]:
                writer.append_data(f)
            writer.close()
        print(ctrl, e1[ctrl])
    out["e1"] = e1

    print("== E2 push (PD vs SpikingPD) ==")
    e2 = {}
    for ctrl in ("pd", "spd"):
        res = run_one(ctrl, duration=10.0, cmd_fn=constant_cmd(0.8), push=([50.0, 0.0, 0.0], 0.15))
        e2[ctrl] = summarize(res)
        print(ctrl, e2[ctrl])
    out["e2"] = e2

    print("== E3 sensor noise (PD vs SpikingPD) ==")
    e3 = {}
    for noise in (0.0, 0.05, 0.10):
        for ctrl in ("pd", "spd"):
            for seed in range(3):
                res = run_one(ctrl, duration=8.0, cmd_fn=constant_cmd(0.8),
                              obs_noise=noise, seed=seed)
                key = f"n{noise:.2f}_{ctrl}_s{seed}"
                e3[key] = summarize(res)
    out["e3"] = e3

    print("== E4 energy ==")
    spd = SpikingPD(12, np.array([100., 100., 100., 150., 40., 40.] * 2),
                    np.array([2., 2., 2., 4., 2., 2.] * 2), neurons_per_joint=80)
    e4 = {}
    for steps in (1000,):
        err_p = np.random.default_rng(1).uniform(-0.2, 0.2, 12)
        err_v = np.random.default_rng(2).uniform(-2.0, 2.0, 12)
        t0 = time.time()
        for _ in range(steps):
            spd.step(err_p, err_v)
        wall = time.time() - t0
        e4.setdefault("spd", {"steps": steps, "wall": wall,
                              "sps": spd.spikes_per_sec, "spikes": spd.spikes})
    e4["stats"] = e4["spd"]
    print(e4["stats"])
    out["e4"] = e4

    with open(RESULTS / "summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("saved:", str(RESULTS / "summary.json"))


if __name__ == "__main__":
    main()