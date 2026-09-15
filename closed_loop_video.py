"""closed_loop_video.py — capture the REAL closed-loop demo to MP4.

Faithful two-stage capture of the CL1-contract closed loop:

 Stage 1: runs the exact demo_walk-style loop (bridge_g1 data source + Nengo-LIF
          hub decide + stims back into the walker) but asks the data source to
          record (t, qpos) into a temp .npz under sandbox/ -- the physical run
          is therefore 100% the real closed-loop pipeline, with zero edits to
          anything under surrogate_cl/ (results/ there stay untouched).
 Stage 2: replays the recorded qpos with a fresh Deploy12 offscreen renderer,
          subsampled to 25 fps, and writes sandbox/video/g1_closed_loop.mp4
          (H.264 High + yuv420p) with a live telemetry overlay
          (vx, height, pitch, hub cmd, spike count).

Usage (project venv with the full stack: cl-sdk, nengo, mujoco, torch, yaml):
    ..\02_cl\.venv312\Scripts\python.exe closed_loop_video.py [duration_sec]
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import numpy as np

CUR = os.path.dirname(os.path.abspath(__file__))
SURROGATE_SRC = r"C:\Proyectos\papers\surrogate_cl\src"
sys.path.insert(0, SURROGATE_SRC)
sys.path.insert(0, os.path.join(CUR, "..", "01_snn", "src"))

import bridge_g1  # noqa: E402  (read-only import)
import cl  # noqa: E402
from cl.sim import (SimulatorDataSourceMetadata, set_simulator_data_source)  # noqa: E402
import nengo  # noqa: E402
import deploy12  # noqa: E402

OUTDIR = os.path.join(CUR, "video")
OUTMP4 = os.path.join(OUTDIR, "g1_closed_loop.mp4")
TMP_NPZ = os.path.join(OUTDIR, "closed_loop_rec.npz")

TPS = 40
BASE_VX = 0.5
HUB_DT = 0.002
HUB_HORIZON_STEPS = 600
TAU_SIGN = 1.0
TAU_UNITS = 0.75
FRAME_W, FRAME_H = 640, 480
VIDEO_HZ = 25
NEW_ENTROPY = os.pathsep.join(
    sys.path + [r"C:\Proyectos\papers\surrogate_cl\config"])


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def decide(x):
    roll, pitch, thsig, nspk, _ = x
    gate = _sig(30.0 * (nspk - 0.15))
    slow = 1.0 - 0.9 * _sig(10.0 * (max(abs(roll), abs(pitch)) - 0.9))
    vx = max(0.0, min(0.75, (thsig - 0.0615) / 0.277)) * gate * slow
    tau = TAU_UNITS * np.tanh(2.0 * TAU_SIGN * roll)
    return [float(vx), float(tau)]


def build_hub(holder):
    net = nengo.Network(seed=1)
    with net:
        inp = nengo.Node(lambda t: holder["last_x"], size_in=0, size_out=5)
        ensemble = nengo.Ensemble(1000, 5, radius=2.0, neuron_type=nengo.LIF())
        out = nengo.Node(size_in=2)
        p = nengo.Probe(out, "output", synapse=None)
        nengo.Connection(inp, ensemble, synapse=0.03)
        nengo.Connection(ensemble, out, function=decide, synapse=0.05)
    sim = nengo.Simulator(net, dt=HUB_DT)
    return inp, out, p, sim


class HubThread:
    def __init__(self, inp, out, p, sim):
        self.inp, self.out, self.p, self.sim = inp, out, p, sim
        self.latest = np.array([0.0, 0.0])
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self._stop = True
        if self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.sim.close()

    def _run(self):
        while not self._stop:
            self.sim.run_steps(HUB_HORIZON_STEPS)
            self.latest = np.asarray(self.sim.data[self.p][-1]).ravel()


def run_closed_loop(duration_sec):
    """Stage 1: the real loop, recording qpos to TMP_NPZ via record_path."""
    if os.path.exists(TMP_NPZ):
        os.remove(TMP_NPZ)
    set_simulator_data_source(
        "bridge_g1:create",
        config={"duration_sec": duration_sec, "cmd_vx": BASE_VX,
                "record_path": TMP_NPZ.replace("\\", "/")},
        metadata=SimulatorDataSourceMetadata(
            channel_count=bridge_g1.N_CHANNELS,
            frames_per_second=bridge_g1.SAMPLE_RATE,
            uV_per_sample_unit=0.195,
            duration_frames=None,
            seekable=False,
            realtime_only=True,
            supports_accelerated=False,
        ),
    )

    counts_holder = {"last_x": np.zeros(5), "lock": threading.Lock()}
    inp, out, p, sim = build_hub(counts_holder)
    hub = HubThread(inp, out, p, sim).start()

    log = {"t": [], "cmd": [], "nspk": []}
    applied = 0.0
    THR = -1400
    prev = np.zeros(bridge_g1.N_CHANNELS)
    with cl.open() as neurons:
        for tick in neurons.loop(TPS, stop_after_seconds=duration_sec + 0.5):
            frames = tick.frames.astype(np.float32) if tick.frames is not None \
                else np.zeros((0, bridge_g1.N_CHANNELS))
            below = frames < THR
            prev_on = (prev >= THR)[None, :]
            down_cross = below & np.vstack([prev_on, frames[:-1] >= THR])
            counts = down_cross.sum(axis=0).astype(float)
            prev = frames[-1] if len(frames) else prev
            nspk = float(counts.sum())
            x = np.array([
                float(min(30.0, counts[12]) / 8.0),
                float(min(30.0, counts[13]) / 8.0),
                0.0,
                float(min(30.0, counts.sum()) / 8.0),
                float(min(5.0, counts[63]) / 2.0),
            ])
            with counts_holder["lock"]:
                counts_holder["last_x"] = x
            cmd = float(max(0.0, min(2.2, hub.latest[0])))
            if abs(cmd - applied) > 0.1:
                applied = cmd
                if applied > 1e-3:
                    neurons.stim(bridge_g1.CH_VX_OVERRIDE,
                                 max(applied, 1e-3) * bridge_g1.STIM_UA_PER_MS)
            log["t"].append(float(tick.iteration) / TPS)
            log["cmd"].append(cmd)
            log["nspk"].append(nspk)
    hub.stop()
    return log


def replay_and_render(log, duration_sec):
    """Stage 2: offscreen render of the recorded qpos + telemetry overlay."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.animation import FFMpegWriter
    import matplotlib.pyplot as plt

    rec = np.load(TMP_NPZ)
    t_q = rec["t"]
    qpos = rec["qpos"]
    dt = float(rec["dt"])
    if t_q[-1] < 1.0:
        print("recording too short (fallen early?):", t_q[-1])
        return 1

    # The npz records every physics step (500 Hz at sim dt 2 ms). Sub-sample
    # to the video rate.
    step = max(1, int(round(1.0 / (dt * VIDEO_HZ))))

    dep = deploy12.Deploy12()
    dep.reset(cmd=[BASE_VX, 0.0, 0.0])

    # Build telemetry series aligned to recorded control times
    np_log_t = np.asarray(log["t"])
    cmd_s = np.asarray(log["cmd"])
    nspk_s = np.asarray(log["nspk"])

    def interp(xq, tv, xv):
        idx = np.clip(np.searchsorted(tv, xq), 0, len(tv) - 1)
        return xv[idx]

    # Subsampled frame times (video). Every physics ctrl step is 20 ms
    # (50 Hz); we take every 2nd -> 25 fps.
    step = max(1, int(round(1.0 / (dt * VIDEO_HZ))))
    idxs = list(range(0, len(t_q), step))
    print(f"recording n={len(t_q)} pts ({t_q[-1]:.2f}s, {1/dt:.0f} Hz) -> "
          f"{len(idxs)} frames at {VIDEO_HZ} fps")

    writer = FFMpegWriter(fps=VIDEO_HZ, bitrate=2500, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p",
                                      "-movflags", "faststart"])
    fig = plt.figure(figsize=(FRAME_W / 96, FRAME_H / 96))
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    im = ax.imshow(np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8))
    ttl = ax.annotate("", xy=(0.012, 0.97), xycoords="axes fraction",
                      fontsize=13, va="top", ha="left", color="white",
                      bbox=dict(boxstyle="round,pad=0.35",
                                fc="#00000099", ec="none"))

    def atten(q):
        qw, qx, qy, qz = q
        roll = float(np.arctan2(2 * (q[2] * q[3] + q[0] * q[1]),
                                1 - 2 * (q[1] * q[1] + q[2] * q[2])))
        pitch = float(np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]),
                                 1 - 2 * (q[1] * q[1] + q[2] * q[2])))
        return roll, pitch

    d = dep.data
    n_done = 0
    _ = dep.render_frame(width=FRAME_W, height=FRAME_H)  # init renderer
    frame = dep.render_frame(width=FRAME_W, height=FRAME_H)
    prev_frame = frame
    with writer.saving(fig, OUTMP4, dpi=96):
        for i in idxs:
            dep.data.qpos[:] = qpos[i]
            frame = dep.render_frame(width=FRAME_W, height=FRAME_H)
            t = float(t_q[i])
            vx = float(np.gradient(qpos[:, 0])[i] / dt)
            h = float(qpos[i, 2])
            roll, pitch = atten(qpos[i, 3:7])
            cmd = interp(t, np_log_t, cmd_s)
            nspk = interp(t, np_log_t, nspk_s)
            txt = (f"G1 closed loop  hub=Nengo-LIF(1000)  substrate=rate\n"
                   f"t={t:5.2f}s  vx={vx:+6.2f} m/s  h={h:5.2f} m  "
                   f"pitch={np.degrees(pitch):+.1f} deg\n"
                   f"hub cmd(vx)={cmd:5.2f} m/s  spikes/tick={nspk:5.0f}")
            im.set_data(frame)
            ttl.remove()
            ttl = ax.annotate(txt, xy=(0.012, 0.97), xycoords="axes fraction",
                              fontsize=13, va="top", ha="left", color="white",
                              bbox=dict(boxstyle="round,pad=0.35",
                                        fc="#00000099", ec="none"))
            writer.grab_frame()
            n_done += 1
    plt.close(fig)
    dep.close()
    print(f"WROTE {OUTMP4} ({n_done} frames)")
    return 0


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = set(a for a in argv[1:] if a.startswith("--"))
    duration_sec = float(args[0]) if args else 12.0
    os.makedirs(OUTDIR, exist_ok=True)
    if "--replay-only" not in flags:
        log = run_closed_loop(duration_sec)
        print("len(log.t)=%d mean_cmd=%.3f" % (len(log["t"]),
                                               float(np.mean(log["cmd"]))))
    else:
        # Reconstruct the command/spike logs from the telemetry the bridge
        # wrote during the recording (last-run values persist in the temp file).
        log = {"t": [], "cmd": [], "nspk": []}
        tel = None
        tp = bridge_g1.TELEMETRY_PATH
        if os.path.exists(tp):
            try:
                tel = json.load(open(tp, encoding="utf-8"))
            except Exception:
                tel = None
        if tel and tel.get("series"):
            log["cmd"] = [s[3] if len(s) > 3 else s[2] for s in tel["series"]]
            log["t"] = [s[0] for s in tel["series"]]
            log["nspk"] = [0.0] * len(log["t"])
        print("replay-only mode (using existing recording + bridge telemetry)")
    return replay_and_render(log, duration_sec)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))