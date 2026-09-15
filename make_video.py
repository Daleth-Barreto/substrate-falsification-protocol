"""make_video.py — animated spike raster + decode for the pulse profile.

Three panels (LIF / Izhikevich / Poisson-dead-null) running the SAME wiring,
input profile and canonical calibrated decode. The animation scans the control
windows, drawing the spike raster (a subsample of neurons) as the spikes
accumulate, and the running decoded vx(t) against the reference.

Output: sandbox/video/pulse_raster.mp4  (ffmpeg, ~24 s, ~15 fps)
Reproducible: seeds fixed (default 7, same as the calibrated track).

Usage:
    ..\02_cl\.venv312\Scripts\python.exe make_video.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gate_d"))
import hub as hubd
import run as rund

OUTDIR = os.path.join(os.path.dirname(__file__), "video")
OUTMP4 = os.path.join(OUTDIR, "pulse_raster.mp4")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cal = rund.calibrate()
    seed = 7
    gw, ww, wrr = hubd.make_wiring(seed)
    u = rund.make_profile("pulse")
    vref = 0.75 * u

    subs = {
        "LIF": (hubd.LIF(), "lif", 0.0, rund.DRIVE_GAIN.get("lif", 0.0),
                rund.REC_SCALE.get("lif", 0.0), cal["lif_k_ro"]),
        "Izhikevich": (hubd.IZH(), "izh", 0.0, rund.DRIVE_GAIN.get("izh", 0.0),
                       rund.REC_SCALE.get("izh", 0.0), cal["izh_k_ro"]),
        "Poisson (dead null)": (hubd.PoissonNull(cal["poisson_rate"]), "poisson",
                                0.0, 0.0, 0.0, cal["poisson_k_ro"]),
    }

    # Run all three full profiles (cheap: ~200 control windows of 40 steps)
    dat = {}
    for label, (sub, sn, bi, dg, rc, k_ro) in subs.items():
        sub.reset(seed)
        thsig, counts = hubd.run_profile(sub, (gw, ww, wrr), u, bi, dg, rc, k_ro)
        xhat = hubd.decide(thsig)
        dat[label] = (counts, xhat)

    SUBN = 60  # neurons shown per panel (subsample)
    nidx = {"LIF": 0, "Izhikevich": 1, "Poisson (dead null)": 2}
    fig, axes = plt.subplots(3, 2, figsize=(11, 8),
                             gridspec_kw={"width_ratios": [1, 1]})
    rasters, decodes = axes[:, 0], axes[:, 1]

    # init
    for ax, label in zip(rasters, subs):
        counts, _ = dat[label]
        ax.imshow(counts[:SUBN, :].astype(bool).astype(float), aspect="auto",
                  cmap="Greys", origin="lower", interpolation="nearest",
                  extent=[0, hubd.N_WIN, 0, SUBN])
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("control window"); ax.set_ylabel("neuron")
        ax.set_yticks([])

    (lif_line,) = decodes[0].plot([], [], color="#2c7fb8", lw=1.6, label="decode vx")
    (lif_ref,) = decodes[0].plot([], [], color="#000000", lw=1.0, ls="--", label="vref")
    (izh_line,) = decodes[1].plot([], [], color="#d95f0e", lw=1.6, label="decode vx")
    (izh_ref,) = decodes[1].plot([], [], color="#000000", lw=1.0, ls="--", label="vref")
    (poi_line,) = decodes[2].plot([], [], color="#636363", lw=1.6, label="decode vx")
    (poi_ref,) = decodes[2].plot([], [], color="#000000", lw=1.0, ls="--", label="vref")
    for ax in decodes:
        ax.set_xlim(0, len(u)); ax.set_ylim(-0.05, 0.85)
        ax.set_xlabel("window"); ax.set_ylabel("velocity")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    decodes[2].set_title("decoded vx vs reference (canonical calibrated map)" +
                         "  |  seed=%d, profile=pulse" % seed, fontsize=10)

    def frame(wi):
        for ax, label in zip(rasters, subs):
            counts, _ = dat[label]
            ax.images[0].set_data(counts[:SUBN, :wi + 1].astype(bool).astype(float))
        wk = np.arange(wi + 1)
        lif_line.set_data(wk, dat["LIF"][1][:wi + 1])
        lif_ref.set_data(wk, vref[:wi + 1])
        izh_line.set_data(wk, dat["Izhikevich"][1][:wi + 1])
        izh_ref.set_data(wk, vref[:wi + 1])
        poi_line.set_data(wk, dat["Poisson (dead null)"][1][:wi + 1])
        poi_ref.set_data(wk, vref[:wi + 1])
        fig.suptitle(f"spike raster + decode  window={wi:03d}/{hubd.N_WIN - 1}",
                     fontsize=12)
        return [lif_line, lif_ref, izh_line, izh_ref, poi_line, poi_ref]

    nframes = hubd.N_WIN  # full sweep
    anim = matplotlib.animation.FuncAnimation(fig, frame, frames=nframes,
                                              interval=80, blit=False, repeat=False)
    # Ensure even dimensions (yuv420p requires w,h both even) then encode
    # H.264 High profile with faststart for web-friendly seeking.
    writer = FFMpegWriter(fps=12, bitrate=1200, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p",
                                      "-movflags", "faststart"])
    anim.save(OUTMP4, writer=writer, dpi=110)
    plt.close(fig)
    print("WROTE", OUTMP4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())