"""robot_video.py — capture an offscreen MP4 of the G1 walker running.

Runs the base MuJoCo G1 plant (01_snn/src/deploy12.py -> Deploy12) under the
pre-trained LSTM policy with a constant cmd=[0.5, 0, 0] m/s, and renders every
frame offscreen to an H.264 yuv420p MP4 with a live telemetry overlay.

Output: sandbox/video/g1_walk.mp4  (H.264 High + yuv420p, universal playback)

Usage:
    ..\02_cl\.venv312\Scripts\python.exe robot_video.py

Requires the project venv (mujoco, torch, yaml, matplotlib). No display.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO, "01_snn", "src"))
import deploy12  # noqa: E402  Deploy12 walker

OUTDIR = os.path.join(os.path.dirname(__file__), "video")
OUTMP4 = os.path.join(OUTDIR, "g1_walk.mp4")

DURATION = 12.0        # s of simulated walking
CMD = [0.5, 0, 0]      # [vx, vy, yaw_rate]
FPS = 33.0             # video frames per second
SUB = int(round(1.0 / (FPS * 0.002)))      # sim steps between rendered frames
W, H = 640, 480
REPORT_EVERY = 200

# Sensor indices for the overlay (pelvis height qpos[2], vx qvel[0], pitch)
from numpy import arctan2  # noqa: E402


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    deploy = deploy12.Deploy12()
    deploy.reset(cmd=CMD)

    writer = FFMpegWriter(fps=FPS, bitrate=1800, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p",
                                      "-movflags", "faststart"])
    fig = plt.figure(figsize=(W / 96, H / 96))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    im = ax.imshow(np.zeros((H, W, 3), dtype=np.uint8))

    nf = int(DURATION * FPS)
    print(f"rendering {nf} frames ({FPS:.0f} fps, {SUB} sim steps/frame) ...")
    with writer.saving(fig, OUTMP4, dpi=96):
        step = 0
        while deploy.data.time < DURATION:
            deploy.step_ctrl()
            step += 1
            if step % deploy.dec == 0:
                deploy.policy_step()
            if step % SUB == 0:
                frame = deploy.render_frame(width=W, height=H)
                im.set_data(frame)
                d = deploy.data
                vx = d.qvel[0]
                h = d.qpos[2]
                q = d.qpos[3:7]
                pitch = arctan2(2 * (q[3] * q[2] + q[0] * q[1]),
                                1 - 2 * (q[1] * q[1] + q[2] * q[2]))
                info = (f"G1 base policy  cmd vx={CMD[0]:.2f} m/s\n"
                        f"t={d.time:5.2f}s  vx={vx:+6.2f} m/s  "
                        f"h={h:5.2f} m  pitch={pitch:+.1f} deg")
                ax.set_title(info, fontsize=9, loc="left",
                             color="#ffffff", backgroundcolor="#00000088")
                writer.grab_frame()
            if (step % REPORT_EVERY) == 0:
                print(f"  t={deploy.data.time:5.2f}/{DURATION:5.2f} s", flush=True)
            if deploy.data.qpos[2] < 0.35:
                print("FALLEN during capture; stopping early.")
                break
    plt.close(fig)
    deploy.close()
    print("WROTE", OUTMP4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())