"""One-entry verification of the paper artifact pipeline.

Regenerates tables/figures and rebuilds the video from committed artifacts,
without re-running any experiment. Run from the repo root (sandbox/).

    python reproduce.py

Then (optionally) recompile paper/main.pdf with MiKTeX:
    cd paper && pdflatex main.tex  (twice)

Exit code 0 on success.
"""
import subprocess
import sys

PY = sys.executable

steps = [
    ("artifacts (tables + figures)", [PY, "build_figures_tables.py"]),
    ("video (pulse MP4)", [PY, "make_video.py"]),
]

for name, cmd in steps:
    print(f"== {name} ==")
    r = subprocess.run(cmd, cwd=__file__.rsplit("\\", 1)[0] or ".")
    if r.returncode != 0:
        print(f"FAILED: {name}")
        sys.exit(r.returncode)

print("reproduce.py: artifacts + video OK")