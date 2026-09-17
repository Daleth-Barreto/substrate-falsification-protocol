"""Gate D, n=20 seeds battery (additive; artifact gates in results20/).

Does NOT touch gate_d_summary.json (v1, 6 seeds). Patches run.py's SEEDS with
a 20-seed list (the 6 canonical seeds plus 14 new ones) and reruns the full
Gate D battery into results20/. Then adds a seed-level paired permutation test
(LIF vs Poisson null, IZH vs Poisson null) over the n=20 seeds, answering the
reviewer's "only 6 seeds" concern with exact statistics at seed granularity.
"""

import json
import os

import numpy as np

import run as gd

SEEDS20 = [1, 7, 13, 29, 55, 91, 2, 3, 5, 11, 17, 19, 23, 31, 37, 41, 43, 47, 53, 59]
OUT20 = os.path.join(os.path.dirname(__file__), "results20")
EXTRA_JSON = os.path.join(OUT20, "gate_d20_seed_stats.json")

assert len(SEEDS20) == 20 and len(set(SEEDS20)) == 20


def main():
    gd.SEEDS = SEEDS20
    gd.OUTDIR = OUT20
    gd.main()

    rows = []
    for pname in gd.PROFILES:
        for seed in SEEDS20:
            with open(os.path.join(OUT20, f"raw_{pname}_s{seed}.json")) as f:
                rows.append((pname, seed, json.load(f)))

    subs = ["lif", "izh", "poisson"]
    # per-seed carried info = meanpd_rmse - ridge_rmse, pooled across the
    # 5 profiles of that seed.
    per_seed = {}
    for seed in SEEDS20:
        d = {sn: [] for sn in subs}
        for _, s2, r in rows:
            if s2 != seed:
                continue
            for sn in subs:
                d[sn].append(r[sn]["meanpd"]["rmse"] - r[sn]["ridge"]["rmse"])
        per_seed[seed] = {sn: float(np.mean(d[sn])) for sn in subs}
        per_seed[seed]["_n_profiles"] = len(d[subs[0]])

    diffs = {sn: np.array([per_seed[s][sn] - per_seed[s]["poisson"] for s in SEEDS20])
             for sn in ("lif", "izh")}

    def paired_perm_p(x, n_perm=100000):
        rng = np.random.default_rng(4242 + x.size)
        obs = float(x.mean())
        count = 0
        for _ in range(n_perm):
            signs = rng.integers(0, 2, size=x.size) * 2 - 1
            count += (signs * x).mean() >= obs
        return obs, (count + 1) / (n_perm + 1)

    res = {}
    for sn in ("lif", "izh"):
        obs, p = paired_perm_p(diffs[sn])
        res[sn] = {
            "mean_effect_vs_poisson": float(obs),
            "se_effect": float(diffs[sn].std() / np.sqrt(len(diffs[sn]))),
            "paired_perm_p_one_sided": float(p),
            "seeds_same_direction": int(np.sum(diffs[sn] > 0)),
            "n_seeds": len(SEEDS20),
        }
    res["seeds"] = SEEDS20
    verdict = (
        f"n=20 seeds: LIF info-over-null {res['lif']['mean_effect_vs_poisson']:+.3f} "
        f"(paired perm p≈{res['lif']['paired_perm_p_one_sided']:.4g}, "
        f"{res['lif']['seeds_same_direction']}/20 seeds same direction); "
        f"IZH {res['izh']['mean_effect_vs_poisson']:+.3f} "
        f"(p≈{res['izh']['paired_perm_p_one_sided']:.4g}, "
        f"{res['izh']['seeds_same_direction']}/20)."
    )
    res["verdict"] = verdict
    with open(EXTRA_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()