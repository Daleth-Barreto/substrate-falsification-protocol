"""Smoke/tuning: report firing rates, channel scale and metrics for one run."""

import json

import numpy as np

import hub
import metrics

from run import calibrate, make_profile, run_one, PROFILES, SEEDS, CAL_SEED


def main():
    cal = calibrate()
    print("calibration k_ro lif=%.3f izh=%.3f poisson=%.3f" % (
        cal["lif_k_ro"], cal["izh_k_ro"], cal["poisson_k_ro"]))
    print("mean firing LIF=%.1f Hz, Poisson null=%.1f Hz" % (
        cal["lif_mean_hz"], cal["poisson_mean_hz"]))
    seed = SEEDS[0]
    for p in PROFILES:
        u = make_profile(p)
        r = run_one(p, u, seed, cal)
        for sn, v in r.items():
            print("  %-8s %-10s rmse=%.3f mi=%.4f te=%.4f te_p=%.3f rho=%.2f rate=%.1fHz" % (
                p, sn, v["rmse"], v["mi"], v["te"], v["te_p"], v["rho"], v["rate_hz"]))


if __name__ == "__main__":
    main()