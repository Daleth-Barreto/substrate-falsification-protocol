import sys
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import bridge_g1

import cl
from cl.sim import (SimulatorDataSource, DataSourceBatch,
                    SimulatorDataSourceMetadata, set_simulator_data_source)

AMPS = [200.0, 2000.0, 8000.0, 20000.0]
N = 250


class ProbeSource(SimulatorDataSource):
    def __init__(self):
        self.runs = 0

    @property
    def metadata(self):
        return SimulatorDataSourceMetadata(
            channel_count=bridge_g1.N_CHANNELS, frames_per_second=25000,
            uV_per_sample_unit=0.195, duration_frames=None, seekable=False,
            realtime_only=True, supports_accelerated=False)

    def open(self):
        pass

    def close(self):
        pass

    def read(self, from_timestamp, frame_count):
        n = int(frame_count)
        f = np.zeros((n, bridge_g1.N_CHANNELS), dtype=np.int16)
        if n <= 0:
            return DataSourceBatch(frames=f)
        rng = np.random.default_rng(from_timestamp)
        if self.runs < 10:
            self.runs += 1
            f[:, 0] = (rng.normal(0.0, 1200.0, n)).astype(np.int16)
            f[:, 1] = (rng.normal(0.0, 6000.0, n)).astype(np.int16)
            f[:, 2] = (rng.normal(0.0, 200.0, n)).astype(np.int16)
            f[:, 3] = (rng.normal(0.0, 30000.0, n)).astype(np.int16)
            f[:, 4] = 30000
            f[:, 5] = -30000
        else:
            tick = (from_timestamp // N) % len(AMPS)
            amp = AMPS[tick]
            for k in range(10):
                nb = min(5 + k * 24, n)
                ne = min(5 + k * 24 + 10, n)
                if ne > nb:
                    seg = np.arange(ne - nb)
                    f[nb:ne, 0] = (amp * np.sin(2 * np.pi * seg / 10.0)).astype(np.int16)
            f[:, 2] = (rng.normal(0.0, 500.0, n)).astype(np.int16)
            f[0:n, 3] = (rng.normal(0.0, 30000.0, n)).astype(np.int16)
        return DataSourceBatch(frames=f)


def create():
    return ProbeSource()


def main():
    set_simulator_data_source(
        "probe_detect:create",
        metadata=SimulatorDataSourceMetadata(
            channel_count=bridge_g1.N_CHANNELS, frames_per_second=25000,
            uV_per_sample_unit=0.195, duration_frames=None, seekable=False,
            realtime_only=True, supports_accelerated=False))
    with cl.open() as neurons:
        for tick in neurons.loop(100, stop_after_seconds=0.4):
            f = tick.frames
            mx = int(np.abs(f).max()) if f is not None else -1
            spks = list(tick.analysis.spikes) if tick.analysis else []
            chs = sorted(set(s.channel for s in spks))
            print("tick", tick.iteration, "max_frame", mx, "nspk", len(spks),
                  "chs", chs[:6])


if __name__ == "__main__":
    main()