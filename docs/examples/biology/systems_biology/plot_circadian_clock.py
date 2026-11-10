"""
# Circadian clock

The PER protein of Drosophila represses the transcription of its own gene after a chain of phosphorylations and its
entry into the nucleus. The delayed negative feedback turns the gene on and off with a period close to 24 hours, a
self-sustained circadian rhythm.
"""

# %%
# The example builds `CircadianClockSystem` with Goldbeter's parameters and runs it for ten days. The observer saves
# every second frame with the default blueprint, whose primary view is the time series of the mRNA and the forms of PER.

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.systems_biology import CircadianClockSystem

SYSTEM = 'circadian_clock'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(CircadianClockSystem(), observers=[observer])

# %%
# Period of the clock
# -------------------
# The time between peaks of the mRNA should be close to 24 hours for the published parameters. Changing the maximal
# degradation rate of PER, the parameter Goldbeter used to system the period mutants of the fly, should shift the period
# by a few hours either way.

fig, (series, periods) = plt.subplots(1, 2, figsize=(11, 4))
series.plot(trajectory.times, trajectory.get_values('clock.M'), label='mRNA M')
series.plot(trajectory.times, trajectory.get_values('clock.PN'), label='nuclear PER')
series.set_xlabel('time (h)')
series.set_ylabel('concentration (µM)')
series.set_title('Self-sustained oscillations')
series.legend()
rates = np.linspace(0.8, 1.6, 9)
lengths = []
for rate in rates:
    run = simulate(CircadianClockSystem(v_d=rate), until=480)
    peaks, _ = find_peaks(run.get_values('clock.M'))
    lengths.append(np.diff(run.times[peaks][-4:]).mean())
periods.plot(rates, lengths, 'o-')
periods.axhline(24.0, color='grey', linestyle=':', label='24 hours')
periods.axvline(0.95, color='black', linestyle='--', label='published v_d')
periods.set_xlabel('maximal degradation rate v_d (µM/h)')
periods.set_ylabel('period (h)')
periods.set_title(
    f'Period {np.diff(trajectory.times[find_peaks(trajectory.get_values("clock.M"))[0]]).mean():.1f} h at v_d = 0.95',
)
periods.legend()
fig.tight_layout()
plt.show()
