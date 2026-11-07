"""
# Intermittent androgen suppression switched by PSA

The therapy starts when the PSA rises through 15 ng/mL and stops when it falls through 10 ng/mL, so the PSA cycles
between the two thresholds.
"""

# %%
# The system is built from its class: five androgen-dependent cells at the normal androgen of 30 nmol/L. The observer
# saves every third frame with the default blueprint, whose primary view is the time series of the cells, the
# androgen, and the therapy.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import IdetaIntermittentAndrogenSystem

SYSTEM = 'ideta_intermittent_androgen'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=3)
trajectory = simulate(IdetaIntermittentAndrogenSystem(), dt=1.0, observers=[observer])

# %%
# Switching at the thresholds
# ---------------------------
# Every `on` event happens with the PSA at the upper threshold and every `off` event at the lower one. The androgen
# follows the therapy with a delay of about `1 / du = 12.5` days, so the PSA overshoots the thresholds a little before
# it turns.

times, psa = trajectory.times, trajectory.values('tumour.AD') + trajectory.values('tumour.AI')
fig, (level, androgen) = plt.subplots(1, 2, figsize=(10, 4))
level.plot(times, psa, label='PSA')
level.axhline(15.0, color='black', linestyle='--', label='upper threshold')
level.axhline(10.0, color='grey', linestyle='--', label='lower threshold')
for event in trajectory.events:
    level.axvline(event.time, color='red' if event.label == 'on' else 'green', alpha=0.3)
level.set_xlabel('time (days)')
level.set_ylabel('PSA (ng/mL)')
level.set_title('Therapy on (red) and off (green)')
level.legend()
androgen.plot(times, trajectory.values('tumour.U'))
androgen.set_xlabel('time (days)')
androgen.set_ylabel('serum androgen (nmol/L)')
androgen.set_title('The androgen falls during therapy')
fig.tight_layout()
plt.show()
