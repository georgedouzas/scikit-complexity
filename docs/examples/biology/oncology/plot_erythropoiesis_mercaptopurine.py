"""
# Red cells under 6-mercaptopurine

From the healthy steady state, a constant level of 6-mercaptopurine makes the mean red-cell volume rise and the mature
red cells and the quiescent cells fall, all settling over a few years at new steady values.
"""

# %%
# The system is built from its class with a drug level of 500 pMol per 8 x 10^8 red cells, within the range of the
# book. The observer saves every second frame with the default blueprint, whose primary view is the time series of the
# cells and the mean volume.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import ErythropoiesisMercaptopurineSystem

SYSTEM = 'erythropoiesis_mercaptopurine'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(ErythropoiesisMercaptopurineSystem(U=500.0), observers=[observer])

# %%
# Volume and cell counts at steady state
# --------------------------------------
# At steady state the mean volume is the volume of the new cells, `MCV0 + g1 U`, and the mature cells balance their
# production, `L = r1 Q / d1`. The runs for several drug levels should approach the straight line of the volume and the
# falling cell counts.

levels = np.array([0.0, 158.0, 300.0, 500.0, 750.0, 1000.0])
volumes, cells = [], []
for level in levels:
    trajectory = simulate(ErythropoiesisMercaptopurineSystem(U=level), dt=20.0, until=8000)
    volumes.append(trajectory.get_values('patient.MCV')[-1])
    cells.append(trajectory.get_values('patient.L')[-1])
fig, (course, steady) = plt.subplots(1, 2, figsize=(10, 4))
course.plot(trajectory.times, trajectory.get_values('patient.MCV'), label='mean volume (fL)')
course.axhline(85.0 + 0.0317 * 500.0, color='grey', linestyle=':', label='MCV0 + g1 U')
course.set_xlabel('time (days)')
course.set_ylabel('mean corpuscular volume (fL)')
course.set_title('The mean volume rises under the drug')
course.legend()
steady.plot(levels, volumes, 'o', label='simulated volume')
steady.plot(levels, 85.0 + 0.0317 * levels, '-', color='black', label='MCV0 + g1 U')
steady.set_xlabel('6-TGN level (pMol per 8 x 10^8 red cells)')
steady.set_ylabel('steady mean volume (fL)')
steady.set_title('Steady volume against drug level')
cells_axis = steady.twinx()
cells_axis.plot(levels, cells, 's', color='tab:red', label='mature cells')
cells_axis.set_ylabel('mature red cells (10^12 per kg)')
steady.legend(loc='upper left')
cells_axis.legend(loc='lower right')
fig.tight_layout()
plt.show()
