"""
# Lattice gauge fields

Angles on the links of a periodic lattice sample the Wilson action `-beta sum_p cos(theta_p)`, where `theta_p` is the
oriented sum of the angles around a plaquette. In two dimensions the plaquettes are independent, so the mean plaquette
has the exact value `I_1(beta) / I_0(beta)`.
"""

# %%
# A run at `beta = 2` on a 16 by 16 lattice records the mean plaquette at every sweep, and the Rerun view shows the
# links and the plaquette sums with a cyclic palette.

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import iv

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.waves import LatticeGaugeFieldsSystem

SYSTEM = 'lattice_gauge_fields'

system = LatticeGaugeFieldsSystem(size=16)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=10)
trajectory = simulate(system, until=400, observers=[observer], random_state=0)
series = trajectory.values('mean_plaquette')

# %%
# The plaquette against the exact result
# --------------------------------------
# The series relaxes from random links to the exact value in a few sweeps.

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(series, label='mean plaquette')
ax.axhline(iv(1, 2) / iv(0, 2), color='black', linestyle='--', label='I_1(2) / I_0(2)')
ax.set_xlabel('sweep')
ax.set_ylabel('mean cosine of the plaquette sums')
ax.set_title(f'Wilson action at beta = 2, mean {np.mean(series[100:]):.3f}')
ax.legend()
plt.show()
