"""
# Brock-Mirman stochastic growth

A representative household with logarithmic utility owns a firm whose productivity jumps between a low and a high
level. With full depreciation its optimal consumption is a fixed share `1 - alpha beta` of output, so capital follows
the map `K' = alpha beta z K^alpha` and fluctuates around a stationary distribution.
"""

# %%
# The economy starts with a capital of 0.2, `alpha = 1/3`, `beta = 0.95`, and a productivity of 0.9 or 1.1 that keeps
# its level with probability 0.9, for 200 periods. The default blueprint shows the capital and the regime.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.macroeconomics import BrockMirmanGrowthSystem

SYSTEM = 'brock_mirman_growth'
ALPHA, BETA = 1 / 3, 0.95

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(BrockMirmanGrowthSystem(), observers=[observer])
long_run = simulate(BrockMirmanGrowthSystem(), until=5000)

# %%
# Consumption rule and the stationary capital
# -------------------------------------------
# Consumption should be `0.68` of output in every period, and `ln K` should average
# `(ln(alpha beta) + E ln z) / (1 - alpha)` over a long run, between the steady states of the two productivities.

fig, (rule, stationary) = plt.subplots(1, 2, figsize=(10, 4))
rule.plot(trajectory.values('economy.output'), trajectory.values('economy.consumption'), 'o', ms=3)
outputs = np.linspace(0, trajectory.values('economy.output').max(), 2)
rule.plot(outputs, (1 - ALPHA * BETA) * outputs, 'k--', lw=0.8, label='C = (1 - alpha beta) Y')
rule.set_xlabel('output Y')
rule.set_ylabel('consumption C')
rule.set_title('The optimal consumption rule')
rule.legend()
logs = np.log(long_run.values('economy.K')[100:])
stationary.hist(logs, bins=40, density=True)
for z in (0.9, 1.1):
    stationary.axvline(np.log(ALPHA * BETA * z) / (1 - ALPHA), color='black', linestyle=':', lw=0.8)
stationary.axvline((np.log(ALPHA * BETA) + 0.5 * np.log(0.99)) / (1 - ALPHA), color='red', lw=1.0, label='mean')
stationary.set_xlabel('ln K')
stationary.set_ylabel('density')
stationary.set_title('Stationary distribution of capital')
stationary.legend()
fig.tight_layout()
plt.show()
