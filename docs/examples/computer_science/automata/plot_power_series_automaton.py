"""
# Power-series automaton

The cells sit on a ring of 31 cells and hold formal power series over the rationals, truncated at the order 8. Every
cell becomes `x(n) - t x(n + 1)`, a bijective automaton over the ring `K[[t]]` whose inverse
`sigma(y)(n) = sum_j t^j y(n + j)` is not continuous, so it is no cellular automaton. From the constant series 1 in the
middle cell, the `k`-th cell to the left holds `C(s, k) (-t)^k` after `s` steps.
"""

# %%
# The run starts from the series 1 in the middle cell and logs every coefficient of every cell.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'power_series_automaton'

system = create_system(SYSTEM, size=31, order=8)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
simulation = Simulation(system, observers=[observer]).prepare().run(until=12)

# %%
# Coefficients and the inverse
# ----------------------------
# The coefficient of `t^k` should sit only at the cell `k` to the left of the middle, with the value `C(s, k) (-1)^k`,
# so each power lights one diagonal. Applying the geometric inverse `sum_j t^j y(n + j)` to the last configuration
# should give back the configuration one step earlier, exactly, since the truncation keeps every coefficient below the
# order.

values = simulation.trajectory_.values('cells.value')
steps = values.shape[0] - 1
last, before = values[-1], values[-2]
recovered = np.zeros_like(last)
for shift in range(last.shape[1]):
    recovered[:, shift:] += np.roll(last, -shift, axis=0)[:, : last.shape[1] - shift]
fig, (image, check) = plt.subplots(1, 2, figsize=(10, 4))
shown = image.imshow(np.sign(last.T) * np.log10(1 + np.abs(last.T)), aspect='auto', cmap='coolwarm')
image.set_xlabel('cell n')
image.set_ylabel('power of t')
image.set_title(f'Signed log of the coefficients at step {steps}')
fig.colorbar(shown, ax=image)
middle = 15
powers = np.arange(last.shape[1])
check.semilogy(powers, np.abs(last[middle - powers, powers]), 'o', label='coefficient of $t^k$ at cell $-k$')
check.semilogy(
    powers,
    [np.prod(np.arange(steps - k + 1, steps + 1)) / np.prod(np.arange(1, k + 1)) for k in powers],
    '--',
    color='black',
    label='$C(s, k)$',
)
check.set_xlabel('k')
check.set_title(f'Inverse recovers the previous step: {np.array_equal(recovered, before)}')
check.legend()
fig.tight_layout()
plt.show()
