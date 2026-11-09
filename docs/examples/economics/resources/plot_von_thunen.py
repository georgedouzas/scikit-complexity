"""
# Von Thünen land use

Around one market, every cell grows the crop that bids the highest rent `Y_i (p_i - c_i) - Y_i f_i d` at its distance
`d`. The crop with the highest yield per cell bids most near the market, and its bid falls fastest with distance, so
the land
use forms concentric rings, beyond which the land lies idle.
"""

# %%
# The system solves the crop of each of 41 x 41 cells once, drawn as an image.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.experiments import sweep
from skcomplex.systems.economics.resources import VonThunenSystem

SYSTEM = 'von_thunen'

system = VonThunenSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, until=1.0, observers=[observer])

# %%
# Rings against the transport rate of the inner crop
# --------------------------------------------------
# A `sweep` varies the transport rates. A higher rate of the first crop lowers its bid faster, so its ring
# shrinks, while the second crop takes the land it gives up.

rates = [(0.2, 0.2, 0.2), (0.25, 0.2, 0.2), (0.3, 0.2, 0.2)]
result = sweep(VonThunenSystem(), {'rates': rates}, until=1.0, outputs=['plain.crop'])
crops = result.get_series('plain.crop')[:, 0]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
for ax, rate, crop in zip(axes, rates, crops, strict=True):
    ax.imshow(crop.reshape(41, 41), cmap='viridis', vmin=0, vmax=3)
    ax.set_title(f'f_1 = {rate[0]}: {np.sum(crop == 1):.0f} cells of crop 1')
    ax.set_axis_off()
fig.tight_layout()
plt.show()
