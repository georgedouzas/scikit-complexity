"""
# Elementary flux modes

Every steady-state flux of a metabolic network is a non-negative combination of its elementary modes, the pathways that
no smaller pathway can replace. Raman's toy network has two, and they differ only in the route from `A` to `B`, so both
have one yield, and the same bound limits the growth whichever of them carries the flux.
"""

# %%
# The model is created by name from the catalog, and its modes come exactly from the steady-state cone of its program.
# The primary view is the bar chart of the optimum.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

from skcomplex.analysis import ElementaryModeAnalysis
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create

MODEL = 'elementary_flux_modes'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
trajectory = Simulation(system, observers=[observer]).check().prepare().run(until=0.0).trajectory_

# %%
# The modes and the optimum
# -------------------------
# Normalised to the uptake `b1`, both modes should have the growth yield 1/12. The FBA optimum should be their
# non-negative combination, with no residual, so its growth of 50 is the bound 1000 on `v6` over the ratio 20.

analysis = ElementaryModeAnalysis(normalize='cell.v_b1', sort='cell.v_vbio').check().prepare(system).run()
modes = np.asarray(analysis.modes_, dtype=float)
names = [label.rpartition('v_')[2] for label in analysis.labels_]
optimum = np.array([float(trajectory.values(f'cell.v_{name}')[0]) for name in names])
weights, residual = nnls(modes.T, optimum)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
image = axes[0].imshow(modes.T, cmap='viridis')
axes[0].set_yticks(np.arange(len(names)), names)
axes[0].set_xticks(np.arange(modes.shape[0]), [f'mode {index + 1}' for index in range(modes.shape[0])])
fig.colorbar(image, ax=axes[0], label='flux per unit of b1')
axes[1].bar(np.arange(len(names)) - 0.2, optimum, width=0.4, label='FBA optimum')
axes[1].bar(np.arange(len(names)) + 0.2, weights @ modes, width=0.4, label='combination of modes')
axes[1].set_xticks(np.arange(len(names)), names, rotation=45)
axes[1].legend()
yields = modes[:, names.index('vbio')]
fig.suptitle(
    f'{modes.shape[0]} modes of yield {yields[0]:.4f}; the optimum is their combination, residual {residual:.1e}',
)
fig.tight_layout()
plt.show()
