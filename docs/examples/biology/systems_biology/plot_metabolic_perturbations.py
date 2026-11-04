"""
# Metabolic perturbations

A gene that is off closes the reactions whose rules need it. Most genes of E. coli core can go, since other routes take
over, but a few are essential. After a knock-out, the regulation of a real cell need not reach the new optimum at once,
and MoMA and ROOM predict the adjusted fluxes from the wild-type solution.
"""

# %%
# The model is created by name from the catalog with the `pgi` gene knocked out, which needs the extra `metabolism`. The
# primary view is the bar chart of the fluxes of the one frame.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create
from skcomplex.models.biology.common.metabolism import CobraAnalysis

MODEL = 'metabolic_perturbations'
GROWTH = 'cell.objective_0'

system = create(MODEL, genes=('b4025',))
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
knocked_out = Simulation(system, observers=[observer]).check().prepare().run(until=0.0).trajectory_

# %%
# Single deletions
# ----------------
# Deleting every gene in turn should leave most of the growth, and exactly 7 genes should stop it.

wild_type = create(MODEL)
deletions = CobraAnalysis('deletions').check().prepare(wild_type).run().growth_
essential = int((deletions < 1e-6).sum())
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(np.sort(deletions.to_numpy())[::-1], '.')
ax.set_xlabel('gene, by the growth without it')
ax.set_ylabel('growth (1/h)')
ax.set_title(f'{essential} of {deletions.size} genes are essential in E. coli core')
fig.tight_layout()
plt.show()

# %%
# Adjustment after the knock-out of pgi
# -------------------------------------
# FBA should give the new optimum, and MoMA and ROOM, which stay close to the wild type, positive growth below it.

optimum = float(knocked_out.values(GROWTH)[0])
moma = CobraAnalysis('moma', knockouts=('b4025',)).check().prepare(wild_type).run().growth_
room = CobraAnalysis('room', knockouts=('b4025',)).check().prepare(wild_type).run().growth_
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(['FBA', 'MoMA', 'ROOM'], [optimum, moma, room])
ax.set_ylabel('growth (1/h)')
ax.set_title(f'After pgi: FBA {optimum:.4f}, MoMA {moma:.4f}, and linear ROOM {room:.4f} 1/h')
fig.tight_layout()
plt.show()
