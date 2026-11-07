"""
# Induced drug resistance

Under a constant dose the drug kills the sensitive cells, and the resistant cells, which the drug itself induces, grow
to fill the tumour's capacity.
"""

# %%
# The system is built from its class with the book's parameters: a spontaneous resistance rate of 10^-6, a resistance
# induced at the rate 0.01 per unit of dose, and a resistant population that grows at a fifth of the rate of the
# sensitive one. The observer saves every frame with the default blueprint, whose primary view is the time series of
# the three populations.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import DrugResistanceSwitchingSystem

SYSTEM = 'drug_resistance_switching'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(DrugResistanceSwitchingSystem(), dt=0.5, observers=[observer], until=120)

# %%
# Spontaneous and induced resistance
# ----------------------------------
# Without induction, resistance arises only from the rare spontaneous mutations and takes much longer to fill the
# tumour. With induction the resistant cells should reach half the capacity in less than half the time.

spontaneous = simulate(DrugResistanceSwitchingSystem(alpha_m=0.0), dt=0.5, until=120)
induced = trajectory
fig, (resistant, sensitive) = plt.subplots(1, 2, figsize=(10, 4))
resistant.plot(induced.times, induced.values('tumour.Rm'), label='induced, alpha_m = 0.01')
resistant.plot(spontaneous.times, spontaneous.values('tumour.Rm'), label='spontaneous only')
resistant.axhline(0.5, color='grey', linestyle=':', label='half the capacity')
resistant.set_xlabel('time')
resistant.set_ylabel('resistant cells (fraction of capacity)')
resistant.set_title('Induction speeds up resistance')
resistant.legend()
sensitive.semilogy(induced.times, induced.values('tumour.S'), label='induced, alpha_m = 0.01')
sensitive.semilogy(spontaneous.times, spontaneous.values('tumour.S'), label='spontaneous only')
sensitive.set_xlabel('time')
sensitive.set_ylabel('sensitive cells (fraction of capacity)')
sensitive.set_title('The drug kills the sensitive cells')
sensitive.legend()
fig.tight_layout()
plt.show()
