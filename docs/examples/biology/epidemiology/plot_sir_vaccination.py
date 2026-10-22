"""
# SIR with optimal vaccination

A health authority vaccinates susceptible people at a bounded rate to minimise the burden of an epidemic plus the
cost of the vaccines. The view shows the optimal rate highest at the start, while most people are susceptible, and
tapering off as vaccination moves them to the recovered, so the infected fraction stays near 2% instead of peaking
near half of the population without vaccination.
"""

# %%
# The model is created by name from the catalog with the transmission rate 0.5, the recovery rate 0.1, and 1% of the
# population infected at the start, so the basic reproduction number is 5. The Pontryagin solver of its catalog entry
# derives the costates and the vaccination rate `u = (lambda_S - lambda_R) S / b` symbolically. The default blueprint
# shows the three fractions and the vaccination rate in one time series view.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'sir_vaccination'

system = create(MODEL)
entry = describe(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(
    system,
    dt=0.2,
    solver=entry.defaults['solver'],
    observers=[observer],
    observables=entry.observables_of(system),
).run(until=30)

# %%
# Vaccination flattens the epidemic
# ---------------------------------
# With no weight on the infections, `a = 0`, the optimal rate is 0 and the epidemic runs its course. With vaccination,
# the infected fraction should peak much lower, because vaccination removes susceptible people before the epidemic
# reaches them.

uncontrolled = Simulation(create(MODEL, a=0.0), dt=0.2, solver=entry.defaults['solver']).run(until=30).trajectory_
trajectory = simulation.trajectory_
fig, (infected, rate) = plt.subplots(1, 2, figsize=(10, 4))
infected.plot(uncontrolled.times, uncontrolled['population.I'], label='without vaccination')
infected.plot(trajectory.times, trajectory['population.I'], label='optimal vaccination')
infected.set_xlabel('time')
infected.set_ylabel('infected fraction I')
infected.set_title('The peak drops with vaccination')
infected.legend()
rate.plot(trajectory.times, trajectory['population.u'])
rate.set_xlabel('time')
rate.set_ylabel('vaccination rate u')
rate.set_title('Vaccinate early')
fig.tight_layout()
plt.show()
