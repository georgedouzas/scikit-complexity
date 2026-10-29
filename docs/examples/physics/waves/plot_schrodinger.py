"""
# Schrödinger equation as real and imaginary fields

A Gaussian wave packet that follows the Schrödinger equation, as its real and imaginary parts, moves at its group
velocity and spreads on a periodic lattice.
"""

# %%
# The model is created by name from the catalog and run with its default settings, the exponential integrator
# ETDRK4 for the two fields. The observer saves every frame with the default blueprint.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.analysis import QuantumAnalysis
from skcomplex.engine.compile import QuantumCompiler
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import ETDRK4Solver
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create, run

MODEL = 'schrodinger'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = run(MODEL, settings={'observers': [observer], 'random_state': 0})

# %%
# Fields, operators, and dephasing
# --------------------------------
# The operator formulation builds the same packet as one particle on the sites of the lattice, whose Hamiltonian
# `-D laplacian + V` uses the same spectral Laplacian, and QuTiP evolves it. The left panel shows that the probability
# of every site agrees between the two formulations. The right panel adds dephasing in the position basis, which only
# the operator formulation can express: the state becomes a density matrix whose purity `Tr(rho²)` falls from 1 while
# its trace stays 1.

spacing = 40.0 / 128
fields = (
    Simulation(create(MODEL, n_sites=128), dt=0.5, solver=ETDRK4Solver(dt=0.01))
    .check()
    .prepare()
    .run(until=4.0)
    .trajectory_
)
operators = create(MODEL, n_sites=128, formulation='operators')
operators = Simulation(operators, compiler=QuantumCompiler(), dt=0.5).check().prepare().run(until=4.0).trajectory_
density = (fields.values('packet.u') ** 2 + fields.values('packet.w') ** 2) * spacing
probability = np.abs(operators.values('schrodinger.state')) ** 2
positions = spacing * np.arange(128) - 20.0
fig, (agreement, coherence) = plt.subplots(1, 2, figsize=(10, 4))
agreement.plot(positions, density[-1], label='fields, u² + w²')
agreement.plot(positions, probability[-1], linestyle='--', label='operators, |psi|²')
agreement.set_xlabel('position')
agreement.set_ylabel('probability of a site at t = 4')
agreement.set_title('The two formulations agree')
agreement.legend()
for gamma in (0.1, 0.5, 2.0):
    dephased = create(MODEL, n_sites=48, length=20.0, formulation='operators', gamma=gamma)
    trajectory = Simulation(dephased, compiler=QuantumCompiler(), dt=0.1).check().prepare().run(until=4.0).trajectory_
    purity = QuantumAnalysis('purity').check().prepare(trajectory).run().purity_
    coherence.plot(trajectory.times, purity, label=f'gamma = {gamma}')
coherence.set_xlabel('time')
coherence.set_ylabel('purity')
coherence.set_title('Position dephasing destroys coherence')
coherence.legend()
fig.tight_layout()
plt.show()
