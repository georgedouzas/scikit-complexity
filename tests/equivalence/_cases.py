"""Build the library models and documentation examples whose results are recorded and compared."""

import contextlib
import io
import re
from collections.abc import Callable
from pathlib import Path

import networkx as nx
import numpy as np
import rerun as rr

from skcomplex.backends import LammpsBackend
from skcomplex.compile import NumbaBackend
from skcomplex.model import EuclideanSpace, System
from skcomplex.models.ecology import ExponentialGrowth, LogisticGrowth, Mortality, Population, Predation, wolf_sheep
from skcomplex.models.economics import Cobweb, Economy, Market, SolowGrowth
from skcomplex.models.epidemiology import Compartment, Infection, Recovery
from skcomplex.models.finance import Asset, GeometricBrownianMotion
from skcomplex.models.physics import (
    ConstantForce,
    Gravity,
    HarmonicPotential,
    LangevinThermostat,
    LennardJones,
    LinearDrag,
    Particle,
    PeriodicBox,
    ReflectingBox,
    SoftSphere,
    Spring,
    ideal_gas,
)
from skcomplex.models.social import boltzmann_wealth, harvesting, schelling, voter_model
from skcomplex.runtime import Simulation
from skcomplex.solvers import ScipyODE, VelocityVerlet

ROOT = Path(__file__).parent.parent.parent
BASELINE = Path(__file__).parent / 'baseline.npz'
DOCUMENTS = [ROOT / 'README.md', ROOT / 'docs' / 'overview' / 'user_guide.md']
CODE_BLOCK = re.compile(r'```python\n(.*?)```', re.DOTALL)
NUMBER = re.compile(r'-?\d+(?:\.\d+)?(?:e[-+]?\d+)?')
PRECISE = ScipyODE(rtol=1e-10, atol=1e-12)


def _lattice(n: int, spacing: float = 1.2, offset: float = 0.6) -> np.ndarray:
    """Return the positions of a jittered square lattice of `n` points and the side of its box."""
    side = int(np.ceil(np.sqrt(n)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)[:n]
    return grid * spacing + offset + np.random.default_rng(0).normal(0, 0.05, (n, 2))


def _gas(interactions: Callable[[float], list], n: int = 64) -> System:
    """Return a group of particles on a lattice with seeded velocities and interactions built from the box size."""
    size = int(np.ceil(np.sqrt(n))) * 1.2 + 0.6
    velocities = np.random.default_rng(1).normal(0, 1, (n, 2))
    return System(
        elements=[Particle.group('gas', n, x=_lattice(n), v=velocities)],
        interactions=interactions(size),
        space=EuclideanSpace(2),
    )


def _states(system: System, until: float, **options: object) -> np.ndarray:
    """Return the state vectors of a run."""
    return Simulation(system, **options).run(until=until).trajectory_.states


def _agents(system: System, until: int, group: str, variable: str, **options: object) -> np.ndarray:
    """Return a state of every agent of a group at the last frame, sorted by agent id."""
    snapshot = Simulation(system, **options).run(until=until).trajectory_.agents(group, until)
    return np.asarray(snapshot.values[variable], dtype=float)[np.argsort(snapshot.ids)]


def _cases() -> dict[str, Callable[[], np.ndarray]]:
    """Return the recorded cases by name."""
    space3, space2, space1 = EuclideanSpace(3), EuclideanSpace(2), EuclideanSpace(1)
    return {
        'constant_force': lambda: _states(
            System(
                elements=[Particle('p', m=2.0)],
                interactions=[ConstantForce('f', target='p', F=[4.0, 0.0, -2.0])],
                space=space3,
            ),
            2,
            dt=0.5,
        ),
        'spring': lambda: _states(
            System(
                elements=[Particle('a', x=[0.0, 0.0], v=[0.0, 1.0]), Particle('b', x=[2.0, 0.0], m=2.0)],
                interactions=[Spring('s', first='a', second='b', k=3.0, length=1.0)],
                space=space2,
            ),
            5,
            solver=PRECISE,
            dt=0.1,
        ),
        'three_bodies': lambda: _states(
            System(
                elements=[
                    Particle('a', x=[1.0, 0.0], v=[0.0, 0.5]),
                    Particle('b', x=[-0.5, 0.87], v=[-0.43, -0.25]),
                    Particle('c', x=[-0.5, -0.87], v=[0.43, -0.25]),
                ],
                interactions=[
                    Gravity('ab', first='a', second='b', G=1.0),
                    Gravity('bc', first='b', second='c', G=1.0),
                    Gravity('ac', first='a', second='c', G=1.0),
                ],
                space=space2,
            ),
            3,
            solver=PRECISE,
            dt=0.1,
        ),
        'drag_and_trap': lambda: _states(
            System(
                elements=[Particle('p', x=1.0, v=1.0)],
                interactions=[
                    LinearDrag('air', target='p', gamma=0.5),
                    HarmonicPotential('trap', target='p', k=2.0, center=0.5),
                ],
                space=space1,
            ),
            5,
            solver=PRECISE,
            dt=0.25,
        ),
        'lotka_volterra': lambda: _states(
            System(
                elements=[Population('rabbits', n=40), Population('foxes', n=9)],
                interactions=[
                    LogisticGrowth('growth', target='rabbits', r=1.1, K=1000),
                    Mortality('death', target='foxes', d=0.4),
                    Predation('hunt', predator='foxes', prey='rabbits', a=0.4, b=0.1),
                ],
            ),
            20,
            solver=PRECISE,
            dt=0.5,
        ),
        'herds': lambda: _states(
            System(
                elements=[Population.group('herds', 5, n=np.linspace(1.0, 50.0, 5))],
                interactions=[ExponentialGrowth('g', target='herds', r=0.1)],
            ),
            5,
            solver=PRECISE,
        ),
        'epidemic_deterministic': lambda: _states(_epidemic(stochastic=False), 50, solver=PRECISE),
        'epidemic_stochastic': lambda: _states(_epidemic(stochastic=True), 50, random_state=3),
        'solow': lambda: _states(
            System(elements=[Economy('country')], interactions=[SolowGrowth('growth', target='country')]),
            100,
            solver=PRECISE,
            dt=10.0,
        ),
        'cobweb': lambda: _states(
            System(elements=[Market('m', price=1.0)], interactions=[Cobweb('c', target='m', a=10, b=2, c=1, d=1)]),
            30,
        ),
        'asset': lambda: _states(
            System(
                elements=[Asset('stock', price=100.0)],
                interactions=[GeometricBrownianMotion('gbm', target='stock', mu=0.05, sigma=0.2)],
            ),
            5,
            dt=0.1,
            random_state=7,
        ),
        'ideal_gas': lambda: _states(ideal_gas(200, random_state=0), 2, dt=0.1),
        'soft_spheres_numba': lambda: _states(
            _gas(
                lambda size: [
                    SoftSphere('contact', first='gas', second='gas', cutoff=1.0, k=50.0),
                    ReflectingBox('box', target='gas', size=size),
                ],
            ),
            1,
            solver=VelocityVerlet(dt=0.005),
            backend=NumbaBackend(),
            dt=0.1,
        ),
        'lennard_jones_periodic': lambda: _states(
            _gas(
                lambda size: [
                    LennardJones('lj', first='gas', second='gas', cutoff=2.5, period=size),
                    PeriodicBox('box', target='gas', size=size),
                ],
            ),
            1,
            solver=VelocityVerlet(dt=0.002),
            backend='numpy',
            dt=0.1,
        ),
        'langevin': lambda: _states(
            _gas(
                lambda size: [
                    LangevinThermostat('bath', target='gas', temperature=2.0),
                    ReflectingBox('box', target='gas', size=size),
                ],
            ),
            1,
            dt=0.1,
            random_state=0,
        ),
        'lammps': lambda: _states(
            _gas(lambda size: [LennardJones('lj', first='gas', second='gas', cutoff=2.5)]),
            1,
            solver=LammpsBackend(dt=0.005),
            dt=0.1,
        ),
        'harvesting': lambda: _states(harvesting(10, effort=0.05), 30, random_state=0),
        'wolf_sheep': lambda: _agents(wolf_sheep(10, 10, random_state=0), 20, 'grass', 'countdown', random_state=0),
        'schelling': lambda: _agents(schelling(10, 10, random_state=0), 10, 'residents', 'kind', random_state=0),
        'boltzmann': lambda: _agents(boltzmann_wealth(50), 20, 'traders', 'wealth', random_state=0),
        'voter': lambda: _agents(
            voter_model(nx.cycle_graph(20), random_state=0),
            20,
            'voters',
            'opinion',
            random_state=0,
        ),
    }


def _epidemic(*, stochastic: bool) -> System:
    """Return an SIR epidemic."""
    return System(
        elements=[Compartment('S', n=990), Compartment('I', n=10), Compartment('R')],
        interactions=[
            Infection('infection', susceptible='S', infected='I', beta=0.0005, stochastic=stochastic),
            Recovery('recovery', infected='I', recovered='R', gamma=0.1, stochastic=stochastic),
        ],
    )


def _documentation_numbers() -> dict[str, np.ndarray]:
    """Return the numbers every documentation block prints, in order, leaving out blocks that print equations."""
    rr.spawn = lambda **_: None  # documentation examples run without a viewer
    numbers = {}
    for document in DOCUMENTS:
        namespace: dict[str, object] = {}
        printed = 0
        for block in CODE_BLOCK.findall(document.read_text()):
            output = io.StringIO()
            code = compile(block, str(document), 'exec')
            with contextlib.redirect_stdout(output):
                exec(code, namespace)  # noqa: S102  # nosec B102: runs the repository's own documentation
            text = output.getvalue()
            if text and 'Eq(' not in text and '==' not in text:
                numbers[f'{document.stem}_output{printed}'] = np.array([float(token) for token in NUMBER.findall(text)])
                printed += 1
    return numbers


def results() -> dict[str, np.ndarray]:
    """Return the result of every case and documentation block by name."""
    return {name: case() for name, case in _cases().items()} | _documentation_numbers()


if __name__ == '__main__':
    np.savez_compressed(BASELINE, **results())
