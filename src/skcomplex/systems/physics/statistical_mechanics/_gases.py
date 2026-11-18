"""Statistical mechanics systems.

The module covers gases, fluids, and spin systems.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import SR, vector

from ....core.components import Observable
from ....core.components.interactions import BaseInteraction, Interaction
from ....core.components.laws import RateLaw
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, VelocityVerletSolver
from ....core.specs import Role, VisualSpec
from ....core.systems import BaseSystem, System
from ..common.particles import LangevinThermostatInteraction, LennardJonesInteraction, MassPointElement


class GasMoleculeElement(MassPointElement):
    """Molecule of an ideal gas, drawn as a point coloured by its velocity.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import GasMoleculeElement
        >>> GasMoleculeElement('molecule', m=2.0).m
        2.0
    """

    visual = VisualSpec('points', {'position': 'x', 'color_by': "x'", 'states': ()})


class FreeFlightInteraction(Interaction):
    """Free flight of molecules, which keep their velocity `x'` between the walls, `x'' = 0`.

    It is Newton's first law for the molecules of an ideal gas, for light rays in a cavity, or for billiard balls.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the molecules.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import FreeFlightInteraction
        >>> FreeFlightInteraction('flight', target='gas').target
        'gas'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the zero acceleration of the positions of the target.

        Args:
            context:
                The namespace of the target and the time `t`.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            self.laws_ = [RateLaw(x.diff(context.t, 2), vector(SR, [0] * len(x)))]


class IdealGasSystem(BaseSystem):
    """An ideal gas of non-interacting molecules in a reflecting box.

    Positions are uniform in the box and velocity components are normal with variance `kT / mass`, with the mean
    velocity removed. The molecules are drawn as points colored by speed. NOTE-RETURNS: The gas.

    Args:
        n_molecules:
            The number of molecules.

        kT:
            The temperature in energy units.

        size:
            The side of the box.

        n_dim:
            The number of dimensions.

        mass:
            The mass of every molecule.

        random_state:
            The seed or generator of the initial positions and velocities.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem
        >>> gas = IdealGasSystem(1000, random_state=0).prepare().system_.elements[0]
        >>> gas.label, type(gas).__name__, gas.size
        ('gas', 'GasMoleculeElement', 1000)
    """

    field = 'physics/statistical_mechanics'
    topics = ('gas', 'ode', 'boundary')
    references = ('Maxwell 1860',)
    until = 2.0
    dt = 0.01

    def __init__(
        self,
        n_molecules: int = 500,
        *,
        kT: float = 1.0,
        size: float = 1.0,
        n_dim: int = 2,
        mass: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_molecules = n_molecules
        self.kT = kT
        self.size = size
        self.n_dim = n_dim
        self.mass = mass
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the molecules at uniform positions with thermal velocities.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        velocities = rng.normal(0.0, np.sqrt(self.kT / self.mass), size=(self.n_molecules, self.n_dim))
        velocities -= velocities.mean(axis=0)
        box = ContinuousSpace(extent=[self.size] * self.n_dim, boundary='reflecting')
        molecules = GasMoleculeElement(
            'gas',
            x=rng.uniform(0.0, self.size, size=(self.n_molecules, self.n_dim)),
            m=self.mass,
            values={"x'": velocities},
            size=self.n_molecules,
            space=box,
        )
        flight = FreeFlightInteraction('flight', target='gas')
        self.system_ = System(self.label, elements=[molecules], interactions=[flight], space=box)


class LennardJonesFluidSystem(BaseSystem):
    """Lennard-Jones particles in a periodic square box.

    A thermostat optionally holds the particles at a temperature.

    The system has the particle group `fluid` in two dimensions.

    Args:
        n_particles:
            The number of particles.

        density:
            The number of particles per unit area.

        temperature:
            The temperature of the Langevin thermostat, or `None` for constant energy. A thermostat makes the system
            stochastic, so it runs with the Euler-Maruyama solver instead of the default velocity Verlet.

        cutoff:
            The distance beyond which pairs do not interact.

        random_state:
            The seed or generator of the initial jitter and velocities.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import LennardJonesFluidSystem
        >>> LennardJonesFluidSystem(100).prepare().system_.elements[0].size
        100
    """

    field = 'physics/statistical_mechanics'
    topics = ('molecular dynamics', 'pair potential', 'ode')
    references = ('Lennard-Jones 1924', 'Verlet 1967')
    until = 0.5
    dt = 0.005

    def __init__(
        self,
        n_particles: int = 400,
        *,
        density: float = 0.5,
        temperature: float | None = None,
        cutoff: float = 2.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.density = density
        self.temperature = temperature
        self.cutoff = cutoff
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the particles on a jittered grid, with an optional thermostat.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        size = float(np.sqrt(self.n_particles / self.density))
        side = int(np.ceil(np.sqrt(self.n_particles)))
        spacing = size / side
        grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)[: self.n_particles]
        positions = np.mod((grid + 0.5) * spacing + rng.normal(0, 0.02 * spacing, (self.n_particles, 2)), size)
        velocities = rng.normal(0, np.sqrt(self.temperature or 1.0), (self.n_particles, 2))
        velocities -= velocities.mean(axis=0)
        interactions: list[BaseInteraction] = [
            LennardJonesInteraction('lj', first='fluid', second='fluid', cutoff=self.cutoff, divisor='m', period=size),
        ]
        if self.temperature is not None:
            interactions.append(
                LangevinThermostatInteraction('thermostat', target='fluid', temperature=self.temperature),
            )
        box = ContinuousSpace(extent=(size, size), boundary='periodic')
        fluid = MassPointElement('fluid', x=positions, values={"x'": velocities}, size=self.n_particles, space=box)
        solver = VelocityVerletSolver() if self.temperature is None else 'auto'
        self.system_ = System(self.label, elements=[fluid], interactions=interactions, space=box, solver=solver)
